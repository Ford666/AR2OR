# -*- coding: utf8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import numpy as np
import pandas as pd
from progressbar import *
import pytorch_ssim
from utils import *
from SRGAN_V7 import *


'''
basic train -> esemble
12 epochs
Cosine Annealing Lr Scheduling, iter per epoch
G_loss = MSE+5e-3*BCE+2e-2*SSIM
G_loss = MSE+TV(2%)+SquCE(20%)

# Adjustments(July 14)
# (1) Mixed Precision SRGAN Training in PyTorch (failed to install Apex)
# (2) loss_G = MAE+lambda1*FMAE+20%*SquCE
# (3) AdamW: Decoupled weight decay(0.01) Adam
# Adjustments(Aug 5)
(1) After fine-tuning Lr via basic_trainTuneLr, we determine G_lr_init=0.0001, D_lr_init=0.0001
(2) 报错行直接插入 import ipdb; ipdb.set_trace()即可断点调试
'''

if __name__ == "__main__":

    # Hyper parameter
    dtype = torch.cuda.FloatTensor
    BATCH_SIZE = 2

    # Load data
    train_set = DatasetFromFolder('../Augdata/ARnpy', '../Augdata/ORnpy')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=4)
    val_set = DatasetFromFolder('../datasplit/test/7th/x', '../datasplit/test/7th/y')
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=4)
    val_OR = np.load("../datasplit/test/7th/OR.npy")
    val_OR = val_OR[0:6*384-5*64, 0:6*384-5*64]

    ITER_PER_EPOCH = len(train_loader)
    NUM_EPOCHS = 12

    #Make records
    RecFile = open('../datasplit/new_train/train9/Record.txt', 'w+')
    progress = ProgressBar()  

    # Import SRGAN model
    G = Generator().cuda()
    D = Discriminator().cuda()

    # Small weight initialization
    G.apply(G_initweights)
    D.apply(D_initweights)
    # G.load_state_dict(torch.load('../datasplit/new_train/train9/BCE/G_model1.pkl'))
    # D.load_state_dict(torch.load('../datasplit/new_train/train9/BCE/D_model1.pkl'))
        
    G_lr_init, G_lr_end = 1e-4, 0
    D_lr_init, D_lr_end = 1e-4, 0

    optimizerG, optimizerD = torch.optim.AdamW(G.parameters(), lr=G_lr_init), \
                torch.optim.AdamW(D.parameters(), lr=D_lr_init) #Weight_decay=1e-2, , betas=(0.9, 0.999)

    # schedulerG, schedulerD = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=NUM_EPOCHS, eta_min=G_lr_end), \
    #                 lr_scheduler.CosineAnnealingLR(optimizerD, T_max=NUM_EPOCHS, eta_min=D_lr_end)

    print("Training Started!")
    RecFile.write("Training started!")
    RecFile.flush()  
    
    #to record iter times in the total training
    train_result = {'iter':[], 'd_loss':[], 'g_loss':[], 'DGz':[], 'Dx':[], 'psnr':[]}
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'psnr': 0}
    iters = 1


    for epoch in range(1, 1+NUM_EPOCHS): #NUM_EPOCHS
        print("Epoch|NUM_EPOCHS: %d/%d" % (epoch, NUM_EPOCHS))
        G.train()
        D.train()
        for x, y in progress(train_loader):  
            AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)           
            
            # Update lr every iteration
            G_lr = snapshot_lr(G_lr_init, G_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)
            D_lr = snapshot_lr(D_lr_init, D_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)        
            optimizerG.param_groups[0]['lr'] = G_lr
            optimizerD.param_groups[0]['lr'] = D_lr

            # train Discriminator
            optimizerD.zero_grad()
            D_loss = 0.0
            
            try:
                SR_img = G(AR_img).detach()   #detach to avoid BP to G
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: CUDA out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            logits_SR, logits_OR = D(SR_img), D(OR_img)
            D_loss = discriminator_loss(logits_OR, logits_SR)
            D_loss.backward()
            optimizerD.step()
        
            # train Generator
            optimizerG.zero_grad()       
            try: 
                SR_img = G(AR_img)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: CUDA out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception       
            logits_SR = D(SR_img)
            G_loss, InteMAE, SquCE = generator_loss(logits_SR, OR_img, SR_img, 0.01)
            G_loss.backward()
            test_grad_value(G.parameters(), optimizerG)
            # optimizerG.step()

            #loss for current batch
            running_results['batch_sizes'] += BATCH_SIZE
            running_results['g_loss'] += G_loss.item() * BATCH_SIZE
            running_results['d_loss'] += D_loss.item() * BATCH_SIZE 
            running_results['d_score'] += logits_OR.mean().item() * BATCH_SIZE
            running_results['g_score'] += logits_SR.mean().item() * BATCH_SIZE   
            running_results['psnr'] += calculate_psnr(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy()) * BATCH_SIZE

            #Save training results
            if iters % 1000 == 0:
                d_loss = running_results['d_loss'] / running_results['batch_sizes']
                g_loss = running_results['g_loss'] / running_results['batch_sizes']
                d_score = running_results['d_score'] / running_results['batch_sizes']
                g_score = running_results['g_score'] / running_results['batch_sizes']
                psnr = running_results['psnr'] / running_results['batch_sizes']

                train_result['iter'].append(iters)
                train_result['d_loss'].append(d_loss)
                train_result['g_loss'].append(g_loss)
                train_result['psnr'].append(psnr)
                train_result['DGz'].append(g_score)
                train_result['Dx'].append(d_score)

                #Show training results within a epoch dynamically
                RecFile.write("Epoch|NUM_EPOCHS: %d/%d, Iter/Iter_num: %d/%d, Loss_D:%.6f, Loss_G:%.6f, \
D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f, G_lr:%.4e, D_lr:%.4e, InteMAE:%.4f, SquCE:%.4f\n" % ( epoch, NUM_EPOCHS, \
iters, NUM_EPOCHS*ITER_PER_EPOCH, d_loss, g_loss, d_score, g_score, psnr, G_lr, D_lr, InteMAE, SquCE))
                RecFile.flush()
             
                img_path = '../datasplit/new_train/train9/'
                show_images(SR_img, img_path+'SR_iter/'+str(iters)+'.png')
                show_images(AR_img, img_path+'AR_iter/'+str(iters)+'.png')
                show_images(OR_img, img_path+'OR_iter/'+str(iters)+'.png')
                                            
            iters = iters+1

        # Val per epoch
        print(" Val in epoch%d " % epoch)
        with torch.no_grad():
            val_qua = {'mean': 0, 'var': 0, 'psnr': 0, 'ssim': 0, 'pcc': 0}
            SR_imgs = torch.Tensor([]).cuda()
            G.eval()
            for x, y in val_loader:
                AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)
                SR_img = G(AR_img)
                SR_img = SR_img.squeeze(1)
                SR_imgs = torch.cat((SR_imgs, SR_img),0)

        SR_path = "../datasplit/new_train/train9/SR%d" % epoch
        SR = stitch_patch(SR_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64) 
        val_qua['mean'], val_qua['var'] = SR.mean(), SR.var()
        val_qua['psnr'] = calculate_psnr(SR, val_OR) 
        val_qua['ssim'], val_qua['pcc'] = calculate_ssim(SR, val_OR), calculate_pcc(SR, val_OR)
        print("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f, PCC: %.4f" % (val_qua['mean'], \
                            val_qua['var'], val_qua['psnr'], val_qua['ssim'], val_qua['pcc']))
        RecFile.write("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f, PCC: %.4f\n" % (val_qua['mean'], \
                            val_qua['var'], val_qua['psnr'], val_qua['ssim'], val_qua['pcc']))

        # #update the learning rate
        # schedulerD.step() 
        # schedulerG.step()

        # Save model per epoch
        D_path, G_path = "../datasplit/new_train/train9/D_model%d.pkl" % epoch, "../datasplit/new_train/train9/G_model%d.pkl" % epoch
        torch.save(D.state_dict(), D_path)  
        torch.save(G.state_dict(), G_path)                 
        print("Save SRGAN model in %dth epoch!\n" % epoch)

        # save train_result per epoch
        stat_path = '../datasplit/new_train/train9/SRGAN_basictrain.csv'
        data_frame = pd.DataFrame(
        data = {'Iters':train_result['iter'], 'Loss_D':train_result['d_loss'],'Loss_G':train_result['g_loss'],  
                'PSNR':train_result['psnr'], 'G_lr':train_result['DGz'], 'D_lr':train_result['Dx']})
        data_frame.to_csv(stat_path, index=False, sep=',')

    print("training Done!")
     



