# -*- coding: utf8 -*-
# Adjustments(July 14)
# (1) Mixed Precision SRGAN Training in PyTorch (failed to install Apex)
# (2) loss_G = MAE+lambda1*FMAE+20%*SquCE
# (3) AdamW: Decoupled weight decay(0.01) Adam

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import numpy as np
import pandas as pd
from tqdm import tqdm

import pytorch_ssim
from utils import *
from SRGAN_V7 import *
# from apex import amp


'''
basic train -> esemble
G_lr_init, G_lr_end = 2e-3, 0
D_lr_init, D_lr_end = 4e-5, 0

12 epochs
Cosine Annealing Lr Scheduling, iter per epoch
G_loss = MSE+5e-3*BCE+2e-2*SSIM
G_loss = MSE+TV(2%)+SquCE(20%)

# Adjustments(July 14)
# (1) Mixed Precision SRGAN Training in PyTorch (failed to install Apex)
# (2) loss_G = MAE+lambda1*FMAE+20%*SquCE
# (3) AdamW: Decoupled weight decay(0.01) Adam
'''

if __name__ == "__main__":

    # Hyper parameter
    dtype = torch.cuda.FloatTensor
    BATCH_SIZE = 2

    # Load data
    train_set = DatasetFromFolder('../datasplit/new_train/x', '../datasplit/new_train/y')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=4)
    val_set = DatasetFromFolder('../datasplit/test/7th/x', '../datasplit/test/7th/y')
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=4)
    val_OR = np.load("../datasplit/test/7th/OR.npy")
    val_OR = val_OR[0:6*384-5*64, 0:6*384-5*64]

    ITER_PER_EPOCH = len(train_loader)
    NUM_EPOCHS = 2

    # Import SRGAN model
    G = Generator().cuda()
    D = Discriminator().cuda()
    lambds = ['bce']  #1e-1,1e-2,1e-3
    for lambd in lambds:
        # Small weight initialization
        # G.apply(G_initweights)
        # D.apply(D_initweights)
        G.load_state_dict(torch.load('../datasplit/new_train/train7/BCE/G_model1.pkl'))
        D.load_state_dict(torch.load('../datasplit/new_train/train7/BCE/D_model1.pkl'))
            
        
        G_lr_init, G_lr_end = 2e-3, 0
        D_lr_init, D_lr_end = 4e-6, 0

        optimizerG, optimizerD = torch.optim.AdamW(G.parameters(), lr=G_lr_init, betas=(0.5, 0.999)), \
                    torch.optim.AdamW(D.parameters(), lr=D_lr_init, betas=(0.5, 0.999)) #Weight_decay=1e-2

        # schedulerG, schedulerD = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=NUM_EPOCHS, eta_min=G_lr_end), \
        #                 lr_scheduler.CosineAnnealingLR(optimizerD, T_max=NUM_EPOCHS, eta_min=D_lr_end)

        # # Changes invoked by Apex.amp
        # # To enable mixed precision training, we introduce the --opt_level argument.
        # [G,D], [optimizerG, optimizerD] = amp.initialize(
        #     [G,D], [optimizerG, optimizerD], opt_level="O1", num_losses=2) #recommended mixed precision

        #to record iter times in the total training
        train_result = {'iter':[], 'd_loss':[], 'g_loss':[], 'psnr':[], 'g_lr':[], 'd_lr':[]}

        print("Initializing Training!")  
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'psnr': 0}
        iters = 1+25569
        
        for epoch in range(2, 1+NUM_EPOCHS): #NUM_EPOCHS
            G.train()
            D.train()
            with tqdm(train_loader, ncols=220) as train_bar:  #tqdm滚动条不能单行显示，需要人为规定长度；以及在进程异常情况下设置tqdm完全退出
                for x, y in train_bar:  

                    # Update lr every iteration
                    G_lr = snapshot_lr(G_lr_init, G_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)
                    D_lr = snapshot_lr(D_lr_init, D_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)        
                    optimizerG.param_groups[0]['lr'] = G_lr
                    optimizerD.param_groups[0]['lr'] = D_lr

                    # train Discriminator
                    optimizerD.zero_grad()
                    D_loss = 0.0
                    AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)           
                    try:
                        SR_img = G(AR_img).detach()
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
                    # # Changes invoked by Apex.amp
                    # with amp.scale_loss(D_loss, optimizerD, loss_id=0) as errD_scaled:
                    #     errD_scaled.backward()
                    #nn.utils.clip_grad_norm_(D.parameters(), max_norm=20, norm_type=2)
                    # clip_grad_value_(D.parameters(), 20)  
                    # test_grad_value(D.parameters(), optimizerD)
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
                    logits_SR, logits_OR = D(SR_img), D(OR_img)
                    G_loss, InteMAE, BCE = generator_loss(logits_SR, OR_img, SR_img, 0.01)
                    G_loss.backward()
                    # # Changes invoked by Apex.amp
                    # with amp.scale_loss(G_loss, optimizerG, loss_id=1) as errG_scaled:
                    #     errG_scaled.backward()
                    #nn.utils.clip_grad_norm_(G.parameters(), max_norm=20, norm_type=2)
                    test_grad_value(G.parameters(), optimizerG)
                    # optimizerG.step()

                    #loss for current batch
                    running_results['batch_sizes'] += BATCH_SIZE
                    running_results['g_loss'] += G_loss.item() * BATCH_SIZE
                    running_results['d_loss'] += D_loss.item() * BATCH_SIZE 
                    running_results['d_score'] += torch.sigmoid(logits_OR.mean()).item() * BATCH_SIZE
                    running_results['g_score'] += torch.sigmoid(logits_SR.mean()).item() * BATCH_SIZE   
                    running_results['psnr'] += calculate_psnr(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy()) * BATCH_SIZE

                    #Save training results
                    if (iters % ITER_PER_EPOCH <= 100 and iters % 2 == 0) or \
                                (iters % ITER_PER_EPOCH > 100 and iters % 100 == 0):
                        d_loss = running_results['d_loss'] / running_results['batch_sizes']
                        g_loss = running_results['g_loss'] / running_results['batch_sizes']
                        d_score = running_results['d_score'] / running_results['batch_sizes']
                        g_score = running_results['g_score'] / running_results['batch_sizes']
                        psnr = running_results['psnr'] / running_results['batch_sizes']

                        train_result['iter'].append(iters)
                        train_result['d_loss'].append(d_loss)
                        train_result['g_loss'].append(g_loss)
                        train_result['psnr'].append(psnr)
                        train_result['g_lr'].append(G_lr)
                        train_result['d_lr'].append(D_lr)

                        #Show training results within a epoch dynamically
                        train_bar.set_description(desc="Epoch|NUM_EPOCHS: %d/%d, Iter/Iter_num: %d/%d, \
Loss_D:%.6f, Loss_G:%.6f, D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f, G_lr:%.4e, D_lr:%.4e, InteMAE:%.4f, BCE:%.4f " % ( epoch, NUM_EPOCHS, \
iters, NUM_EPOCHS*ITER_PER_EPOCH, d_loss, g_loss, d_score, g_score, psnr, G_lr, D_lr, InteMAE, BCE))  
            
                    if (iters % ITER_PER_EPOCH) % 1000 == 0:
                        img_path = '../datasplit/new_train/train7/'
                        show_images(SR_img, img_path+'%s/SR_iter/'%lambd+str(iters)+'.png')
                        show_images(AR_img, img_path+'%s/AR_iter/'%lambd+str(iters)+'.png')
                        show_images(OR_img, img_path+'%s/OR_iter/'%lambd+str(iters)+'.png')
                        # show_images(SR_img, img_path+'SR_iter/'+str(iters)+'.png')
                        # show_images(AR_img, img_path+'AR_iter/'+str(iters)+'.png')
                        # show_images(OR_img, img_path+'OR_iter/'+str(iters)+'.png')
                                                    
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

            SR_path = "../datasplit/new_train/train7/%s/SR%d" % (lambd, epoch)
            SR = stitch_patch(SR_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64) 
            val_qua['mean'], val_qua['var'] = SR.mean(), SR.var()
            val_qua['psnr'] = calculate_psnr(SR, val_OR) 
            val_qua['ssim'], val_qua['pcc'] = calculate_ssim(SR, val_OR), calculate_pcc(SR, val_OR)
            print("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f, PCC: %.4f" % (val_qua['mean'], \
                                val_qua['var'], val_qua['psnr'], val_qua['ssim'], val_qua['pcc']))

            # #update the learning rate
            # schedulerD.step() 
            # schedulerG.step()

            # Save model per epoch
            D_path, G_path = "../datasplit/new_train/train7/%s/D_model%d.pkl" % (lambd,epoch), "../datasplit/new_train/train7/%s/G_model%d.pkl" % (lambd,epoch)
            torch.save(D.state_dict(), D_path)  
            torch.save(G.state_dict(), G_path)                 
            print("Save SRGAN model in %dth epoch!\n" % epoch)

            # save train_result per epoch
            stat_path = '../datasplit/new_train/train7/%s/SRGAN_basictrain.csv' % lambd
            data_frame = pd.DataFrame(
            data = {'Iters':train_result['iter'], 'Loss_D':train_result['d_loss'],'Loss_G':train_result['g_loss'],  
                    'PSNR':train_result['psnr'], 'G_lr':train_result['g_lr'], 'D_lr':train_result['d_lr']})
            data_frame.to_csv(stat_path, index=False, sep=',')

    print("training Done!")
     



