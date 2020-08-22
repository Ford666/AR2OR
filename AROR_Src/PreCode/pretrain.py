# -*- coding: utf8 -*-
# Perceptual-driven SRGAN outputs produces SR images with sharp edges 
# and richer textures but with some unpleasant artifacts.

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
from SRGAN_V3 import *

# pre-train: AR-AR and OR-OR
# 1epoch, cosine annealing scheduler
 
if __name__ == "__main__":

    # Hyper parameter
    dtype = torch.cuda.FloatTensor
    NUM_EPOCHS, BATCH_SIZE = 1, 2

    # Load data
    train_set = DatasetFromFolder('../datasplit/new_train/y', '../datasplit/new_train/y')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=4)
    val_set = DatasetFromFolder('../datasplit/test/1/y', '../datasplit/test/1/y')
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=4)
    val_OR = np.load("../datasplit/test/1/OR.npy")
    val_OR = val_OR[0:6*384-5*64, 0:6*384-5*64]

    ITER_PER_EPOCH = len(train_loader)


    # Import SRGAN model
    G = Generator().cuda()
    print('# generator parameters:', sum(param.numel() for param in G.parameters()))
    D = Discriminator().cuda()
    print('# discriminator parameters:', sum(param.numel() for param in D.parameters()))
    
    # Vertify model
    data = next(enumerate(train_loader))
    ipt, target = data[-1][0].unsqueeze(1).type(dtype), data[-1][1].unsqueeze(1).type(dtype) #(BATCH_SIZE,384,384)
    show_images(ipt, '../demo_AR.png')
    show_images(target, '../demo_OR.png')
    print(ipt.size())
    outG, outD = G(ipt), D(target)   
    print(outG.size(), outD.size())

    # Training
    G.apply(G_initweights)
    D.apply(D_initweights)
    # G.load_state_dict(torch.load('../datasplit/pretrain/OR-OR/G_model1.pkl'))
    # D.load_state_dict(torch.load('../datasplit/pretrain/OR-OR/D_model1.pkl'))


    G_lr_init, G_lr_end = 2e-3, 0
    D_lr_init, D_lr_end = 4e-5, 0
    
    optimizerG, optimizerD = torch.optim.Adam(G.parameters(), lr=G_lr_init, betas=(0.5, 0.999)), \
                torch.optim.Adam(D.parameters(), lr=D_lr_init, betas=(0.5, 0.999))
                               
    #to record iter times in the total training
    train_result = {'iter':[], 'd_loss':[], 'g_loss':[], 'psnr':[]}
    iter_count = 1 

    print("Initializing Pre-training!")
    
    for epoch in range(1, NUM_EPOCHS+1):   
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'psnr': 0}
        iters = 1
        G.train()
        D.train()
        for x, y in train_bar:
            running_results['batch_sizes'] += BATCH_SIZE
            
            # Update lr every iteration
            G_lr = snapshot_lr(G_lr_init, G_lr_end, iters, ITER_PER_EPOCH)
            D_lr = snapshot_lr(D_lr_init, D_lr_end, iters, ITER_PER_EPOCH)        
            optimizerG.param_groups[0]['lr'] = G_lr
            optimizerD.param_groups[0]['lr'] = D_lr          

            # train Discriminator
            optimizerD.zero_grad()
            AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)
            SR_img = G(AR_img).detach() #without detach(), training time would increase greatly!
            logits_SR, logits_OR = D(SR_img), D(OR_img)
            D_loss = discriminator_loss(logits_OR, logits_SR)
            D_loss.backward()
            optimizerD.step()
            

            # train Generator
            optimizerG.zero_grad()
            SR_img = G(AR_img)
            logits_SR = D(SR_img) 
            G_loss = generator_loss(logits_SR, OR_img, SR_img)
            G_loss.backward()
            optimizerG.step()
            

            #loss for current batch
            running_results['g_loss'] += G_loss.item() * BATCH_SIZE
            running_results['d_loss'] += D_loss.item() * BATCH_SIZE    
            running_results['d_score'] += (torch.sigmoid(logits_OR).mean()).item() * BATCH_SIZE
            running_results['g_score'] += (torch.sigmoid(logits_SR).mean()).item() * BATCH_SIZE
            running_results['psnr'] += calculate_psnr(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy()) * BATCH_SIZE

            #Save training results
            if (iters % ITER_PER_EPOCH <= 100 and iters % 2 == 0) or \
                        (iters % ITER_PER_EPOCH > 100 and iters % 100 == 0):
                d_loss = running_results['d_loss'] / running_results['batch_sizes']
                g_loss = running_results['g_loss'] / running_results['batch_sizes']
                d_score = running_results['d_score'] / running_results['batch_sizes']
                g_score = running_results['g_score'] / running_results['batch_sizes']
                psnr = running_results['psnr'] / running_results['batch_sizes']

                train_result['iter'].append(iter_count)
                train_result['d_loss'].append(d_loss)
                train_result['g_loss'].append(g_loss)
                train_result['psnr'].append(psnr)

                #Show training results within a epoch dynamically
                train_bar.set_description(desc="Epoch: [%d|%d], Iter/Iter_per_epoch: %d/%d, \
Loss_D:%.6f, Loss_G:%.6f, D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f, G_lr:%.4e, D_lr:%.4e" % (epoch, NUM_EPOCHS, 
iters, ITER_PER_EPOCH, d_loss, g_loss, d_score, g_score, psnr, G_lr, D_lr))

            if  (iter_count % ITER_PER_EPOCH) % 1000 == 0:
                img_path = '../datasplit/pretrain/OR-OR/'
                show_images(SR_img, img_path+'SR_iter/'+str(iter_count)+'.png')
                show_images(AR_img, img_path+'AR_iter/'+str(iter_count)+'.png')
                show_images(OR_img, img_path+'OR_iter/'+str(iter_count)+'.png')
                                        
            iters +=1
            iter_count += 1

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

        SR_path = "../datasplit/test/1/pretrain/OR-OR/SR_epoch%d" % epoch
        SR = stitch_patch(SR_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64) 
        val_qua['mean'], val_qua['var'] = SR.mean(), SR.var()
        val_qua['psnr'] = calculate_psnr(SR, val_OR) 
        val_qua['ssim'], val_qua['pcc'] = calculate_ssim(SR, val_OR), calculate_pcc(SR, val_OR)
        print("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f, PCC: %.4f" % (val_qua['mean'], \
                            val_qua['var'], val_qua['psnr'], val_qua['ssim'], val_qua['pcc']))

        # Save model per epoch
        D_path, G_path = "../datasplit/pretrain/OR-OR/D_model%d.pkl" % epoch, "../datasplit/pretrain/OR-OR/G_model%d.pkl" % epoch
        torch.save(D.state_dict(), D_path)  
        torch.save(G.state_dict(), G_path)                 
        print("Take a snapshot of SRGAN in %dth epoch!\n" % epoch)

        # save train_result per epoch
        stat_path = '../datasplit/pretrain/OR-OR/SRGAN_pretrain.csv'
        data_frame = pd.DataFrame(
        data = {'Iters':train_result['iter'], 'Loss_D':train_result['d_loss'],'Loss_G':train_result['g_loss'],  
                            'PSNR':train_result['psnr']})
        data_frame.to_csv(stat_path, index=False, sep=',')

    print("preliminary training Done!")


            
                
    

    

            



