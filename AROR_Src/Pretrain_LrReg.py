# -*- coding: utf8 -*-

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


# Use the validation set to tune the learning rate and regularization strength
# train 2 epochs

if __name__ == "__main__":

    # Hyper parameter
    dtype = torch.cuda.FloatTensor
    BATCH_SIZE = 2

    # Load data
    train_set = DatasetFromFolder('../datasplit/new_train/x', '../datasplit/new_train/y')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=4)
    val_set = DatasetFromFolder('../datasplit/test/1/x', '../datasplit/test/1/y')
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=4)
    val_OR = np.load("../datasplit/test/1/OR.npy")
    val_OR = val_OR[0:6*384-5*64, 0:6*384-5*64]

    ITER_PER_EPOCH = len(train_loader)
    NUM_EPOCHS = 1

    # Import SRGAN model
    G = Generator().cuda()
    print('# generator parameters:', sum(param.numel() for param in G.parameters()))
    D = Discriminator().cuda()
    print('# discriminator parameters:', sum(param.numel() for param in D.parameters()))

    # Training
    G.apply(G_initweights)
    D.apply(D_initweights)
         
    
    # Set the learning rate and regularization strength
    G_lr_init, G_lr_end = [2e-3, 5e-3, 1e-2], 0
    D_lr_init, D_lr_end = [4e-5, 1e-4, 2e-4], 0
    lambda1, lambda2 = [5e-3, 1e-2], [1e-3, 1e-2]

    best_comb = {"G_lr_init": 0, "D_lr_init": 0, "lambda1": 0, "lambda2": 0}
    best_pcc = -1

    comb = 1
    for idx in range(len(G_lr_init)):
        for lba1 in lambda1:
            for lba2 in lambda2:
                optimizerG, optimizerD = torch.optim.Adam(G.parameters(), lr=G_lr_init[idx], betas=(0.5, 0.999)), \
                            torch.optim.Adam(D.parameters(), lr=D_lr_init[idx], betas=(0.5, 0.999))

                print("Initializing preliminary training!")

                running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'psnr': 0}
                iters = 1
                G.train()
                D.train() 

                for epoch in range(1, NUM_EPOCHS+1):
                    train_bar = tqdm(train_loader)
                    for x, y in train_bar:

                        # Update lr every iteration
                        G_lr = snapshot_lr(G_lr_init[idx], G_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)
                        D_lr = snapshot_lr(D_lr_init[idx], D_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)        
                        optimizerG.param_groups[0]['lr'] = G_lr
                        optimizerD.param_groups[0]['lr'] = D_lr

                        # train Discriminator
                        optimizerD.zero_grad()
                        AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)
                        SR_img = G(AR_img).detach()
                        logits_SR, logits_OR = D(SR_img), D(OR_img)
                        D_loss = discriminator_loss(logits_OR, logits_SR)
                        D_loss.backward()
                        optimizerD.step()
                    
                        # train Generator
                        optimizerG.zero_grad()
                        SR_img = G(AR_img)
                        logits_SR = D(SR_img) 
                        G_loss = generator_loss(logits_SR, OR_img, SR_img, lba1, lba2)
                        G_loss.backward()
                        optimizerG.step()
                        
                        #loss for current batch
                        running_results['batch_sizes'] += BATCH_SIZE
                        running_results['g_loss'] += G_loss.item() * BATCH_SIZE
                        running_results['d_loss'] += D_loss.item() * BATCH_SIZE 
                        running_results['d_score'] += (torch.sigmoid(logits_OR).mean()).item() * BATCH_SIZE
                        running_results['g_score'] += (torch.sigmoid(logits_SR).mean()).item() * BATCH_SIZE   
                        running_results['psnr'] += calculate_psnr(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy()) * BATCH_SIZE

                        #Save training results
                        if (iters % NUM_EPOCHS*ITER_PER_EPOCH <= 100 and iters % 2 == 0) or \
                                    (iters % NUM_EPOCHS*ITER_PER_EPOCH > 100 and iters % 100 == 0):
                            d_loss = running_results['d_loss'] / running_results['batch_sizes']
                            g_loss = running_results['g_loss'] / running_results['batch_sizes']
                            d_score = running_results['d_score'] / running_results['batch_sizes']
                            g_score = running_results['g_score'] / running_results['batch_sizes']
                            psnr = running_results['psnr'] / running_results['batch_sizes']

                            #Show training results within a cycle dynamically
                            train_bar.set_description(desc="Comb|Comb_num: %d/12, Iter/Iter_num: %d/%d, \
Loss_D:%.6f, Loss_G:%.6f, D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f, G_lr:%.4e, D_lr:%.4e" % (comb,
iters, NUM_EPOCHS*ITER_PER_EPOCH, d_loss, g_loss, d_score, g_score, psnr, G_lr, D_lr))
                                                        
                        iters = iters+1

                # Val per comb
                print(" Val in comb%d " % comb)
                with torch.no_grad():
                    val_qua = {'mean': 0, 'var': 0, 'psnr': 0, 'ssim': 0, 'pcc': 0}
                    SR_imgs = torch.Tensor([]).cuda()
                    G.eval()
                    for x, y in val_loader:
                        AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) 
                        SR_img = G(AR_img)

                        SR_img = SR_img.squeeze(1)
                        SR_imgs = torch.cat((SR_imgs, SR_img),0)

                SR_path = "../datasplit/new_train/pre_train/SR_comb%d" % comb
                SR = stitch_patch(SR_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64) 
                val_qua['mean'], val_qua['var'] = SR.mean(), SR.var()
                val_qua['psnr'] = calculate_psnr(SR, val_OR) 
                val_qua['ssim'], val_qua['pcc'] = calculate_ssim(SR, val_OR), calculate_pcc(SR, val_OR)
                print("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f, PCC: %.4f" % (val_qua['mean'], \
                                    val_qua['var'], val_qua['psnr'], val_qua['ssim'], val_qua['pcc']))
                
                if val_qua['pcc'] > best_pcc:
                    best_comb["G_lr_init"], best_comb["D_lr_init"] = G_lr_init[idx], D_lr_init[idx]
                    best_comb["lambda1"], best_comb["lambda2"] = lba1, lba2

                comb = comb+1

    print("The best combination is: G_lr_init=%.2e, D_lr_init=%.2e, lambda1=%.2e, lambda2=%.2e" % (
    best_comb["G_lr_init"], best_comb["D_lr_init"], best_comb["lambda1"], best_comb["lambda2"]))

    print("preliminary training Done!")


      
    

    

            



