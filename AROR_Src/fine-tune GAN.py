# -*- coding: utf8 -*-
# Fine-tune the Generator of GAN model using pre-trained PSNR-oriented SRCNN

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
from SRGAN_V2 import *

 
if __name__ == "__main__":
    # SetUp
    #dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor
    BATCH_SIZE = 2
    NUM_EPOCHS = 10

    # Load data
    train_set = DatasetFromFolder('../datasplit/new_train/x', '../datasplit/new_train/y')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=BATCH_SIZE)
    val_set = DatasetFromFolder('../datasplit/test/1/x', '../datasplit/test/1/y')
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=BATCH_SIZE)
    val_OR = np.load("../datasplit/test/1/OR.npy")
    val_OR = val_OR[0:6*384-5*64, 0:6*384-5*64]


    # Import SRGAN model
    G = Generator().cuda()
    print('# generator parameters:', sum(param.numel() for param in G.parameters()))
    D = Discriminator().cuda()
    print('# discriminator parameters:', sum(param.numel() for param in D.parameters()))

    # Training
    G.load_state_dict(torch.load('../datasplit/SRCNN_train/model8.pkl'))
    D.load_state_dict(torch.load('../datasplit/new_train/D_model9.pkl'))
    
    optimizerG, optimizerD = torch.optim.Adam(G.parameters(), lr=2.5e-4, betas=(0.5, 0.999)), \
                torch.optim.Adam(D.parameters(), lr=6.25e-6, betas=(0.5, 0.999))
    schedulerG, schedulerD = lr_scheduler.StepLR(optimizerG, step_size=4, gamma = 0.5), \
                lr_scheduler.StepLR(optimizerD, step_size=2, gamma = 0.5)
                               
    iter_count = 0
    iter_per_epoch = int(51138/BATCH_SIZE)
    
    train_result = {'iter':[], 'd_loss':[], 'g_loss':[], 'pcc':[]}
    print("Initializing Training!")

    for epoch in range(1, NUM_EPOCHS+1):   
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'pcc': 0}

        G.train()
        D.train()
        for x, y in train_bar:
            running_results['batch_sizes'] += BATCH_SIZE          

            # train Discriminator
            optimizerD.zero_grad()
            AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(4,1,384,384)
            SR_img = G(AR_img).detach()
            logits_SR, logits_OR = D(SR_img), D(OR_img)
            D_loss = discriminator_loss(logits_OR, logits_SR)
            D_loss.backward()
            optimizerD.step()
            

            # train Generator
            optimizerG.zero_grad()
            SR_img = G(AR_img)
            logits_SR = D(SR_img) 
            G_loss = tine_G_loss(logits_SR, OR_img, SR_img)
            G_loss.backward()
            optimizerG.step()
            

            #loss for current batch
            running_results['g_loss'] += G_loss.item() * BATCH_SIZE
            running_results['d_loss'] += D_loss.item() * BATCH_SIZE    
            running_results['d_score'] += (torch.sigmoid(logits_OR).mean()).item() * BATCH_SIZE
            running_results['g_score'] += (torch.sigmoid(logits_SR).mean()).item() * BATCH_SIZE
            running_results['pcc'] += calculate_pcc(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy()) * BATCH_SIZE

            #Save training results
            if (iter_count % iter_per_epoch < 100 and iter_count % 2 == 0) or \
                        (iter_count % iter_per_epoch >= 100 and iter_count % 100 == 0): 
                train_result['iter'].append(iter_count)
                train_result['d_loss'].append(running_results['d_loss']/running_results['batch_sizes'])
                train_result['g_loss'].append(running_results['g_loss']/running_results['batch_sizes'])
                train_result['pcc'].append(running_results['pcc']/running_results['batch_sizes'])

            #Show training results dynamically
            if iter_count % 100 == 0:
                train_bar.set_description(desc='Epoch: [%d|%d]  Iter/iter_per_epoch: %d/%d  Loss_D: %.6f Loss_G: %.6f \
D(x): %.4f D(G(z)): %.4f  PCC: %.4f' % (epoch, NUM_EPOCHS, iter_count, iter_per_epoch, 
                running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes'],
                running_results['pcc'] / running_results['batch_sizes']))

            if iter_count % 1000 == 0:
                img_path = '../datasplit/new_train/'
                show_images(SR_img, img_path+'SR_iter/'+str(iter_count)+'.png')
                # show_images(AR_img, img_path+'AR_iter/'+str(iter_count)+'.png')
                # show_images(OR_img, img_path+'OR_iter/'+str(iter_count)+'.png')
                
                         
            iter_count += 1

        #update the learning rate
        schedulerD.step() 
        schedulerG.step()

        # Val per epoch
        print(" Val in epoch%d " % epoch)
        with torch.no_grad():
            val_qua = {'mean': 0, 'var': 0, 'psnr': 0, 'pcc': 0}
            SR_imgs = torch.Tensor([]).cuda()
            G.eval()
            for x, y in val_loader:
                iter_count += 1
                AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(4,1,384,384)
                SR_img = G(AR_img)

                SR_img = SR_img.squeeze(1)
                SR_imgs = torch.cat((SR_imgs, SR_img),0)

            SR_path = "../datasplit/test/1/SRGAN/SR%d" % epoch
            SR = stitch_patch(SR_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64) #(BATCH_SIZE,1,384,384)
            val_qua['mean'], val_qua['var'] = SR.mean(), SR.var()
            val_qua['psnr'], val_qua['pcc'] = calculate_psnr(SR, val_OR), calculate_pcc(SR, val_OR)
            print("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, PCC: %.4f" % (val_qua['mean'], \
                                        val_qua['var'], val_qua['psnr'], val_qua['pcc']))

        # Save model per epoch
        D_path, G_path = "../datasplit/new_train/fine-tuneD_model%d.pkl" % epoch, "../datasplit/new_train/fine-tuneG_model%d.pkl" % epoch
        torch.save(D.state_dict(), D_path)  
        torch.save(G.state_dict(), G_path)                 
        print("SRGAN model saved in %dth epoch!\n" % epoch)

        # save train_result per epoch
        stat_path = '../fine-tune_SRGAN_train_result.csv'
        data_frame = pd.DataFrame(
            data = {'Iters':train_result['iter'], 'Loss_D':train_result['d_loss'],
                    'Loss_G':train_result['g_loss'], 'PCC':train_result['pcc']})
        data_frame.to_csv(stat_path, index=False, sep=',')

    print("Training Done!")


            
                
    

    

            



