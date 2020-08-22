# -*- coding: utf8 -*-
# PSNR-oriented SRCNNo utputs blurry SR images.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import numpy as np
import pandas as pd
from tqdm import tqdm

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


    # Import SRCNN model
    SRCNN = Generator().cuda()
    print("SRCNN parameters:", sum(param.numel() for param in SRCNN.parameters()))
    
    # Training
    SRCNN.apply(initialize_weights)
    
    optimizer = torch.optim.Adam(SRCNN.parameters(), lr=1e-3, betas=(0.5, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma = 0.5)
                               
    iter_count = 0
    iter_per_epoch = int(51138/BATCH_SIZE)
    
    train_result = {'iter':[],  'loss':[], 'psnr':[]}
    print("Initializing Training!")

    for epoch in range(1, NUM_EPOCHS+1):   
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0, 'psnr': 0}

        SRCNN.train()
        for x, y in train_bar:
            running_results['batch_sizes'] += BATCH_SIZE         
            
            optimizer.zero_grad()
            AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(4,1,384,384)

            SR_img = SRCNN(AR_img) 
            loss = MSE_loss(SR_img, OR_img)
            loss.backward()
            optimizer.step()
            
            #loss for current batch
            running_results['loss'] += loss.item() * BATCH_SIZE
            running_results['psnr'] += calculate_psnr(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy()) * BATCH_SIZE
            
            #Save training results
            if (iter_count % iter_per_epoch < 100 and iter_count % 2 == 0) or \
                        (iter_count % iter_per_epoch >= 100 and iter_count % 100 == 0): 
                train_result['iter'].append(iter_count)
                train_result['loss'].append(running_results['loss']/running_results['batch_sizes'])
                train_result['psnr'].append(running_results['psnr']/running_results['batch_sizes'])

            #Show training results dynamically
            if iter_count % 100 == 0:
                train_bar.set_description(desc='Epoch: [%d|%d]  Iter/iter_per_epoch: %d/%d  Loss: %.6f PSNR: %.4f' % (
                epoch, NUM_EPOCHS, iter_count, iter_per_epoch, 
                running_results['loss'] / running_results['batch_sizes'],
                running_results['psnr'] / running_results['batch_sizes']))

            if iter_count % 1000 == 0:
                img_path = '../datasplit/SRCNN_train/'
                show_images(SR_img, img_path+'SR_iter/'+str(iter_count)+'.png')
               
                                         
            iter_count += 1

        #update the learning rate
        scheduler.step() 


        # Val per epoch
        print(" Val in epoch%d " % epoch)
        with torch.no_grad():
            val_qua = {'mean': 0, 'var': 0, 'psnr': 0, 'ssim': 0}
            SR_imgs = torch.Tensor([]).cuda()
            SRCNN.eval()
            for x, y in val_loader:
                iter_count += 1
                AR_img = (x.unsqueeze(1)).type(dtype) #(4,1,384,384)
                SR_img = SRCNN(AR_img)

                SR_img = SR_img.squeeze(1)
                SR_imgs = torch.cat((SR_imgs, SR_img),0)

            SR_path = "../datasplit/test/1/SRCNN/SR%d" % epoch
            SR = stitch_patch(SR_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64) #(BATCH_SIZE,1,384,384)
            val_qua['mean'], val_qua['var'] = SR.mean(), SR.var()
            val_qua['psnr'], val_qua['ssim'] = calculate_psnr(SR, val_OR), calculate_ssim(SR, val_OR)
            print("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f" % (val_qua['mean'], \
                                        val_qua['var'], val_qua['psnr'], val_qua['ssim']))
                                        
        # Save model per epoch
        SRCNN_path = "../datasplit/SRCNN_train/model%d.pkl" % epoch
        torch.save(SRCNN.state_dict(), SRCNN_path)                  
        print("SRCNN model saved in %dth epoch!\n" % epoch)

        # save train_result per epoch
        stat_path = '../SRCNN_train_result.csv'
        data_frame = pd.DataFrame(
            data = {'Iters':train_result['iter'], 'Loss':train_result['loss'], 'PSNR':train_result['psnr']})
        data_frame.to_csv(stat_path, index=False, sep=',')

    print("Training Done!")


            
                
    

    

            



