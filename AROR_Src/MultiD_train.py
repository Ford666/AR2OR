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
from SRGAN_V4 import *

'''
pre-train: OR-OR, for Ref

Cosine Annealing Lr Scheduling

Using multi-Stage discriminators can refine the network outputs gradually
'''

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
    NUM_EPOCHS = 12

    # Import SRGAN model
    student_G, teacher_G = Generator().cuda(), Generator().cuda()
    D = Discriminator().cuda()

    # Training
    # student_G.apply(G_initweights)
    # D.apply(D_initweights)
    teacher_G.load_state_dict(torch.load('../datasplit/pretrain/OR-OR/G_model1.pkl'))
    student_G.load_state_dict(torch.load('../datasplit/new_train/train2/G_model1.pkl'))
    D.load_state_dict(torch.load('../datasplit/new_train/train2/D_model1.pkl'))
          
    
    G_lr_init, G_lr_end = 2e-3, 0
    D_lr_init, D_lr_end = 4e-5, 0

    optimizerG, optimizerD = torch.optim.Adam(student_G.parameters(), lr=G_lr_init, betas=(0.5, 0.999)), \
                torch.optim.Adam(D.parameters(), lr=D_lr_init, betas=(0.5, 0.999))

    # schedulerG, schedulerD = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=NUM_EPOCHS, eta_min=G_lr_end), \
    #                 lr_scheduler.CosineAnnealingLR(optimizerD, T_max=NUM_EPOCHS, eta_min=D_lr_end)

    #to record iter times in the total training
    train_result = {'iter':[], 'd_loss':[], 'g_loss':[], 'psnr':[]}

    print("Initializing Training!")  
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'psnr': 0}
    iters = 1+25569
    D_lossCoeff = [1e-3, 5e-2, 1]

      
    for epoch in range(2, NUM_EPOCHS+1):
        train_bar = tqdm(train_loader)
        student_G.train()
        teacher_G.eval()
        D.train() 
        
#         for x, y in train_bar:

#             # Update lr every iteration
#             G_lr = snapshot_lr(G_lr_init, G_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)
#             D_lr = snapshot_lr(D_lr_init, D_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)        
#             optimizerG.param_groups[0]['lr'] = G_lr
#             optimizerD.param_groups[0]['lr'] = D_lr

#             # train Discriminator
#             optimizerD.zero_grad()
#             D_loss = 0.0
#             AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)
#             SR_img = student_G(AR_img)
#             SR_ref = teacher_G(OR_img)
#             for idx in range(len(SR_img)):
#                 D_loss += D_lossCoeff[idx] * discriminator_loss(D(SR_ref[idx].detach()), D(SR_img[idx].detach()))

#             D_loss.backward()
#             optimizerD.step()
        
#             # train Generator
#             optimizerG.zero_grad()
#             SR_img = student_G(AR_img)
#             logits_SR, logits_OR = D(SR_img[idx]), D(SR_ref[idx])
#             G_loss = generator_loss(logits_SR, SR_ref[idx], SR_img[idx])
#             G_loss.backward()
#             optimizerG.step()
            
#             #loss for current batch
#             running_results['batch_sizes'] += BATCH_SIZE
#             running_results['g_loss'] += G_loss.item() * BATCH_SIZE
#             running_results['d_loss'] += D_loss.item() * BATCH_SIZE 
#             running_results['d_score'] += (torch.sigmoid(logits_OR).mean()).item() * BATCH_SIZE
#             running_results['g_score'] += (torch.sigmoid(logits_SR).mean()).item() * BATCH_SIZE   
#             running_results['psnr'] += calculate_psnr(SR_img[idx].data.cpu().numpy(), SR_ref[idx].data.cpu().numpy()) * BATCH_SIZE

#             #Save training results
#             if (iters % ITER_PER_EPOCH <= 100 and iters % 2 == 0) or \
#                         (iters % ITER_PER_EPOCH > 100 and iters % 100 == 0):
#                 d_loss = running_results['d_loss'] / running_results['batch_sizes']
#                 g_loss = running_results['g_loss'] / running_results['batch_sizes']
#                 d_score = running_results['d_score'] / running_results['batch_sizes']
#                 g_score = running_results['g_score'] / running_results['batch_sizes']
#                 psnr = running_results['psnr'] / running_results['batch_sizes']

#                 #Show training results within a epoch dynamically
#                 train_bar.set_description(desc="Epoch|NUM_EPOCHS: %d/%d, Iter/Iter_num: %d/%d, \
# Loss_D:%.6f, Loss_G:%.6f, D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f, G_lr:%.4e, D_lr:%.4e" % (epoch, NUM_EPOCHS,
# iters, NUM_EPOCHS*ITER_PER_EPOCH, d_loss, g_loss, d_score, g_score, psnr, G_lr, D_lr))

#             if (iters % ITER_PER_EPOCH) % 1000 == 0:
#                 img_path = '../datasplit/new_train/train2/'
#                 show_images(SR_img[idx], img_path+'SR_iter/'+str(iters)+'.png')
#                 show_images(AR_img, img_path+'AR_iter/'+str(iters)+'.png')
#                 show_images(OR_img, img_path+'OR_iter/'+str(iters)+'.png')
                                            
#             iters = iters+1


        # Val per epoch
        print(" Val in epoch%d " % epoch)
        with torch.no_grad():
            val_qua = {'mean': 0, 'var': 0, 'psnr': 0, 'ssim': 0, 'pcc': 0}
            SR_imgs = torch.Tensor([]).cuda()
            student_G.eval()
            for x, y in val_loader:
                AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)
                SR_img = student_G(AR_img)

                SR_img = SR_img[-1].squeeze(1)
                SR_imgs = torch.cat((SR_imgs, SR_img),0)

        SR_path = "../datasplit/new_train/train2/SR%d" % epoch
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
        D_path, G_path = "../datasplit/new_train/train2/D_model%d.pkl" % epoch, "../datasplit/new_train/train2/G_model%d.pkl" % epoch
        torch.save(D.state_dict(), D_path)  
        torch.save(student_G.state_dict(), G_path)                 
        print("Save SRGAN model in %dth epoch!\n" % epoch)

        # save train_result per epoch
        stat_path = '../datasplit/new_train/train2/SRGAN_MultiDtrain.csv'
        data_frame = pd.DataFrame(
        data = {'Iters':train_result['iter'], 'Loss_D':train_result['d_loss'],'Loss_G':train_result['g_loss'],  
                            'PSNR':train_result['psnr']})
        data_frame.to_csv(stat_path, index=False, sep=',')

    print("Snapshot Esembles training Done!")


            
                
    

    

            



