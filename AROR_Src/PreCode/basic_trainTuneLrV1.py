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
    NUM_EPOCHS = 12

    # Import SRGAN model
    G = Generator().cuda()
    D = Discriminator().cuda()
    Loss = 'SquCE'  #1e-1,1e-2,1e-3
    RecFile = open('../datasplit/new_train/train7/%s/Recordfine.txt' % Loss, 'w+')

    # G.load_state_dict(torch.load('../datasplit/new_train/train7/BCE/G_model1.pkl'))
    # D.load_state_dict(torch.load('../datasplit/new_train/train7/BCE/D_model1.pkl'))
            
    D_lr_end, G_lr_end = 0, 0
    G_lr_inits = [1e-3,1e-3,5e-4,1e-4,1e-4]  #1e-3, 5e-4, 1e-4, 5e-5, 
    D_lr_inits = [1e-4,1e-7,1e-7,1e-4,5e-6]  #1e-4, 5e-5, 1e-5,

    #Visulaize Loss_G, D(x), D(G(z)) given different Lrs
    fig1, fig2, fig3 = plt.figure(1), plt.figure(2), plt.figure(3)
    ax1, ax2, ax3 = fig1.add_subplot(111), fig2.add_subplot(111), fig3.add_subplot(111)
    
    for i in range(5):
        G_lr_init = G_lr_inits[i]
        D_lr_init = D_lr_inits[i]

        # Small weight initialization
        G.apply(G_initweights)
        D.apply(D_initweights)
        optimizerG, optimizerD = torch.optim.AdamW(G.parameters(), lr=G_lr_init), \
                    torch.optim.AdamW(D.parameters(), lr=D_lr_init) #Weight_decay=1e-2, , betas=(0.9, 0.999)

        # schedulerG, schedulerD = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=NUM_EPOCHS, eta_min=G_lr_end), \
        #                 lr_scheduler.CosineAnnealingLR(optimizerD, T_max=NUM_EPOCHS, eta_min=D_lr_end)

        # # Changes invoked by Apex.amp
        # # To enable mixed precision training, we introduce the --opt_level argument.
        # [G,D], [optimizerG, optimizerD] = amp.initialize(
        #     [G,D], [optimizerG, optimizerD], opt_level="O1", num_losses=2) #recommended mixed precision

        
        print("Training started, G_lr_init=%s, D_lr_init=%s" % (G_lr_init, D_lr_init)) 
        RecFile.write("Training started, G_lr_init=%s, D_lr_init=%s\n" % (G_lr_init, D_lr_init)) 
        RecFile.flush()
        #to record iter times in the total training
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'psnr': 0}
        Loss_Gs, Dxs, DGzs = [], [], []

        iters = 1
        G.train()
        D.train()

        for epoch in range(1, 1+NUM_EPOCHS): #NUM_EPOCHS             
            for x, y in train_loader:
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
                    SR_img = G(AR_img).detach()
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: CUDA out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                D_loss = discriminator_loss(D(OR_img), D(SR_img))

                D_loss.backward()
                # # Changes invoked by Apex.amp
                # with amp.scale_loss(D_loss, optimizerD, loss_id=0) as errD_scaled:
                #     errD_scaled.backward()
                #nn.utils.clip_grad_norm_(D.parameters(), max_norm=20, norm_type=2)
                # clip_grad_value_(D.parameters(), 20)  
                # test_grad_value(D.parameters(), optimizerD)
                optimizerD.step()

                # train Generator TWICE
            
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
                G_loss, InteMAE, SquCE = generator_loss(logits_SR, OR_img, SR_img, 0.01)
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
                if (iters % 100 == 0): #(iters % ITER_PER_EPOCH <= 100 and iters % 2 == 0) or (iters % ITER_PER_EPOCH > 100 and iters % 100 == 0)
                            
                    d_loss = running_results['d_loss'] / running_results['batch_sizes']
                    g_loss = running_results['g_loss'] / running_results['batch_sizes']
                    d_score = running_results['d_score'] / running_results['batch_sizes']
                    g_score = running_results['g_score'] / running_results['batch_sizes']
                    psnr = running_results['psnr'] / running_results['batch_sizes']

                    Loss_Gs.append(g_loss)
                    Dxs.append(d_score)
                    DGzs.append(g_score)

                    #Show training results within a epoch dynamically
                    RecFile.write("Epoch|NUM_EPOCHS: %d/%d, Iter/Iter_num: %d/%d, Loss_D:%.6f, Loss_G:%.6f, \
D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f, G_lr:%.4e, D_lr:%.4e, InteMAE:%.4f, SquCE:%.4f\n" % ( epoch, NUM_EPOCHS, \
iters, NUM_EPOCHS*ITER_PER_EPOCH, d_loss, g_loss, d_score, g_score, psnr, G_lr, D_lr, InteMAE, SquCE))
                    RecFile.flush()
                    
        
                if (iters % ITER_PER_EPOCH) % 1000 == 0:
                    img_path = '../datasplit/new_train/train7/'
                    show_images(SR_img, img_path+'%s/SR_iter/%s_%s%s.png' % (Loss, int(iters), G_lr_init, D_lr_init))
                    show_images(AR_img, img_path+'%s/SR_iter/%s_%s%s.png' % (Loss, int(iters), G_lr_init, D_lr_init))
                    show_images(OR_img, img_path+'%s/SR_iter/%s_%s%s.png' % (Loss, int(iters), G_lr_init, D_lr_init))

                iters = iters+1
                if iters == 5000+1:
                    break                                                                      
            break
        print("D(x):%.4f, D(G(z)):%.4f" % (d_score, g_score))

        #Visulaization
        ax1.plot(np.arange(1,50+1),Loss_Gs, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
        ax2.plot(np.arange(1,50+1), Dxs, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
        ax3.plot(np.arange(1,50+1), DGzs, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
        

    Xtick = np.array([1]+list(range(10,50+1,10)))
    Xlabel = ("100","1000","2000","3000","4000","5000")
    ax1.grid(True, linestyle='-')
    ax1.legend(loc = 'lower right')
    ax1.set_xticks(Xtick)
    ax1.set_xticklabels(Xlabel)
    ax1.set_xlabel("Iter")
    ax1.set_ylabel("Loss_G")

    ax2.grid(True, linestyle='-')
    ax2.legend(loc = 'lower right')
    ax2.set_xticks(Xtick)
    ax2.set_xticklabels(Xlabel)
    ax2.set_xlabel("Iter")
    ax2.set_ylabel("D(x)")

    ax3.grid(True, linestyle='-')
    ax3.legend(loc = 'lower right')
    ax3.set_xticks(Xtick)
    ax3.set_xticklabels(Xlabel)
    ax3.set_xlabel("Iter")
    ax3.set_ylabel("D(G(z))")
    plt.show()
    
    print("Finished!\n")



     



