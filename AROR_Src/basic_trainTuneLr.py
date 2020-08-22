# -*- coding: utf8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import numpy as np
import pandas as pd

import pytorch_ssim
from utils import *
from SRGAN_V9 import *


'''
Finetune Lr for WGAN-GP model
'''

if __name__ == "__main__":

    
    # Hyper parameter
    dtype = torch.cuda.FloatTensor
    BATCH_SIZE = 2

    # Load data
    train_set = DatasetFromFolder('../Augdata/ARnpy', '../Augdata/ORnpy')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=4)

    ITER_PER_EPOCH = len(train_loader)
    NUM_EPOCHS = 12

    # Import SRGAN model
    G = Generator().cuda()
    D = Discriminator_WGAN((2,1,384,384),2).cuda() 
    Loss = 'LossG_adv'  #1e-1,1e-2,1e-3
    RecFile = open('../datasplit/new_train/train8/RecordLr.txt', 'w+')

    # G.load_state_dict(torch.load('../datasplit/new_train/train7/BCE/G_model1.pkl'))
    # D.load_state_dict(torch.load('../datasplit/new_train/train7/BCE/D_model1.pkl'))
            
    D_lr_end, G_lr_end = 0, 0
    G_lr_inits = [1e-4]   
    D_lr_inits = [1e-6,1e-5]  
    
    #Visulaize Loss_G, D(x), D(G(z)) given different Lrs
    fig1, fig2, fig3 = plt.figure(1), plt.figure(2), plt.figure(3)
    fig4, fig5, fig6 = plt.figure(4), plt.figure(5), plt.figure(6)
    ax1, ax2, ax3 = fig1.add_subplot(111), fig2.add_subplot(111), fig3.add_subplot(111)
    ax4, ax5, ax6 = fig4.add_subplot(111), fig5.add_subplot(111), fig6.add_subplot(111)
    
    for G_lr_init in G_lr_inits:
        for D_lr_init in D_lr_inits:

            #Small weight initialization
            G.load_state_dict(torch.load("../datasplit/new_train/train8/Weights/PretrainG.pth")) #Pre-training G
            D.apply(D_initweights)
            # optimizerG, optimizerD = torch.optim.RMSprop(G.parameters(), lr=G_lr_init), \
            #             torch.optim.RMSprop(D.parameters(), lr=D_lr_init)   #WGAN
            optimizerG, optimizerD = torch.optim.AdamW(G.parameters(), lr=G_lr_init, betas=(0.5, 0.9)), \
                    torch.optim.AdamW(D.parameters(), lr=D_lr_init, betas=(0.5, 0.9)) #Weight_decay=1e-2   #WGAN-GP
            schedulerG, schedulerD = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=NUM_EPOCHS, eta_min=G_lr_end), \
                        lr_scheduler.CosineAnnealingLR(optimizerD, T_max=NUM_EPOCHS, eta_min=D_lr_end)
        
            print("Training started, G_lr_init=%s, D_lr_init=%s" % (G_lr_init, D_lr_init)) 
            RecFile.write("Training started, G_lr_init=%s, D_lr_init=%s\n" % (G_lr_init, D_lr_init)) 
            RecFile.flush()

            #To record iter times in the total training
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'l1_loss': 0, 'd_score': 0, 'g_score': 0, 'psnr': 0}
            Loss_Gs, Loss_Ds, L1Loss, Dxs, DGzs, PSNRs = [], [], [], [], [], []

            #Make directories
            train_path = '../datasplit/new_train/train8/'                 
            SR_path, AR_path, OR_path = train_path+'SR_iter/G_lr%sD_lr%s/'%(G_lr_init, D_lr_init), \
                train_path+'AR_iter/G_lr%sD_lr%s/'%(G_lr_init, D_lr_init), train_path+'OR_iter/G_lr%sD_lr%s/'%(G_lr_init, D_lr_init)
            mkdir(SR_path)
            mkdir(AR_path)
            mkdir(OR_path)

            iters = 1
            G.train()
            D.train()
            for epoch in range(1, 1+NUM_EPOCHS): #NUM_EPOCHS             
                for x, y in train_loader:
                    AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)     

                    # # Update lr every iteration
                    # G_lr = snapshot_lr(G_lr_init, G_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)
                    # D_lr = snapshot_lr(D_lr_init, D_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)        
                    # optimizerG.param_groups[0]['lr'] = G_lr
                    # optimizerD.param_groups[0]['lr'] = D_lr

                    # train Discriminator
                    optimizerD.zero_grad()
                    WDreal = D(OR_img)
                                        
                    try:
                        SR_img = G(AR_img)  #.detach()
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            print("WARNING: CUDA out of memory")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise exception
                    WDfake = D(SR_img)
                    GP = GradPenalty(D,OR_img,SR_img)

                    # Wasserstein GAN Loss
                    D_loss = discriminator_loss(WDreal,WDfake,GP)
                    D_loss.backward(retain_graph=True)
                    
                    # clip_grad_value_(D.parameters(), 0.01)  #Weight clipping               
                    optimizerD.step()

                    
                    # train Generator TWICE          
                    optimizerG.zero_grad()       
                    # try: 
                    #     SR_img = G(AR_img)
                    # except RuntimeError as exception:
                    #     if "out of memory" in str(exception):
                    #         print("WARNING: CUDA out of memory")
                    #         if hasattr(torch.cuda, 'empty_cache'):
                    #             torch.cuda.empty_cache()
                    #     else:
                    #         raise exception
                    # WDfake = D(SR_img)
                    WDfake = D(SR_img) #必须重新求WDfake，反向求导阶段要用到的tensor不能变
                    G_loss, MAE, FMAE, LossG_adv = generator_loss(WDfake, OR_img, SR_img, 1, 0.1)
                    L1_loss = MAE+1e-2*FMAE
                    G_loss.backward()
                    test_grad_value(G.parameters(), optimizerG)
                    # optimizerG.step()

                    #loss for current batch
                    running_results['batch_sizes'] += BATCH_SIZE
                    running_results['g_loss'] += G_loss.item() * BATCH_SIZE
                    running_results['l1_loss'] += L1_loss.item() * BATCH_SIZE
                    running_results['d_loss'] += D_loss.item() * BATCH_SIZE 
                    running_results['g_score'] += WDfake.mean().item() * BATCH_SIZE
                    running_results['d_score'] += WDreal.mean().item() * BATCH_SIZE
                    running_results['psnr'] += calculate_psnr(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy()) * BATCH_SIZE

                    #Save training results
                    if (iters % 100 == 0): 

                        g_loss = running_results['g_loss'] / running_results['batch_sizes']
                        l1_loss = running_results['l1_loss'] / running_results['batch_sizes']
                        d_loss = running_results['d_loss'] / running_results['batch_sizes']
                        d_score = running_results['d_score'] / running_results['batch_sizes']
                        g_score = running_results['g_score'] / running_results['batch_sizes']
                        psnr = running_results['psnr'] / running_results['batch_sizes']

                        Loss_Gs.append(g_loss)
                        L1Loss.append(l1_loss)
                        Loss_Ds.append(d_loss)
                        Dxs.append(d_score)
                        DGzs.append(g_score)
                        PSNRs.append(psnr)


                        #Show training results within a epoch dynamically
                        RecFile.write("Epoch|NUM_EPOCHS: %d/%d, Iter/Iter_num: %d/%d, Loss_D:%.6f, Loss_G:%.6f, \
L1Loss:%.6f, D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f\n" % (epoch, NUM_EPOCHS, \
iters, NUM_EPOCHS*ITER_PER_EPOCH, d_loss, g_loss, l1_loss, d_score, g_score, psnr))
                        RecFile.flush()

                        #Save images  
                        show_images(SR_img, SR_path+str(iters)+'.png')
                        show_images(AR_img, AR_path+str(iters)+'.png')
                        show_images(OR_img, OR_path+str(iters)+'.png')

                    iters = iters+1
                    if iters == 2000+1:
                        break                                                                      
                break
                #update the learning rate
                schedulerD.step() 
                schedulerG.step()
                
            print("Loss_G:%.6f, L1Loss: %.4f, Loss_D: %.6f, D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f" % (g_loss, \
                                                        l1_loss, d_loss, d_score, g_score, psnr))

            #Visulaization
            Recordlen = len(Loss_Gs)
            ax1.plot(np.arange(1,Recordlen+1),Loss_Gs, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
            ax2.plot(np.arange(1,Recordlen+1), L1Loss, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
            ax3.plot(np.arange(1,Recordlen+1),Loss_Ds, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
            ax4.plot(np.arange(1,Recordlen+1), Dxs, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
            ax5.plot(np.arange(1,Recordlen+1), DGzs, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
            ax6.plot(np.arange(1,Recordlen+1), PSNRs, label='G_lr=%s, D_lr=%s' % (G_lr_init,D_lr_init))
        

    Xtick = [1]+list(range(Recordlen//5,Recordlen+1,Recordlen//5)) 
    Xlabel = [100*x for x in Xtick]
    Xlabel = tuple([str(a) for a in Xlabel])
    ax1.grid(True, linestyle='-')
    ax1.legend(loc = 'upper right')
    ax1.set_xticks(Xtick)
    ax1.set_xticklabels(Xlabel)
    ax1.set_xlabel("Iter")
    ax1.set_ylabel("Loss_G")

    ax2.grid(True, linestyle='-')
    ax2.legend(loc = 'upper right')
    ax2.set_xticks(Xtick)
    ax2.set_xticklabels(Xlabel)
    ax2.set_xlabel("Iter")
    ax2.set_ylabel("L1Loss")

    ax3.grid(True, linestyle='-')
    ax3.legend(loc = 'upper right')
    ax3.set_xticks(Xtick)
    ax3.set_xticklabels(Xlabel)
    ax3.set_xlabel("Iter")
    ax3.set_ylabel("Loss_D")

    ax4.grid(True, linestyle='-')
    ax4.legend(loc = 'upper right')
    ax4.set_xticks(Xtick)
    ax4.set_xticklabels(Xlabel)
    ax4.set_xlabel("Iter")
    ax4.set_ylabel("D(x)")

    ax5.grid(True, linestyle='-')
    ax5.legend(loc = 'upper right')
    ax5.set_xticks(Xtick)
    ax5.set_xticklabels(Xlabel)
    ax5.set_xlabel("Iter")
    ax5.set_ylabel("D(G(z))")

    ax6.grid(True, linestyle='-')
    ax6.legend(loc = 'upper right')
    ax6.set_xticks(Xtick)
    ax6.set_xticklabels(Xlabel)
    ax6.set_xlabel("Iter")
    ax6.set_ylabel("PSNR")
    plt.show()
    
    print("Finished!\n")

