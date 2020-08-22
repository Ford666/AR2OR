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
# from apex import amp


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
    RecFile = open('../datasplit/new_train/train8/RecordLossfa.txt', 'w+')

            
    D_lr_end, G_lr_end = 0, 0
    lambds = [10]   
    betas = [1e-3,1e-2,0.1] 
    
    #Visulaize Loss_G, D(x), D(G(z)) given different Lrs
    fig1, fig2, fig3 = plt.figure(1), plt.figure(2), plt.figure(3)
    ax1, ax2, ax3 = fig1.add_subplot(111), fig2.add_subplot(111), fig3.add_subplot(111)
    
    for lambd in lambds:
        for beta in betas:
            G_lr_init, D_lr_init = 1e-4, 1e-6    

            #Small weight initialization
            G.load_state_dict(torch.load("../datasplit/new_train/train8/Weights/PretrainG.pth"))
            D.apply(D_initweights)
            # optimizerG, optimizerD = torch.optim.RMSprop(G.parameters(), lr=G_lr_init), \
            #             torch.optim.RMSprop(D.parameters(), lr=D_lr_init)   #WGAN
            optimizerG, optimizerD = torch.optim.AdamW(G.parameters(), lr=G_lr_init, betas=(0.5, 0.9)), \
                    torch.optim.AdamW(D.parameters(), lr=D_lr_init, betas=(0.5, 0.9)) #Weight_decay=1e-2   #WGAN-GP

            schedulerG, schedulerD = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=NUM_EPOCHS, eta_min=G_lr_end), \
                        lr_scheduler.CosineAnnealingLR(optimizerD, T_max=NUM_EPOCHS, eta_min=D_lr_end)
        
            print("WGAN-GP training started, GP paramter=%s, LossG_adv paramter=%s" % (lambd, beta)) 
            RecFile.write("WGAN-GP training started, GP paramter=%s, LossG_adv paramter=%s\n" % (lambd, beta))
            RecFile.flush()

            #To record iter times in the total training
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'l1_loss':0, 'd_score': 0, 'g_score': 0, 'psnr': 0}
            Loss_Gs, Loss_Ds, L1losss = [], [], []

            #Make directories
            train_path = '../datasplit/new_train/train8/'                 
            SR_path, AR_path, OR_path = train_path+'SR_iter/lambd_%sbeta_%s/'%(lambd, beta), \
                train_path+'AR_iter/lambd_%sbeta_%s/'%(lambd, beta), train_path+'OR_iter/lambd_%sbeta_%s/'%(lambd, beta)
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

                    # train Discriminator TWICE
                    # for k in range(2):
                    optimizerD.zero_grad()
                    WDreal = D(OR_img)
                                        
                    try:
                        SR_img = G(AR_img)
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
                    WDfake = D(SR_img)
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
                    running_results['d_score'] += torch.sigmoid(WDreal.mean()).item() * BATCH_SIZE
                    running_results['g_score'] += torch.sigmoid(WDfake.mean()).item() * BATCH_SIZE   
                    running_results['psnr'] += calculate_psnr(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy()) * BATCH_SIZE

                    #Save training results
                    if (iters % 200 == 0): 
                                
                        d_loss = running_results['d_loss'] / running_results['batch_sizes']
                        g_loss = running_results['g_loss'] / running_results['batch_sizes']
                        l1_loss = running_results['l1_loss'] / running_results['batch_sizes']
                        d_score = running_results['d_score'] / running_results['batch_sizes']
                        g_score = running_results['g_score'] / running_results['batch_sizes']
                        psnr = running_results['psnr'] / running_results['batch_sizes']

                        Loss_Gs.append(g_loss)
                        L1losss.append(l1_loss)
                        Loss_Ds.append(d_loss)
                        

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
                
            print("Loss_G:%.6f, L1loss:%.6f, Loss_D:%.6f" % (g_loss, \
                                        l1_loss, d_loss))

            #Visulaization
            Recordlen = len(Loss_Gs) 
            ax1.plot(np.arange(1,Recordlen+1), Loss_Gs, label="GP par=%s, LossG_adv par=%s" % (lambd, beta))
            ax2.plot(np.arange(1,Recordlen+1), L1losss, label="GP par=%s, LossG_adv par=%s" % (lambd, beta))
            ax3.plot(np.arange(1,Recordlen+1), Loss_Ds, label="GP par=%s, LossG_adv par=%s" % (lambd, beta))
        

    Xtick = [1]+list(range(2,Recordlen+1,2)) 
    Xlabel = [200*x for x in Xtick]
    Xlabel = tuple([str(a) for a in Xlabel])
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
    ax2.set_ylabel("L1loss")

    ax3.grid(True, linestyle='-')
    ax3.legend(loc = 'lower right')
    ax3.set_xticks(Xtick)
    ax3.set_xticklabels(Xlabel)
    ax3.set_xlabel("Iter")
    ax3.set_ylabel("Loss_D")
    plt.show()
    
    print("Finished!\n")

