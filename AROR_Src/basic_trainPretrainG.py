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
August 16-17
WGAN-GP训练
做了大量预实验对学习率、LossG构成项及其比例进行调参，并用MSE预训练Generator
G_lr=1e-4, D_lr=1e-6, LossG=MAE+0.01*FMAE-1e-3*D(G(z)), LossD=D(G(z))-D(x), Discriminator用InstanceNorm2d归一化
但上述设置，训练3个epoch似乎仍不能保持对抗平衡，Discriminator过强，LossD曲线从+4不断下降到-6，LossG小幅回升，PSNR小幅增加到19.8
内心凄惶，满腔苦闷...

再怎样设置学习率和LossG_adv，也始终无法达到对抗平衡，如何解决？
1. LossG_adv factor不是固定值，而随L1loss变化，占LossG约20%
(这种方法下LossG_advc非独立变量，相当于3中对L1loss乘上系数)；
2. LossD = D(G(z))-D(x)反向求导时太强了，乘一个系数衰减
3. LossG = MAE+0.01*FMAE-1e-3*D(G(z))反向求导时太弱了，乘一个系数增强
4. WGAN直接使用Wasserstein距离之差作为GAN Loss，调整为平方误差 s*(1-z)^2+(1-s)*z^2，
把D(G(z))和D(x)约束在(0,1)，防止LossD过度下降，方便构造占比合适的LossG_adv，有利于对抗平衡
'''

if __name__ == "__main__":

    # Hyper parameter
    dtype = torch.cuda.FloatTensor
    BATCH_SIZE = 2
    NUM_EPOCHS = 12
    PretrainNum = 2 
    train_path = '../datasplit/new_train/train8/'  

    # Load data
    train_set = DatasetFromFolder('../Augdata/ARnpy', '../Augdata/ORnpy')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=0)
    val_set = DatasetFromFolder('../datasplit/test/7th/x', '../datasplit/test/7th/y')
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, 
                                shuffle=False, num_workers=0)                               
    val_OR = np.load("../datasplit/test/7th/OR.npy")
    val_OR = val_OR[0:6*384-5*64, 0:6*384-5*64]

    ITER_PER_EPOCH = len(train_loader)

    # Import SRGAN model
    G = Generator().cuda()
    D = Discriminator_WGAN((2,1,384,384),2).cuda()  #InstanceNorm2d
    print('\n# Generator parameters:', sum(param.numel() for param in G.parameters()))  
    print('# Discriminator parameters:', sum(param.numel() for param in D.parameters()))

    RecFile = open('../datasplit/new_train/train8/RecordPretrainG.txt', 'w+')
    
    #Small weight initialization
    # G.apply(G_initweights)
    # D.apply(D_initweights)              
    
    # #Pre-train Generator using only MSE loss
    # print("Pretraining Generator started" )
    # RecFile.write("Pretraining Generator started\n")
    # RecFile.flush()

    # optimizerG = torch.optim.AdamW(G.parameters(), lr=1e-4)
    # preiter = 1

    # #Make directories
    # PreSR_path, PreAR_path, PreOR_path = train_path+'SR_iter/Pretrain/Pre', \
    #                     train_path+'AR_iter/Pretrain/Pre', train_path+'OR_iter/Pretrain/Pre'
    # mkdir(PreSR_path)
    # mkdir(PreAR_path)
    # mkdir(PreOR_path)

    # #Save Generator weights
    # best_mse = 1

    # for epoch in range(1,1+PretrainNum): #NUM_EPOCHS 
    #     G.train()
    #     cache = {'g_loss': 0}
    #     for x, y in progressbar(train_loader):
    #         AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype)         
    #         optimizerG.zero_grad()       
    #         try: 
    #             SR_img = G(AR_img)
    #         except RuntimeError as exception:
    #             if "out of memory" in str(exception):
    #                 print("WARNING: CUDA out of memory")
    #                 if hasattr(torch.cuda, 'empty_cache'):
    #                     torch.cuda.empty_cache()
    #             else:
    #                 raise exception
    #         G_MSEloss = MAE_loss(SR_img, OR_img)
    #         cache['g_loss'] += G_MSEloss
    #         g_loss = cache['g_loss']/preiter

    #         G_MSEloss.backward()
    #         optimizerG.step()
            
    #         if g_loss < best_mse:
    #             best_mse = G_MSEloss
    #             G_weight = G.state_dict()

    #         if preiter % 1000 == 0:
    #             RecFile.write("Epoch|PreTrainNum: %d|%d, Iter/Iter_num: %d/%d, G_MSELoss=%.6f\n" % (epoch, \
    #             PretrainNum, preiter, PretrainNum*ITER_PER_EPOCH, g_loss))
    #             RecFile.flush()

    #             #Save images  
    #             show_images(SR_img, PreSR_path+str(preiter)+'.png')
    #             show_images(AR_img, PreAR_path+str(preiter)+'.png')
    #             show_images(OR_img, PreOR_path+str(preiter)+'.png')
    #         preiter += 1
    # torch.save(G_weight,"../datasplit/new_train/train8/Weights/PretrainG.pth")

    #Ready to train GAN!
    G.apply(G_initweights) 
    D.apply(D_initweights) 

    #Visulaize Loss_G, D(x), D(G(z)) given different Lrs
    fig1, fig2, fig3 = plt.figure(1), plt.figure(2), plt.figure(3)
    fig4, fig5, fig6 = plt.figure(4), plt.figure(5), plt.figure(6)
    ax1, ax2, ax3 = fig1.add_subplot(111), fig2.add_subplot(111), fig3.add_subplot(111)
    ax4, ax5, ax6 = fig4.add_subplot(111), fig5.add_subplot(111), fig6.add_subplot(111)
    
    #Fine-tune parameters
    G_lr_init, D_lr_init = 1e-4, 1e-5
    G_lr_end, D_lr_end = 0, 0
    # optimizerG, optimizerD = torch.optim.RMSprop(G.parameters(), lr=G_lr_init), \
    #             torch.optim.RMSprop(D.parameters(), lr=D_lr_init)   #WGAN
    optimizerG, optimizerD = torch.optim.AdamW(G.parameters(), lr=G_lr_init), \
            torch.optim.AdamW(D.parameters(), lr=D_lr_init) #Weight_decay=1e-2   #WGAN-GP
    schedulerG, schedulerD = lr_scheduler.CosineAnnealingLR(optimizerG, T_max=NUM_EPOCHS, eta_min=G_lr_end), \
                lr_scheduler.CosineAnnealingLR(optimizerD, T_max=NUM_EPOCHS, eta_min=D_lr_end)

    print("Training GAN started" )
    RecFile.write("Training GAN started\n")
    RecFile.flush()

    #To record iter times in the total training
    train_result = {'iters':[], 'g_losses':[], 'l1_losses':[], 'd_losses':[], 'Dxs':[], 'DGzs':[], 'psnrs':[]}
    running_results = {'d_loss': 0, 'g_loss': 0, 'l1_loss': 0, 'd_gz': 0, 'd_x': 0, 'psnr': 0}

    #Make directories               
    SR_path, AR_path, OR_path = train_path+'SR_iter/GANtrain/', \
                        train_path+'AR_iter/GANtrain/', train_path+'OR_iter/GANtrain/'
    mkdir(SR_path)
    mkdir(AR_path)
    mkdir(OR_path)

    iters = 1 
    reidx = 0 #index of train results

    for epoch in range(1, 1+NUM_EPOCHS): #NUM_EPOCHS   
        G.train()
        D.train()     
        for x, y in progressbar(train_loader):
            AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)     

            # # Update lr every iteration
            # G_lr = snapshot_lr(G_lr_init, G_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)
            # D_lr = snapshot_lr(D_lr_init, D_lr_end, iters, NUM_EPOCHS*ITER_PER_EPOCH)        
            # optimizerG.param_groups[0]['lr'] = G_lr
            # optimizerD.param_groups[0]['lr'] = D_lr

            # train Discriminator                                  
            try:
                SR_img = G(AR_img).detach()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: CUDA out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            optimizerD.zero_grad()

            WDreal = D(OR_img)
            WDfake = D(SR_img)
            GP = GradPenalty(D, OR_img, SR_img)

            # Wasserstein GAN Loss
            D_loss = discriminator_loss(WDreal,WDfake,GP)
            D_loss.backward()
            
            # clip_grad_value_(D.parameters(), 0.01)  #Weight clipping               
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
            WDfake = D(SR_img).detach()      
            G_loss, MAE, FMAE, LossG_adv = generator_loss(WDfake, OR_img, SR_img, 1, 1e-1)
            L1_loss = MAE+1e-2*FMAE
            G_loss.backward()
            test_grad_value(G.parameters(), optimizerG)
            # optimizerG.step()

            #loss for current batch
            running_results['g_loss'] += G_loss.item()
            running_results['l1_loss'] += L1_loss.item()
            running_results['d_loss'] += D_loss.item() 
            running_results['d_x'] += WDfake.mean().item()
            running_results['d_gz'] += WDreal.mean().item()
            running_results['psnr'] += calculate_psnr(SR_img.data.cpu().numpy(), OR_img.data.cpu().numpy())

            #Save training results
            if ((iters % ITER_PER_EPOCH) % 100 == 0): 
                train_result['iters'].append(iters)
                train_result['g_losses'].append(running_results['g_loss'] / iters)
                train_result['l1_losses'].append(running_results['l1_loss'] / iters)
                train_result['d_losses'].append(running_results['d_loss'] / iters)
                train_result['Dxs'].append(running_results['d_gz'] / iters)
                train_result['DGzs'].append(running_results['d_x'] / iters)
                train_result['psnrs'].append(running_results['psnr'] / iters)


                #Show training results within a epoch dynamically
                RecFile.write("Epoch|NUM_EPOCHS: %d/%d, Iter/Iter_num: %d/%d, Loss_D:%.6f, Loss_G:%.6f, \
L1Loss:%.6f, D(x):%.4f, D(G(z)):%.4f, PSNR:%.4f\n" % (epoch, NUM_EPOCHS, iters, NUM_EPOCHS*ITER_PER_EPOCH, \
train_result['d_losses'][reidx], train_result['g_losses'][reidx], train_result['l1_losses'][reidx], \
train_result['Dxs'][reidx], train_result['DGzs'][reidx], train_result['psnrs'][reidx]))
                RecFile.flush()
                reidx += 1

                #Save images  
                show_images(SR_img, SR_path+str(iters)+'.png')
                show_images(AR_img, AR_path+str(iters)+'.png')
                show_images(OR_img, OR_path+str(iters)+'.png')

            iters += 1
        #     if iters == 2000+1:
        #         break                                                                      
        # break
        #update the learning rate
        schedulerD.step() 
        schedulerG.step()


        # Save model per epoch
        D_path = "../datasplit/new_train/train8/Weights/D_model%d.pth" % epoch
        G_path =  "../datasplit/new_train/train8/Weights/G_model%d.pth" % epoch
        torch.save(D.state_dict(), D_path)  
        torch.save(G.state_dict(), G_path)                 
        print("Save SRGAN model in %dth epoch!\n" % epoch)

        
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

        SRimg_path = "../datasplit/new_train/train8/SR%d" % epoch
        SR = stitch_patch(SRimg_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64) 
        val_qua['mean'], val_qua['var'] = SR.mean(), SR.var()
        val_qua['psnr'] = calculate_psnr(SR, val_OR) 
        val_qua['ssim'], val_qua['pcc'] = calculate_ssim(SR, val_OR), calculate_pcc(SR, val_OR)
        print("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f, PCC: %.4f" % (val_qua['mean'], \
                            val_qua['var'], val_qua['psnr'], val_qua['ssim'], val_qua['pcc']))
        RecFile.write("SR|OR, Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f, PCC: %.4f\n" % (val_qua['mean'], \
                            val_qua['var'], val_qua['psnr'], val_qua['ssim'], val_qua['pcc']))

        # save train_result per epoch
        stat_path = '../datasplit/new_train/train8/SRGAN_basictrain%d.csv' % epoch
        data_frame = pd.DataFrame(
        data = {'Iters':train_result['iters'], 'Loss_G':train_result['g_losses'], 'L1Loss':train_result['l1_losses'], 
'Loss_D':train_result['d_losses'],'D(x)':train_result['Dxs'], 'D(G(z))':train_result['DGzs'], 'PSNR':train_result['psnrs']})
        data_frame.to_csv(stat_path, index=False, sep=',')

    #Visualization
Recordlen = reidx
ax1.plot(np.arange(1,Recordlen+1),train_result['g_losses'])
ax2.plot(np.arange(1,Recordlen+1),train_result['l1_losses'])
ax3.plot(np.arange(1,Recordlen+1),train_result['d_losses'])
ax4.plot(np.arange(1,Recordlen+1), train_result['Dxs'])
ax5.plot(np.arange(1,Recordlen+1), train_result['DGzs'])
ax6.plot(np.arange(1,Recordlen+1), train_result['psnrs'])
    

Xtick = [1]+list(range(Recordlen//6,Recordlen+1,Recordlen//6))+[Recordlen]
Xlabel = [1000*x for x in Xtick]
Xlabel = tuple([str(a) for a in Xlabel])
ax1.grid(True, linestyle='-')
ax1.set_xticks(Xtick)
ax1.set_xticklabels(Xlabel)
ax1.set_xlabel("Iter")
ax1.set_ylabel("Loss_G")

ax2.grid(True, linestyle='-')
ax2.set_xticks(Xtick)
ax2.set_xticklabels(Xlabel)
ax2.set_xlabel("Iter")
ax2.set_ylabel("L1Loss")

ax3.grid(True, linestyle='-')
ax3.set_xticks(Xtick)
ax3.set_xticklabels(Xlabel)
ax3.set_xlabel("Iter")
ax3.set_ylabel("Loss_D")

ax4.grid(True, linestyle='-')
ax4.set_xticks(Xtick)
ax4.set_xticklabels(Xlabel)
ax4.set_xlabel("Iter")
ax4.set_ylabel("D(x)")

ax5.grid(True, linestyle='-')
ax5.set_xticks(Xtick)
ax5.set_xticklabels(Xlabel)
ax5.set_xlabel("Iter")
ax5.set_ylabel("D(G(z))")

ax6.grid(True, linestyle='-')
ax6.set_xticks(Xtick)
ax6.set_xticklabels(Xlabel)
ax6.set_xlabel("Iter")
ax6.set_ylabel("PSNR")
plt.show()

    print("Finished!\n")

