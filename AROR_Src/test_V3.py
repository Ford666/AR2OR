import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from SRGAN_V6 import *
from utils import *
import gc


# AR_aug =  np.load('../Data_aug/AR_aug.npy') #(8,144,2000,2000)
# OR_aug =  np.load('../Data_aug/OR_aug.npy') #(8,144,2000,2000)


# # #选取弹性变换后的大图裁出图像块，构成测试集
# extract_patch(AR_aug[4][0], OR_aug[4][0], 2000, 2000, 384, 384, 64, '../datasplit/test/7th/')
# extract_patch(AR_aug[5][0], OR_aug[5][0], 2000, 2000, 384, 384, 64, '../datasplit/test/7th/')
# extract_patch(AR_aug[7][0], OR_aug[7][0], 2000, 2000, 384, 384, 64, '../datasplit/test/7th/')


# del AR_aug, OR_aug
# gc.collect()


if __name__ == "__main__":

    #dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
    BATCH_SIZE = 2

    #Load data
    test_set = DatasetFromFolder('../datasplit/test/7th/x', '../datasplit/test/7th/y')

    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, 
                                    shuffle=False, num_workers=BATCH_SIZE)

    test_imgs = []
    AR = np.load("../datasplit/test/7th/AR.npy")
    OR = np.load("../datasplit/test/7th/OR.npy")

        
    ARMean, ARStd = np.mean(AR), np.std(AR, ddof=1)
    ORMean, ORStd = np.mean(OR), np.std(OR, ddof=1) 
    AR_var, OR_var = AR.var(), OR.var()
    AR, OR = AR[0:6*384-5*64, 0:6*384-5*64], OR[0:6*384-5*64, 0:6*384-5*64]
    psnr, ssim, pcc = calculate_psnr(AR, OR), calculate_ssim(AR, OR), calculate_pcc(AR, OR)
    print("AR|OR, AR_mean: %.4f, AR_var: %.4f, OR_mean: %.4f, OR_var: %.4f, \
PSNR: %.6f, SSIM: %.6f, PCC: %.6f\n" % (ARMean, AR_var, ORMean, OR_var, psnr, ssim, pcc))          
  

    #Test using different SRGAN models
    G = Generator().cuda()
    D = Discriminator().cuda()
    

    for i in range(12):

        test_result =  {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}
        #Load and initialize GAN      
        G.load_state_dict(torch.load("../datasplit/new_train/train6/G_model%d.pkl" % (i+1)))
        D.load_state_dict(torch.load("../datasplit/new_train/train6/D_model%d.pkl" % (i+1)))
        print("Use trained GAN model %d to test!" % (i+1))
        
        with torch.no_grad(): 
            SR_imgs = torch.Tensor([]).cuda()
            G.eval()

            for x, y in test_loader:
                AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(4,1,384,384)          
                SR_img = G(AR_img) 

                logits_SR, logits_OR = D(SR_img), D(OR_img)
                G_loss = generator_loss(logits_SR, OR_img, SR_img, 0.01)
                D_loss = discriminator_loss(D(OR_img), D(SR_img))

                test_result['batch_sizes'] += BATCH_SIZE
                test_result['g_loss'] += G_loss.item() * BATCH_SIZE
                test_result['d_loss'] += D_loss.item() * BATCH_SIZE 

                SR_img = SR_img.squeeze(1)
                SR_imgs = torch.cat((SR_imgs, SR_img),0)

            SR_path = "../datasplit/test/7th/basic/train6/SR_cycle%d" % (i+1)
            SRpatch_path = "../datasplit/test/7th/basic/train6/cycle%d/" % (i+1)
            SR = stitch_SavePatch(SR_path, SRpatch_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64)
            
        #Test loss
        d_loss = test_result['d_loss'] / test_result['batch_sizes']
        g_loss = test_result['g_loss'] / test_result['batch_sizes']
        print("SR%d| Loss_D:%.6f, Loss_G:%.6f" % (i+1,d_loss,g_loss))

        # Image assessment metrics  
        mean, var = SR.mean(), SR.var()
        psnr, ssim, pcc = calculate_psnr(SR, OR), calculate_ssim(SR, OR), calculate_pcc(SR, OR)
        print("Mean: %.4f, Var: %.4f, PSNR: %.6f, SSIM: %.6f, PCC: %.6f\n" % ( mean, var, psnr, ssim, pcc))

        # spectrum analysis
        # #AROR: Golden standard   
        # fAR_dir, fOR_dir = "../datasplit/test/7th/SRGAN_esemble/AR_spectrum.tiff", "../datasplit/test/7th/SRGAN_esemble/OR_spectrum.tiff"
        # fAR, fOR = np.fft.fftshift(np.fft.fft2(AR)), np.fft.fftshift(np.fft.fft2(OR))
        fSR = np.fft.fftshift(np.fft.fft2(SR))
        fSR_dir = "../datasplit/test/7th/basic/train6/SR%d_spectrum.tiff" % (i+1)
        # plt.imsave(fAR_dir, np.log(np.abs(fAR)), cmap = 'gray')
        # plt.imsave(fOR_dir, np.log(np.abs(fOR)), cmap = 'gray')
        plt.imsave(fSR_dir, np.log(np.abs(fSR)), cmap=cm.gray)
    
    print("Finish test!\n")
