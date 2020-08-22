import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from SRGAN_V9 import *
from utils import *
import gc


# AR_aug =  np.load('../Data_aug/AR_aug.npy') #(8,144,2000,2000)
# OR_aug =  np.load('../Data_aug/OR_aug.npy') #(8,144,2000,2000)


# # #选取弹性变换后的大图裁出图像块，构成测试集
# extract_patch(AR_aug[1][0], OR_aug[1][0], 2000, 2000, 384, 384, 64, '../datasplit/test/2th/')
# extract_patch(AR_aug[6][0], OR_aug[6][0], 2000, 2000, 384, 384, 64, '../datasplit/test/9th/')
# extract_patch(AR_aug[6][6], OR_aug[6][6], 2000, 2000, 384, 384, 64, '../datasplit/test/9thEla/')
# extract_patch(AR_aug[6][1], OR_aug[6][1], 2000, 2000, 384, 384, 64, '../datasplit/test/9thNoi/')
# extract_patch(AR_aug[6][4], OR_aug[6][4], 2000, 2000, 384, 384, 64, '../datasplit/test/9thGa1.4/')
# extract_patch(AR_aug[7][6], OR_aug[7][6], 2000, 2000, 384, 384, 64, '../datasplit/test/10thEla/')
# extract_patch(AR_aug[7][1], OR_aug[7][1], 2000, 2000, 384, 384, 64, '../datasplit/test/10thNoi/')
# extract_patch(AR_aug[7][4], OR_aug[7][4], 2000, 2000, 384, 384, 64, '../datasplit/test/10thGa1.4/')

# del AR_aug, OR_aug
# gc.collect()


if __name__ == "__main__":

    #dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
    BATCH_SIZE = 2

    #Load data
    datadirs = ['2th','7th','8th','9th','10th'] #'9thEla','9thGa1.4','9thNoi','10thEla','10thGa1.4','10thNoi'
    # datadirs = ['ear_AR_2','ear_AR_3','ear_AR_4']
    plt.figure()
    for datadir in datadirs:
        print("Test on %s" % datadir)
        AR = np.load("../datasplit/test/%s/AR.npy" % datadir)
        OR = np.load("../datasplit/test/%s/OR.npy" % datadir)
        
        # extract_patch(AR, OR, 2000, 2000, 384, 384, 64, "../datasplit/test/%s/" % datadir)

        test_set = DatasetFromFolder('../datasplit/test/%s/x' % datadir, '../datasplit/test/%s/y' % datadir)

        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, 
                                        shuffle=False, num_workers=BATCH_SIZE)

       
        ARMean, ARStd = np.mean(AR), np.std(AR, ddof=1)
        ORMean, ORStd = np.mean(OR), np.std(OR, ddof=1) 
        AR_var, OR_var = AR.var(), OR.var()
        AR, OR = AR[0:6*384-5*64, 0:6*384-5*64], OR[0:6*384-5*64, 0:6*384-5*64]
        psnr, ssim, pcc = calculate_psnr(AR, OR), calculate_ssim(AR, OR), calculate_pcc(AR, OR)
        print("AR|OR, AR_mean: %.4f, AR_var: %.4f, OR_mean: %.4f, OR_var: %.4f, \
PSNR: %.6f, SSIM: %.6f, PCC: %.6f" % (ARMean, AR_var, ORMean, OR_var, psnr, ssim, pcc))          
    

        #Test using different SRGAN models
        G = Generator().cuda()
        D = Discriminator_WGAN((2,1,384,384),2).cuda()
        
        Loss_G = []
        for i in range(1,6):   
            test_result =  {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}

            #Load and initialize GAN      
            G.load_state_dict(torch.load("../datasplit/new_train/train8/Weights/G_model%d.pth" % i))
            D.load_state_dict(torch.load("../datasplit/new_train/train8/Weights/D_model%d.pth" % i))
            print("Use trained GAN model %d to test!" % i)
            
            with torch.no_grad(): 
                SR_imgs = torch.Tensor([]).cuda()
                G.eval()

                for x, y in test_loader:
                    AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(4,1,384,384)          
                    SR_img = G(AR_img) 

                    logits_SR, logits_OR = D(SR_img), D(OR_img)
                    G_loss = generator_loss(D(SR_img), OR_img, SR_img, 1, 1e-1)[0]
                    D_loss = discriminator_loss(D(OR_img),D(SR_img),0)

                    test_result['batch_sizes'] += BATCH_SIZE
                    test_result['g_loss'] += G_loss.item() * BATCH_SIZE
                    test_result['d_loss'] += D_loss.item() * BATCH_SIZE 

                    SR_img = SR_img.squeeze(1)
                    SR_imgs = torch.cat((SR_imgs, SR_img),0)

                SR_path = "../datasplit/test/%s/basic/train8/SR_cycle%d" % (datadir,i)
                # SRpatch_path = "../datasplit/test/7th/basic/train8/cycle%d/" % (i+1)
                SR = stitch_SavePatch(SR_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64)#SRpatch_path,
                
            #Test loss
            d_loss = test_result['d_loss'] / test_result['batch_sizes']
            g_loss = test_result['g_loss'] / test_result['batch_sizes']
            Loss_G.append(g_loss)
            print("SR%d| Loss_D:%.6f, Loss_G:%.6f" % (i,d_loss,g_loss))

            # Image assessment metrics  
            mean, var = SR.mean(), SR.var()
            psnr, ssim, pcc = calculate_psnr(SR, OR), calculate_ssim(SR, OR), calculate_pcc(SR, OR)
            print("Mean: %.4f, Var: %.4f, PSNR: %.6f, SSIM: %.6f, PCC: %.6f\n" % ( mean, var, psnr, ssim, pcc))

        
        plt.plot(range(len(Loss_G)), Loss_G, label='Test on %s' % datadir)

    plt.grid(True, linestyle='-')
    plt.legend(loc = 'lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss_G')
    plt.title("Generator Loss during test")
    plt.show()
    
    print("Finish test!\n")
