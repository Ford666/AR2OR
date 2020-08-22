# -*- coding: utf-8 -*
import matplotlib.image as mpimg 
from utils import *
from PIL import Image

# DataName = ['depth-0','depth-300','depth-700','depth-1000','depth-1300','depth-1700'] #depth
# DataName = ['AR1','AR2','AR3','AR4','day2.1','day2.2'] #skin tumor
# Num = len(DataName)


AR = np.load("../datasplit/test/7th/AR.npy")
OR = np.load("../datasplit/test/7th/OR.npy")
ARMean, ARStd = np.mean(AR), np.std(AR, ddof=1)
ORMean, ORStd = np.mean(OR), np.std(OR, ddof=1) 
AR_var, OR_var = AR.var(), OR.var()  
psnr, ssim, pcc = calculate_psnr(AR, OR), calculate_ssim(AR, OR), calculate_pcc(AR, OR)
print('psnr:%f, ssim:%f, pcc:%f' % (psnr,ssim,pcc))

SR1 = Image.open('../datasplit/new_train/train6/0.01/SR1.png').convert('L')
SR1 = np.array(SR1,dtype=np.float32)
plt.figure(1)
plt.imshow(SR1)
SR2 = Image.open('../datasplit/new_train/train6/0.001/SR1.png').convert('L')
SR2 = np.array(SR2,dtype=np.float32)
plt.figure(2)
plt.imshow(SR2)
SR3 = Image.open('../datasplit/new_train/train6/0.0001/SR1.png').convert('L')
SR3 = np.array(SR3,dtype=np.float32)
plt.figure(3)
plt.imshow(SR3)
plt.show()
OR = OR[0:6*384-5*64, 0:6*384-5*64]

psnr1, ssim1, pcc1 = calculate_psnr(SR1, OR), calculate_ssim(SR1, OR), calculate_pcc(SR1, OR)
print('psnr:%f, ssim:%f, pcc:%f' % (psnr1,ssim1,pcc1))
psnr2, ssim2, pcc2 = calculate_psnr(SR2, OR), calculate_ssim(SR2, OR), calculate_pcc(SR2, OR)
print('psnr:%f, ssim:%f, pcc:%f' % (psnr2,ssim2,pcc2))
psnr3, ssim3, pcc3 = calculate_psnr(SR3, OR), calculate_ssim(SR3, OR), calculate_pcc(SR3, OR)
print('psnr:%f, ssim:%f, pcc:%f' % (psnr3,ssim3,pcc3))

# for i in range(5):
#     # path = '../datasplit/test/Tumor_ARdata/skin tumor/%s/' % DataName[i] + 'SR.jpg'
#     path = '../datasplit/test/10th/SRGAN_esemble/test7/'+ 'SR_cycle%d.tiff' % (i+1) 
#     # SR = Image.open(path).convert('L')
#     SR = mpimg.imread(path)
#     SR = np.array(SR,dtype=np.float32)
#     SRmin, SRmax =  np.amin(SR), np.amax(SR)
#     SR = np.divide(SR-SRmin, SRmax-SRmin)
    
#     # quantitative image assessment  
#     [H,W] = SR.shape  
#     mean, var = SR.mean(), SR.var()
#     ARTemp, ORTemp = AR[0:H,0:W], OR[0:H,0:W]
#     psnr, ssim, pcc = calculate_psnr(ARTemp, ORTemp), calculate_ssim(ARTemp, ORTemp), calculate_pcc(ARTemp, ORTemp)
#     print("AR|OR, AR_mean: %.4f, AR_var: %.4f, OR_mean: %.4f, OR_var: %.4f, \
#     PSNR: %.6f, SSIM: %.6f, PCC: %.6f\n" % (ARMean, AR_var, ORMean, OR_var, psnr, ssim, pcc))
#     psnr, ssim, pcc = calculate_psnr(SR, ORTemp), calculate_ssim(SR, ORTemp), calculate_pcc(SR, ORTemp)
#     print("SR%d| Mean: %.4f, Var: %.4f, PSNR: %.6f, SSIM: %.6f, PCC: %.6f\n" % (i+1, mean, var, psnr, ssim, pcc))
#     path = '../datasplit/test/10th/SRGAN_esemble/test7/'+ 'SR_cycle%d_CO.tiff' % (i+1) 
#     plt.imsave(path, SR, cmap=plt.cm.hot)
