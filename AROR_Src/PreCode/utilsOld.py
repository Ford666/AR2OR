import torch
from torch.utils.data import Dataset
import numpy as np
import os
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
import math
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndi
from scipy import misc
import scipy.io as scio
import h5py
from PIL import Image
from skimage import color, measure, morphology
import json
from math import fabs, sin, cos, radians

# Custom Dataset class
class DatasetFromFolder(Dataset):
    def __init__(self, x_dir, y_dir):
        super(DatasetFromFolder, self).__init__()
        self.x_filenames = [join(x_dir, x) for x in listdir(x_dir)]
        self.y_filenames = [join(y_dir, y) for y in listdir(y_dir)]

    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.x_filenames[index])).float()
        y = torch.from_numpy(np.load(self.y_filenames[index])).float()
        return x, y

    def __len__(self):
        return len(self.x_filenames)

# Cyclic Cosine Annealing lr schedule
def snapshot_lr(lr_init, lr_end, iters, iter_per_cycle):
    step_no = iters - 1
    lr = lr_end + (lr_init-lr_end)/2 * (np.cos(np.pi*(step_no % iter_per_cycle)/iter_per_cycle)+1)
    return lr

#Show images during training
def show_images(images, path):
    images = images.data.cpu().numpy()
    
    gs = gridspec.GridSpec(1, 2) 
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.squeeze(img)) 
    if path != None:
        plt.savefig(path, bbox_inches='tight') #save plot instead of image itself
    plt.close()


# Image patch extraction 
def extract_patch(img1, img2, rows, cols, patchH, patchW, overlap, path):
    ImgMean1, ImgStd1 = np.mean(img1), np.std(img1, ddof=1)
    plt.imsave(path+'AR.tiff', img1, cmap=cm.hot, vmin=ImgMean1-ImgStd1, vmax=1)
    # np.save(path+'AR.npy', img1)
    ImgMean2, ImgStd2 = np.mean(img2), np.std(img2, ddof=1)
    plt.imsave(path+'OR.tiff', img2, cmap=cm.hot, vmin=ImgMean2-ImgStd2, vmax=1)
    # np.save(path+'OR.npy', img2)

    count = 0
    for i in range((rows-overlap)//(patchH-overlap)):
        for j in range((cols-overlap)//(patchW-overlap)):
            count += 1
            np.save(path+'x/'+str(count).zfill(3)+'.npy', img1[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                            j*(patchW-overlap):(j+1)*patchW-j*overlap])
            np.save(path+'y/'+str(count).zfill(3)+'.npy', img2[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                j*(patchW-overlap):(j+1)*patchW-j*overlap])
            plt.imsave(path+'AR patch/'+str(count).zfill(3)+'.tiff', img1[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                j*(patchW-overlap):(j+1)*patchW-j*overlap], cmap='hot')
            plt.imsave(path+'OR patch/'+str(count).zfill(3)+'.tiff', img2[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                j*(patchW-overlap):(j+1)*patchW-j*overlap], cmap='hot')

    return 

def extract_TestPatch(img, rows, cols, patchH, patchW, overlap, path):

    ImgMean, ImgStd = np.mean(img), np.std(img, ddof=1)
    plt.imsave(path+'AR.png', img, cmap=cm.hot)#, vmin=ImgMean-ImgStd, vmax=1
    np.save(path+'AR.npy', img)

    count = 0
    for i in range((rows-overlap)//(patchH-overlap)):
        for j in range((cols-overlap)//(patchW-overlap)):
            count += 1
            np.save(path+'x/'+str(count).zfill(3)+'.npy', img[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                            j*(patchW-overlap):(j+1)*patchW-j*overlap])
            plt.imsave(path+'AR patch/'+str(count).zfill(3)+'.tiff', img[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                j*(patchW-overlap):(j+1)*patchW-j*overlap], cmap='hot')
    return

#Image patch stitch
def stitch_patch(path, patchs, rows, cols, patchH, patchW, overlap):
      
    imgs = np.zeros([rows, cols])

    numW, numH = (rows-overlap)//(patchH-overlap), (cols-overlap)//(patchW-overlap)
    count = 0
    for i in range(numW):
        for j in range(numH):
            imgs[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                      j*(patchW-overlap):(j+1)*patchW-j*overlap] = patchs[count]
            count += 1
    imgs = imgs[0:(i+1)*patchH-i*overlap, 0:(j+1)*patchW-j*overlap]
    # np.save(path+'.npy', imgs)
    # scio.savemat(path+'.mat', {'SR':imgs})
    ImgMean, ImgStd = np.mean(imgs), np.std(imgs, ddof=1)
    plt.imsave(path+'.png', imgs, cmap=cm.hot) # , vmin=0, vmax=ImgMean-ImgStd
    return imgs

def stitch_SavePatch(SR_path, patchs, rows, cols, patchH, patchW, overlap):  # patch_path,
    imgs = np.zeros([rows, cols])

    numW, numH = (rows-overlap)//(patchH-overlap), (cols-overlap)//(patchW-overlap)
    count = 0
    for i in range(numW):
        for j in range(numH):
            imgs[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                      j*(patchW-overlap):(j+1)*patchW-j*overlap] = patchs[count]
            # misc.imsave(patch_path + str(count+1).zfill(2) + '.tiff', patchs[count])
            count += 1 
    imgs = imgs[0:(i+1)*patchH-i*overlap, 0:(j+1)*patchW-j*overlap]
    # np.save(path+'.npy', imgs)
    # scio.savemat(path+'.mat', {'SR':imgs})
    ImgMean, ImgStd = np.mean(imgs), np.std(imgs, ddof=1)
    imgs[imgs<0] = 0
    plt.imsave(SR_path+'.tiff', np.power(imgs,0.8), cmap=cm.hot) # cmap=cm.hot, vmin=0, vmax=ImgMean-ImgStd
    return imgs

def ssim(img1, img2):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose()) # flatten inputs into 1-dimensional for matrix multiplication

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[1] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[1] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))
    
def calculate_pcc(x,y):
    '''
    To calculate PCC between two images(2D ndarray) 
    x: distorted image, y: ground truth
    '''
    if x.ndim == 2:
        x_mean, y_mean = np.mean(x), np.mean(y)
        vx, vy = (x-x_mean), (y-y_mean)
        sigma_xy = np.mean(vx*vy)
        sigma_x, sigma_y = np.std(x, ddof=0), np.std(y, ddof=0)
        PCC = sigma_xy / ((sigma_x+1e-8) * (sigma_y+1e-8))
        return PCC.mean()
    elif x.ndim == 4:
        x_mean, y_mean = torch.mean(x, dim=[2,3], keepdim=True), torch.mean(y, dim=[2,3], keepdim=True)
        vx, vy = (x-x_mean), (y-y_mean)
        sigma_xy = torch.mean(vx*vy, dim=[2,3])
        sigma_x, sigma_y = torch.std(x, dim=[2,3]), torch.std(y, dim=[2,3])   
        PCC = sigma_xy / ((sigma_x+1e-8) * (sigma_y+1e-8))
        return PCC.mean().item()  
    else:
        raise ValueError("ndim error!")  


class TestDataFromFolder(Dataset):
    def __init__(self, x_dir):
        super(TestDataFromFolder, self).__init__()
        self.x_filenames = [join(x_dir, x) for x in listdir(x_dir)]

    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.x_filenames[index])).float()
        return x

    def __len__(self):
        return len(self.x_filenames)

# Data argumentation functions
def Flip(img, options):
    """
    img: ndarry
    options: 0, 1, 2, 3 for different flipping
    p: probability, [0,1]
    """
    if options == 0:
        image = img  #no operation
    elif options == 1:
        image = cv2.flip(img, 0, dst=None) #vertical flipping
    elif options == 2:
        image = cv2.flip(img, 1, dst=None) #horizontal flipping(mirroring)
    elif options == 3:
        image = cv2.flip(img, -1, dst=None) #diagonal flipping
        
    return image

 
def New_Rotation(img, options):

    height,width=img.shape[:2]

    if options == 0:
        angle = 0
    elif options == 1:
        angle = 90
    elif options == 2:
        angle = 270

    heightNew=int(width*fabs(sin(radians(angle)))+height*fabs(cos(radians(angle)))) #旋转
    widthNew=int(height*fabs(sin(radians(angle)))+width*fabs(cos(radians(angle))))

    matRotation=cv2.getRotationMatrix2D((width/2,height/2),angle,1)

    matRotation[0,2] +=(widthNew-width)/2  #加入平移操作
    matRotation[1,2] +=(heightNew-height)/2  

    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew))
    return imgRotation

def Elastic_Deformation(img, sigma, seed):
    """
    img: ndarray
    alpha: scaling factor to control the deformation intensity
    sigma: elasticity coefficient
    seed: random integer from 1 to 100
    """  
    if seed > 40:
        alpha = 34
        random_state = np.random.RandomState(seed) 
        #image = np.pad(img, pad_size, mode="symmetric")
        center_square, square_size = np.float32(img.shape)//2, min(img.shape)//3

        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-img.shape[1]*0.08, img.shape[1]*0.08, size=pts1.shape).astype(np.float32)
        m = cv2.getAffineTransform(pts1,pts2)
        image = cv2.warpAffine(img, m, img.shape,  borderMode=cv2.BORDER_REFLECT_101) #random affine transform

        #generate random displacement fields by gaussian conv
        dx = dy = gaussian_filter((random_state.rand(*image.shape)*2 - 1),
                             sigma, mode="constant", cval=0) * alpha 
        x, y =np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])) #grid coordinate matrix
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        img_ela = map_coordinates(image, indices, order=1).reshape(*image.shape) #preform bilinear interpolation
    else:
        img_ela = img
    return img_ela


def Add_Gaussian_Noise(img, options):
    """
        image : numpy array of image
        standard deviation : pixel standard deviation of gaussian noise
    """
    if options:
        std = 1e-2
        img_mean = img.mean()
        gaus_noise = np.random.normal(img_mean, std, img.shape)
        noise_img = img + gaus_noise
        noise_img[noise_img<0] = 0
        noise_img[noise_img>1] = 1
    else:
        noise_img = img
    return noise_img

def Gamma_Transform(img, options): # We choose gamma value 0.8 as the gray transformation.
    """
    img: ndarray
    gamma: coefficient of gamma transform
    """
    if options == 0:
        gamma = 0.6
        img_gamma = np.power(img, gamma)
    elif options == 1:
        img_gamma = img 
    elif options == 2:
        gamma = 1.4
        img_gamma = np.power(img, gamma)
    return img_gamma

def Gamma_Transform1(img, gamma):
   """
   img: ndarray
   gamma: coefficient of gamma transform
   """
   img_gamma = np.power(img, gamma)

   return img_gamma    


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        '''
        os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候
        os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
        '''
        os.makedirs(path) 
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+ 'Directory existed!')

def Denoise(img, TempSize):
    #Recognize isolated pixel points, apply median filtering 
    rows, cols = img.shape
    img_denoise = img.copy()
    mean, std = np.mean(img.reshape(-1,1)), np.std(img.reshape(-1,1))
    threshold1, threshold2 = mean+4*std, mean+2*std  
    edge = int((TempSize-1)/2)
    for i in range(rows):
        for j in range(cols):
            if (img[i,j]>threshold1):
                if i-edge>=0 and i+edge<rows and j-edge>=0 and j+edge<cols:
                    Temp = img_denoise[i-edge:i+edge+1, j-edge:j+edge+1]
                    img_denoise[i,j] = np.median(Temp)  #孤立亮点进行中值滤波

                    TempMean, TempStd = np.mean(Temp), np.std(Temp)
                    Templen, ncount = TempSize**2, np.sum(Temp > threshold2)     
                    #斑块噪声较过曝血管区域像素灰度值差异更大
                    if TempStd>1.5*std or ncount<2/3*Templen:    
                        img_denoise[i, j] = 0  #img_denoise[i-edge:i+edge+1, j-edge:j+edge+1]
                    
    return img_denoise