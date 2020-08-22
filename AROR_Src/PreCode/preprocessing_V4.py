# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import cv2 
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndi
from skimage import color, measure, morphology
import json
import gc  #garbage collector


f = open('../preprocessing record.txt','w+')
#Read the data
#AR_npy = np.zeros(shape=(10,2000,2000), dtype=np.float32)
#OR_npy = np.zeros(shape=(10,2000,2000), dtype=np.float32)

#for i in np.arange(10):
#   #dict_keys(['__header__', '__version__', '__globals__', 'C532'])
#   ARPAM_mat = scipy.io.loadmat('../PAaror/AR_'+str(i+1)+'.mat')
#   ORPAM_mat = scipy.io.loadmat('../PAaror/OR_'+str(i+1)+'.mat')
#   if i < 6:
#       AR_npy[i], OR_npy[i] = ARPAM_mat['C532'], ORPAM_mat['C532']
#   elif i == 6: 
#       AR_npy[i], OR_npy[i] = ARPAM_mat['C2'], ORPAM_mat['C1']
#   else: 
#       AR_npy[i], OR_npy[i] = ARPAM_mat['C1'], ORPAM_mat['C1']
  
#f.write ("Read data done" + '\n')
#f.flush()

## preprocessing 1: (0,1) standardization
#AR_min = (np.amin(np.amin(AR_npy,axis=1),axis=1)).reshape(10,1,1)
#AR_max = (np.amax(np.amax(AR_npy,axis=1),axis=1)).reshape(10,1,1) 
#OR_min = (np.amin(np.amin(OR_npy,axis=1),axis=1)).reshape(10,1,1) 
#OR_max = (np.amax(np.amax(OR_npy,axis=1),axis=1)).reshape(10,1,1)
#AR_npy,OR_npy = (AR_npy - AR_min) / (AR_max - AR_min), (OR_npy - OR_min) / (OR_max - OR_min)

#f.write ("Data standardization done" + '\n')
#f.flush()

#for i in range(10):
#   export_path1 = '../PAaror/AR_norm/'+str(i+1)+'.png'
#   mpimg.imsave(export_path1, AR_npy[i])

#   export_path2 = '../PAaror/OR_norm/'+str(i+1)+'.png'
#   mpimg.imsave(export_path2, OR_npy[i])

## preprocessing 2: recognize isolated pixel points, apply median filtering
#def Denoising(img, TempH, TempW):

#    rows, cols = img.shape
#    img_denoise = img.copy()
#    mean, std = np.mean(img.reshape(-1,1)), np.std(img.reshape(-1,1))
#    print("mean, std: ", mean, std)
#    threshold1, threshold2 = mean+std,  mean+3*std 

#    for i in range(rows):
#        for j in range(cols):
#            TempValue = []
#            nCount = 0
#            if (img[i,j] < threshold2):
#                continue
#            for m in range(-int((TempH-1)/2),int((TempH+1)/2)): #孤立亮点尺寸大概为(5,5）
#                for n in range(-int((TempW-1)/2),int((TempW+1)/2)):
#                    if (i+m)<0 or (i+m)>=2000 or (j+n)<0 or (j+n)>=2000:
#                        continue
#                    TempValue.append(img_denoise[i+m,j+n]) #一般列表有25个元素
#            lenTemp = len(TempValue)
#            nCount1 = sum(k < threshold1 for k in TempValue) 
#            nCount2 = sum(k > threshold2 for k in TempValue)
#            TempValue.sort()
#            if nCount1 > lenTemp/5 : #孤立亮点较边缘点而言，邻域内所含像素值陡低的点要多（一般占1/5以上）
#                if lenTemp % 2 == 1: #中值滤波
#                    img_denoise[i,j] = TempValue[int((lenTemp-1)/2)]
#                else:
#                    img_denoise[i,j] = (TempValue[int(lenTemp/2)]+TempValue[int(lenTemp/2-1)])/2
#            if img[i,j] > mean+8*std:
#                img_denoise[i,j] = 0
#    return img_denoise

#def Denoising1(img, TempH, TempW):

#    rows, cols = img.shape
#    img_denoise = img.copy()
#    mean, std = np.mean(img.reshape(-1,1)), np.std(img.reshape(-1,1))
#    print("mean, std: ", mean, std)
#    threshold1, threshold2 = mean+std,  mean+3*std 

#    for i in range(rows):
#        for j in range(cols):
#            TempValue = []
#            nCount = 0
#            if (img[i,j] < threshold2):
#                continue
#            for m in range(-int((TempH-1)/2),int((TempH+1)/2)): #孤立亮点尺寸大概为(5,5）
#                for n in range(-int((TempW-1)/2),int((TempW+1)/2)):
#                    if (i+m)<0 or (i+m)>=2000 or (j+n)<0 or (j+n)>=2000:
#                        continue
#                    TempValue.append(img_denoise[i+m,j+n]) #一般列表有25个元素
#            lenTemp = len(TempValue)
#            nCount1 = sum(k < threshold1 for k in TempValue) 
#            nCount2 = sum(k > threshold2 for k in TempValue)
#            TempValue.sort()
#            if nCount1 > lenTemp/5 : #孤立亮点较边缘点而言，邻域内所含像素值陡低的点要多（一般占1/5以上）
#                if lenTemp % 2 == 1: #中值滤波
#                    img_denoise[i,j] = TempValue[int((lenTemp-1)/2)]
#                else:
#                    img_denoise[i,j] = (TempValue[int(lenTemp/2)]+TempValue[int(lenTemp/2-1)])/2
#            if img[i,j] > mean+4*std and nCount2 < lenTemp*2/3: #孤立亮点不如过曝血管区域的整体像素值都高
#                img_denoise[i,j] = 0
#    return img_denoise

#for i in np.arange(10):
#   if i <= 6 and i != 2:
#       AR_npy[i] = Denoising(AR_npy[i],5,5)
#       OR_npy[i] = Denoising(OR_npy[i],5,5)
#   else:
#       AR_npy[i] = Denoising1(AR_npy[i],5,5)
#       OR_npy[i] = Denoising1(OR_npy[i],5,5)


#for i in np.arange(10):
#   export_path1 = '../PAaror/AR_denoise/'+str(i+1)+'.png'
#   mpimg.imsave(export_path1, AR_npy[i])

#   export_path2 = '../PAaror/OR_denoise/'+str(i+1)+'.png'
#   mpimg.imsave(export_path2, OR_npy[i])


#f.write ("Recognize isolated pixels done" + '\n')
#f.flush()



#np.save('../Data_aug/AR_npy.npy', AR_npy) 
#np.save('../Data_aug/OR_npy.npy', OR_npy)
#f.write ("Preprocessing data saved: AR_npy(10,2000,2000), OR_npy(10,2000,2000)" + '\n')
#f.flush()

AR_npy = np.load('../Data_aug/AR_npy.npy') 
OR_npy = np.load('../Data_aug/OR_npy.npy')

# For the 3rd pair of AR/OR images, crop the area of size (1567,1567) 
AR_match = AR_npy[2,177:2000,100:2000].copy()
OR_match = OR_npy[2,0:1823,0:1900].copy()
AR3_match, OR3_match = AR_match[256:1823, 0:1567], OR_match[256:1823, 0:1567] #(1567,1567)

AR_npy[2], OR_npy[2] = 0, 0
AR_npy[2,0:AR3_match.shape[0],0:AR3_match.shape[1]], OR_npy[2,0:OR3_match.shape[0],0:OR3_match.shape[1]] = AR3_match, OR3_match


#export_path3 = '../PAaror/AR_denoise/'+str(3)+'_new.png'
#export_path4 = '../PAaror/OR_denoise/'+str(3)+'_new.png'
#mpimg.imsave(export_path3, AR3_match)
#mpimg.imsave(export_path4, OR3_match)

f.write ("Load data done: AR_npy(10,2000,2000), OR_npy(10,2000,2000)" + '\n')
f.flush()

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


def Rotation(img, options):
    """
    img: ndarray
    angle: rotation angle: 0,90,180,270
    """
    rows, cols = img.shape
    if options == 0:
        angle = 0
    elif options == 1:
        angle = 90
    elif options == 2:
        angle = 270

    rot_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    image = cv2.warpAffine(img,rot_matrix,(rows, cols))
    return image
 


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

    #return Cropping(map_coordinates(image, indices, order=1).reshape(rows,cols), 
    #                 rows, pad_size, pad_size)


def Crop_Pad(img, options):
    """
    img: ndarray
    crop_size: size of cropped image
    options: 1, 2, 3, 4
    """
    crop_size0, crop_size1 = int(img.shape[0] * 0.7), int(img.shape[1] * 0.7)
    if options == 1: #left top
        cropped_img = img[0:crop_size0, 0:crop_size1]
    elif options == 2: #left bottom
        cropped_img = img[img.shape[0]-crop_size0:img.shape[0], 0:crop_size1]
    elif options == 3: #right top
        cropped_img = img[0:crop_size0, img.shape[1]-crop_size1:img.shape[1]]
    elif options == 4: #right bottom
        cropped_img = img[img.shape[0]-crop_size0:img.shape[0], img.shape[1]-crop_size1:img.shape[1]]

    pad_size = int(img.shape[0] * 0.15)
    return np.pad(cropped_img, pad_size, mode="symmetric")


def Add_Gaussian_Noise(img, options):
    """
        image : numpy array of image
        standard deviation : pixel standard deviation of gaussian noise
    """
    if options:
        std = 1e-3
        img_mean = np.mean(img.reshape(-1,1))
        gaus_noise = np.random.normal(img_mean, std, img.shape)
        noise_img = img + gaus_noise
        noise_img[noise_img<0] = 0
        noise_img[noise_img>1] = 1
    else:
        noise_img = img
    return noise_img

def Gamma_Transform(img, options): # We choose gamma value 0.8 as the brinary transformation.
    """
    img: ndarray
    gamma: coefficient of gamma transform
    """
    if options:
        gamma = 0.8
        img_gamma = np.power(img, gamma)
    else:
        img_gamma = img       
    return img_gamma

#def Gamma_Transform1(img, gamma):
#    """
#    img: ndarray
#    gamma: coefficient of gamma transform
#    """
#    img_gamma = np.power(img, gamma)

#    return img_gamma

#gamma_value  = [0.4,0.6,0.8,1,2] # Compare brightness shift of image for different gamma values 
#plt.figure
#plt.tight_layout()
#for i in range(5):
#    plt.subplot(2,5,i+1)
#    plt.imshow(Gamma_Transform1(AR_npy[1], gamma_value[i]))
#    plt.title("ARPAM, gamma="+str(gamma_value[i]))

#for i in range(5):
#    plt.subplot(2,5,i+6)
#    plt.imshow(Gamma_Transform1(OR_npy[1], gamma_value[i]))
#    plt.title("ORPAM, gamma="+str(gamma_value[i]))
#plt.show()

## 4 flipping * 3 rotation * 3 elastic deformation * 2 gamma trasform * 2 add noise = 144 
AR_aug = np.zeros(shape=(10,144,2000,2000), dtype=np.float32)
OR_aug = np.zeros(shape=(10,144,2000,2000), dtype=np.float32)
parms, parm = [], {}
for i in range(4):
    for j in range(3):
        for k in range(3):
            for m in range(2):
                for n in range(2):
                    parm = {'flip_option':i, 'rot_option':j, 
                                'ela_parm':{'sigma':6, 'seed':10*(k-1)+50}, 'gamma_option':m, 'noise_option':n}
                    parms.append(parm)

# save parms list as json file
parms_json = open('../parms.json', 'w+')
for i in parms:
    parms_json.write(json.dumps(i)+'\n')
parms_json.close()

for i in range(10):
    for j in range(144):
        if i == 2: #only AR/OR_3 with size(1567,1567)
            AR3_aug = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(AR3_match, 
                    parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
                    parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
            AR_aug[2,j,0:AR3_aug.shape[0], 0:AR3_aug.shape[1]] = AR3_aug
            OR3_aug = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(OR3_match, 
                    parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
                    parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
            OR_aug[2,j,0:OR3_aug.shape[0], 0:OR3_aug.shape[1]] = OR3_aug
        else:
            AR_aug[i, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(AR_npy[i], 
                          parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
                           parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
            OR_aug[i, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(OR_npy[i], 
                          parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
                           parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
    
np.save('../Data_aug/AR_aug.npy', AR_aug) 
np.save('../Data_aug/OR_aug.npy', OR_aug)

## To see the results of data augumentation of 3rd pair AR/OR images.
#for i in range(144):
#    path1 = '../PAaror/AR3_144/'+str(i+1)+'.png'
#    path2 = '../PAaror/OR3_144/'+str(i+1)+'.png'
#    mpimg.imsave(path1, AR_aug[2,i,:,:])
#    mpimg.imsave(path2, OR_aug[2,i,:,:])

f.write ("Data augmentation done: AR_aug(10,144,2000,2000), OR_aug(10,144,2000,2000)")
f.flush()
f.close()





