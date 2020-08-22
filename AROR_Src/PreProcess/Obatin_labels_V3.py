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
import re
from math import fabs, sin, cos, radians
import gc  #garbage collector

'''
Given AR_npy and OR_npy both of size (10,2000,2000), we obtain corresponding AR/OR train/val/test labels of same size (N,256,256) as train/val/test set.

This is done by first exerting same transformation combined with fliping, rotation and elastic transform, while free of gamma 
transform and adding noise, to get AR/OR labels of same size(8,144,2000,2000) as AR/OR_aug. Then with the recorded coordinates 
of extracted patches of train/val/test sets, we further exert corresponding inverse transformations to those patches regionally to obtain their labels.
'''

# Load 8 pairs of AR/OR PAM images
f = open('../Obtain labels record.txt','w+')
AR_npy = np.load('../Data_aug/AR_npyV2.npy') 
OR_npy = np.load('../Data_aug/OR_npyV2.npy')
f.write ("Load AR_npy and OR_npy done" + '\n')
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

def Gamma_Transform(img, options): # We choose gamma value 0.6 as the brinary transformation.
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


# read coordinates of patches in train/val/test sets
data_coord = [] #[8,144,[()]] Nested list of coordinate sets of extracted patches

coord_file = open('../data_coord.txt')
line = coord_file.readline()
for i in range(8):
    data_coord.append([])
    for j in range(144):
        if line:
            curline = list(line.split(','))[:-1]
            newline = iter(curline)
            strline = [(int(re.findall('\d+',e)[0]), int(re.findall('\d+',next(newline))[0])) for e in newline]
            data_coord[i].append(strline)
        line = coord_file.readline()

coord_file.close()

f.write ("Read done the recorded coordinates of extracted patches of train/val/test sets" + '\n')
f.flush()


# Obtain train/val/test labels

AR_label = np.zeros(shape=(8,144,2000,2000), dtype=np.float32)
OR_label = np.zeros(shape=(8,144,2000,2000), dtype=np.float32)

parms_label, parm = [], {}
for i in range(4):
    for j in range(3):
        for k in range(3):
            for m in range(2):
                for n in range(2):
                    parm = {'flip_option':i, 'rot_option':j, 
                                'ela_parm':{'sigma':6, 'seed':10*(k-1)+50}, 'gamma_option':0, 'noise_option':0}
                    parms_label.append(parm)
    
k = 0
for i in range(10):
    if k < 8:
        if i==3 or i ==5: # Abandon 4h and 6th pair
                continue
        elif i == 0:  # (2000,1427)
            for j in range(144):                             
                AR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(AR_npy[0,0:2000,0:1427], 
                        parms_label[j]['flip_option']), parms_label[j]['rot_option']), parms_label[j]['ela_parm']['sigma'],
                        parms_label[j]['ela_parm']['seed']), parms_label[j]['gamma_option']), parms_label[j]['noise_option'])
                AR_label[k,j,0:AR.shape[0], 0:AR.shape[1]] = AR
                OR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(OR_npy[0,0:2000,0:1427], 
                        parms_label[j]['flip_option']), parms_label[j]['rot_option']), parms_label[j]['ela_parm']['sigma'],
                        parms_label[j]['ela_parm']['seed']), parms_label[j]['gamma_option']), parms_label[j]['noise_option'])
                OR_label[k,j,0:OR.shape[0], 0:OR.shape[1]] = OR
            k += 1
        elif i == 2:  #(1500,1500)
            for j in range(144): 
                AR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(AR_npy[2,0:1500,0:1500], 
                        parms_label[j]['flip_option']), parms_label[j]['rot_option']), parms_label[j]['ela_parm']['sigma'],
                        parms_label[j]['ela_parm']['seed']), parms_label[j]['gamma_option']), parms_label[j]['noise_option'])
                AR_label[k,j,0:AR.shape[0], 0:AR.shape[1]] = AR
                OR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(OR_npy[2,0:1500,0:1500], 
                        parms_label[j]['flip_option']), parms_label[j]['rot_option']), parms_label[j]['ela_parm']['sigma'],
                        parms_label[j]['ela_parm']['seed']), parms_label[j]['gamma_option']), parms_label[j]['noise_option'])
                OR_label[k,j,0:OR.shape[0], 0:OR.shape[1]] = OR
            k += 1
        else:
            for j in range(144): 
                AR_label[k, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(AR_npy[i], 
                            parms_label[j]['flip_option']), parms_label[j]['rot_option']), parms_label[j]['ela_parm']['sigma'],
                            parms_label[j]['ela_parm']['seed']), parms_label[j]['gamma_option']), parms_label[j]['noise_option'])
                OR_label[k, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(OR_npy[i], 
                        parms_label[j]['flip_option']), parms_label[j]['rot_option']), parms_label[j]['ela_parm']['sigma'],
                        parms_label[j]['ela_parm']['seed']), parms_label[j]['gamma_option']), parms_label[j]['noise_option'])
            k += 1

np.save('../Data_aug/AR_label.npy', AR_label) 
np.save('../Data_aug/OR_label.npy', OR_label)
f.write ("Data label done: AR_label(8,144,2000,2000), OR_label(8,144,2000,2000)"+ '\n')
f.flush()

#AR_label =  np.load('../Data_aug/AR_label.npy') #(8,144,2000,2000)
#OR_label =  np.load('../Data_aug/OR_label.npy') #(8,144,2000,2000)
#f.write ("Read data labels done: AR_label(8,144,2000,2000), OR_label(8,144,2000,2000)" + '\n')
#f.flush()


f.write("Starting finding training/val/test label" + '\n')
f.flush()

def Find_PatchLabel(img1, img2, patchH, patchW, overlap, Coord, parm): # Without coordinate transform but only image transform
    '''
    img1,img2: ndarray
    Coord: list of coordinate sets of extracted patches
    parm: dict of transform parameters
    return one int-ype data and two ndarray: DataLen, img1_label, img2_label
    '''
    DataLen = len(Coord)
    DataSet1, DataSet2 = [], []
    flip_inv = parm['flip_option']
    rot_inv = (3-parm['rot_option'])*(parm['rot_option']!=0)

    for i in range(DataLen):
        DataSet1.append(Flip(Rotation(img1[Coord[i][0]*(patchH-overlap):(Coord[i][0]+1)*patchH-Coord[i][0]*overlap,
                                    Coord[i][1]*(patchW-overlap):(Coord[i][1]+1)*patchW-Coord[i][1]*overlap],rot_inv),flip_inv))

        DataSet2.append(Flip(Rotation(img2[Coord[i][0]*(patchH-overlap):(Coord[i][0]+1)*patchH-Coord[i][0]*overlap,
                                    Coord[i][1]*(patchW-overlap):(Coord[i][1]+1)*patchW-Coord[i][1]*overlap],rot_inv),flip_inv))
        
    
    DataSet1, DataSet2 = np.array(DataSet1), np.array(DataSet2)
    f.write("Number of patch labels: " + str(DataLen) + '\n')
    f.flush()

    return DataLen, DataSet1, DataSet2

AR_train_label = np.zeros(shape=(6*144*100,256,256), dtype=np.float16)
OR_train_label = np.zeros(shape=(6*144*100,256,256), dtype=np.float16)

AR_val_label = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)
OR_val_label = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)

AR_test_label = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)
OR_test_label = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)


train_label_sum, val_label_sum, test_label_sum = 0, 0, 0
for i in range(8):
    if i == 0:    # Val label, (2000,1427), patch overlap: 64
        f.write('Val label begin' + '\n')
        f.flush()
        for j in range(144):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()
            mixed_data = Find_PatchLabel(AR_label[i,j,:,:],OR_label[i,j,:,:], 256, 256, 64, data_coord[i][j],parms_label[j]) 
            data_len = mixed_data[0]
            val_label_sum += data_len
            AR_val_label[val_label_sum-data_len:val_label_sum,:,:], \
                       OR_val_label[val_label_sum-data_len:val_label_sum,:,:] = mixed_data[1], mixed_data[2]

    elif i == 4:  # Test label, (1500,1500), patch overlap: 64
        f.write('Test label begin' + '\n')
        f.flush()
        for j in range(144):           
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush() 
            mixed_data = Find_PatchLabel(AR_label[i,j,:,:],OR_label[i,j,:,:], 256, 256, 64, data_coord[i][j],parms_label[j]) 
            data_len = mixed_data[0]
            test_label_sum += data_len
            AR_test_label[test_label_sum-data_len:test_label_sum,:,:], \
                       OR_test_label[test_label_sum-data_len:test_label_sum,:,:] = mixed_data[1], mixed_data[2]
            
    else:      # Train label
        f.write('Train label begin' + '\n')
        f.flush()
        for j in range(144):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()
            if i == 2: # 3rd pair of images of size (1500,1500), patch overlap: 80
                mixed_data = Find_PatchLabel(AR_label[i,j,:,:],OR_label[i,j,:,:], 256, 256, 80, data_coord[i][j],parms_label[j]) # 8*8 patches
            else:
                mixed_data = Find_PatchLabel(AR_label[i,j,:,:],OR_label[i,j,:,:], 256, 256, 64, data_coord[i][j],parms_label[j])
             
            data_len = mixed_data[0]
            train_label_sum += data_len
            AR_train_label[train_label_sum-data_len:train_label_sum,:,:], \
                       OR_train_label[train_label_sum-data_len:train_label_sum,:,:] = mixed_data[1], mixed_data[2]

del AR_label, OR_label
gc.collect()

label_len = [train_label_sum, val_label_sum, test_label_sum]
len_str =','.join(str(e) for e in label_len)
f.write("length of train/val/test label: " + len_str + '\n')
f.flush()


# Save train/val/test labels 
np.save('../Data_aug/AR_train_label.npy', AR_train_label[0:train_label_sum])
np.save('../Data_aug/AR_val_label.npy', AR_val_label[0:val_label_sum])
np.save('../Data_aug/AR_test_label.npy', AR_test_label[0:test_label_sum])

np.save('../Data_aug/OR_train_label.npy', OR_train_label[0:train_label_sum])
np.save('../Data_aug/OR_val_label.npy', OR_val_label[0:val_label_sum])
np.save('../Data_aug/OR_test_label.npy', OR_test_label[0:test_label_sum])

f.write("All training/val/test labels saved!")
f.flush()
f.close()





