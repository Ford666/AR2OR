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
from PIL import Image
import json
import gc  #garbage collector

'''
Provided AR/OR label of size (10,144,2000,2000), through same flipig, rotation and elastic transform, free of gamma and noise
To obtain AR/OR train/val/test labels of same size (N,256,256) as train/val/test sets, and same dtype np.float16

'''

f = open('../Obtain labels record.txt','w+')

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


def Gamma_Transform(img, options):
    """
    img: ndarray
    gamma: coefficient of gamma transform
    """
    if options:
        gamma = 0.8
        img_gamma = np.power(img, gamma)
        img_gamma[img>1] = 1
    else:
        img_gamma = img       
    return img_gamma


## 4 flipping * 3 rotation * 3 elastic deformation * 2 gamma trasform * 2 add noise = 144 

parms, parm = [], {}
for i in range(4):
    for j in range(3):
        for k in range(3):
            for m in range(2):
                for n in range(2):
                    parm = {'flip_option':i, 'rot_option':j, 
                                'ela_parm':{'sigma':6, 'seed':10*(k-1)+50}, 'gamma_option':m, 'noise_option':n}
                    parms.append(parm)


AR_aug =  np.load('../Data_aug/AR_aug.npy') #(10,144,2000,2000)
OR_aug =  np.load('../Data_aug/OR_aug.npy') #(10,144,2000,2000)

f.write("size of AR_aug: " + str(AR_aug.shape) + '\n')
f.write("size of OR_aug: " + str(OR_aug.shape) + '\n')
f.write ("Read augmented data done" + '\n')
f.flush()

#Image patch extraction based on vessel density sceening result
def Extract_Patch(img1, img2, rows, cols, patchH, patchW, overlap):
    """
    img: ndarray
    patch_size: touple, size of image patch 
    """
    # threshld segmentation
    img_uint8 = img2.copy()
    img_uint8 = np.rint(255 * img_uint8).astype(np.uint8)
    img_thr = cv2.adaptiveThreshold(img_uint8, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 201, -4)  # thresholding

    # remove small connected areas less than min_size
    img_remove = morphology.remove_small_objects(img_thr.astype(np.bool), min_size=10, connectivity=2)
    img_remove = img_remove.astype(np.int) #binary image
   
    #calculate  vessel density
    density_mean = np.mean(img_remove)
    density_std = np.std(img_remove, ddof=1)
    f.write ("mean of OR PAM vessel density is: " + str(density_mean) + '\n')
    f.flush()

    # extract AR and OR image patch given OR PAM vessel density
    patch_num = (rows-overlap)//(patchH-overlap) * (cols-overlap)//(patchW-overlap)

    Coord = []
    
    for i in range((rows-overlap)//(patchH-overlap)):
        for j in range((cols-overlap)//(patchW-overlap)):
            density = np.mean(img_remove[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                       j*(patchW-overlap):(j+1)*patchW-j*overlap])
            if density < density_mean/6: # threshold for screening
                continue
            else:

                Coord.append((i,j))
            
    return Coord

## Divide into training/val/test set 

f.write("Starting Dividing into training/val/test set" + '\n')
f.flush()

data_coord = [] #[7,144,[()]] Nested list of coordinate sets of extracted patches

#coord = [(i,j) for i in range(10) for j in range(10)] # example
#for i in range(10):
#    data_coord.append([])
#    for j in range(144):
#        data_coord[i].append(coord)

train_sum, val_sum, test_sum = 0, 0, 0
for i in range(10):
    data_coord.append([])

    if i == 1:    # Val set
        f.write('Val set begin' + '\n')
        f.flush()
        for j in range(144):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()

            coord_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64) 
            data_coord[i].append(coord_data)


    elif i == 6:  # Test set
        f.write('Test set begin' + '\n')
        f.flush()
        for j in range(144):           
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush() 

            coord_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64) 
            data_coord[i].append(coord_data)


    else:  #Train set
        f.write('Train set begin' + '\n')
        f.flush()     
        for j in range(144):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()
            if i == 2:
                coord_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 1567, 1567, 256, 256, 70) # 8*8 patches
            else:
                coord_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64)
            data_coord[i].append(coord_data)


# save data_coord
coord_file = open('../data_coord.txt', 'w+')
for i in range(10):
    for j in range(144):
        for coord in data_coord[i][j]:
            coord_file.write(str(coord) + ',', encoding = "utf-8")
        coord_file.write('\n',encoding = "utf-8")
coord_file.close()

set_len = [train_sum, val_sum, test_sum]
len_str =','.join(str(e) for e in set_len)
f.write("length of train/val/test: " + len_str + '\n')
f.flush()

del AR_aug, OR_aug
gc.collect()

# Obtain train/val/test labels

#AR_label = np.zeros(shape=(10,144,2000,2000), dtype=np.float32)
#OR_label = np.zeros(shape=(10,144,2000,2000), dtype=np.float32)

parms_label, parm = [], {}
for i in range(4):
    for j in range(3):
        for k in range(3):
            for m in range(2):
                for n in range(2):
                    parm = {'flip_option':i, 'rot_option':j, 
                                'ela_parm':{'sigma':6, 'seed':10*(k-1)+50}, 'gamma_option':0, 'noise_option':0}
                    parms_label.append(parm)

#for i in range(10):
#    for j in range(144):
#        AR_label[i, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(AR_npy[i], 
#                      parms_label[j]['flip_option']), parms_label[j]['rot_option']), parms_label[j]['ela_parm']['sigma'],
#                       parms_label[j]['ela_parm']['seed']), parms_label[j]['gamma_option']), parms_label[j]['noise_option'])
#        OR_label[i, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(OR_npy[i], 
#                      parms_label[j]['flip_option']), parms_label[j]['rot_option']), parms_label[j]['ela_parm']['sigma'],
#                       parms_label[j]['ela_parm']['seed']), parms_label[j]['gamma_option']), parms_label[j]['noise_option'])
    
#np.save('../Data_aug/AR_label.npy', AR_label) 
#np.save('../Data_aug/OR_label.npy', OR_label)
#f.write ("Data label done: AR_label(10,144,2000,2000), OR_label(10,144,2000,2000)"+ '\n')
#f.flush()

AR_label =  np.load('../Data_aug/AR_label.npy') #(10,144,2000,2000)
OR_label =  np.load('../Data_aug/OR_label.npy') #(10,144,2000,2000)
f.write ("Read data label done: AR_label(10,144,2000,2000), OR_label(10,144,2000,2000)" + '\n')
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


f.write("Starting finding training/val/test label" + '\n')
f.flush()

AR_train_label = np.zeros(shape=(8*144*100,256,256), dtype=np.float16)
OR_train_label = np.zeros(shape=(8*144*100,256,256), dtype=np.float16)

AR_val_label = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)
OR_val_label = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)

AR_test_label = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)
OR_test_label = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)


train_label_sum, val_label_sum, test_label_sum = 0, 0, 0
for i in range(10):
    if i == 1:    # Val label
        f.write('Val label begin' + '\n')
        f.flush()
        for j in range(53,54):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()
            mixed_data = Find_PatchLabel(AR_label[i,j,:,:],OR_label[i,j,:,:], 256, 256, 64, data_coord[i][j],parms_label[j]) 
            data_len = mixed_data[0]
            val_label_sum += data_len
            AR_val_label[val_label_sum-data_len:val_label_sum,:,:], \
                       OR_val_label[val_label_sum-data_len:val_label_sum,:,:] = mixed_data[1], mixed_data[2]

    elif i == 6:  # Test label
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
            if i == 2:
                mixed_data = Find_PatchLabel(AR_label[i,j,:,:],OR_label[i,j,:,:], 256, 256, 70, data_coord[i][j],parms_label[j]) # 8*8 patches
            else:
                mixed_data = Find_PatchLabel(AR_label[i,j,:,:],OR_label[i,j,:,:], 256, 256, 64, data_coord[i][j],parms_label[j])
             
            data_len = mixed_data[0]
            train_label_sum += data_len
            AR_train_label[train_label_sum-data_len:train_label_sum,:,:], \
                       OR_train_label[train_label_sum-data_len:train_label_sum,:,:] = mixed_data[1], mixed_data[2]

label_len = [train_label_sum, val_label_sum, test_label_sum]
len_str =','.join(str(e) for e in label_len)
f.write("length of train/val/test label: " + len_str + '\n')
f.flush()



# Save tain/val/test labels 

np.save('../Data_aug/AR_train_label.npy', AR_train_label)
np.save('../Data_aug/AR_val_label.npy', AR_val_label)
np.save('../Data_aug/AR_test_label.npy', AR_test_label)

np.save('../Data_aug/OR_train_label.npy', OR_train_label)
np.save('../Data_aug/OR_val_label.npy', OR_val_label)
np.save('../Data_aug/OR_test_label.npy', OR_test_label)

f.write("All training/val/test labels saved!")
f.flush()
f.close()





