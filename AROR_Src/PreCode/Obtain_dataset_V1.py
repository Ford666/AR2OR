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
Provided AR/OR aug of size (10,144,2000,2000), through 144 diierent combinations of 5 transforms
Obtain AR/OR train/val/test set of size (N,256,256) and dtype is np.float16

'''


f = open('../Obtain dataset record.txt','w+')

#AR_npy = np.load('../Data_aug/AR_npy.npy') 
#OR_npy = np.load('../Data_aug/OR_npy.npy')
#f.write ("Load data done: AR_npy(10,2000,2000), OR_npy(10,2000,2000)" + '\n')
#f.flush()

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


# 4 flipping * 3 rotation * 3 elastic deformation * 2 gamma trasform * 2 add noise = 144 
#AR_aug = np.zeros(shape=(10,144,2000,2000), dtype=np.float32)
#OR_aug = np.zeros(shape=(10,144,2000,2000), dtype=np.float32)
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

#for i in range(10):
#    for j in range(144):
#        AR_aug[i, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(AR_npy[i], 
#                      parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                       parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#        OR_aug[i, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(OR_npy[i], 
#                      parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                       parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
    
#np.save('../Data_aug/AR_aug.npy', AR_aug) 
#np.save('../Data_aug/OR_aug.npy', OR_aug)

#f.write ("Data augmentation done: AR_aug(10,144,2000,2000), OR_aug(10,144,2000,2000)")
#f.flush()

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
    #density_std = np.std(img_remove, ddof=1)
    f.write ("mean of OR PAM vessel density is: " + str(density_mean) + '\n')
    f.flush()

    # extract AR and OR image patch given OR PAM vessel density
    patch_num = (rows-overlap)//(patchH-overlap) * (cols-overlap)//(patchW-overlap)
    DataSet1, DataSet2 = [], []
    Coord = []
    
    for i in range((rows-overlap)//(patchH-overlap)):
        for j in range((cols-overlap)//(patchW-overlap)):
            density = np.mean(img_remove[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                       j*(patchW-overlap):(j+1)*patchW-j*overlap])
            if density < density_mean/6: # threshold for screening
                continue
            else:
                DataSet1.append(img1[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                j*(patchW-overlap):(j+1)*patchW-j*overlap])
                DataSet2.append(img2[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                    j*(patchW-overlap):(j+1)*patchW-j*overlap])
                Coord.append((i,j))
            

    DataSet1, DataSet2 = np.array(DataSet1), np.array(DataSet2)
    DataLen = len(DataSet2)
    f.write("Number of patches: " + str(DataLen) + '\n')
    f.flush()

    return DataLen, Coord, DataSet1, DataSet2


## Divide into training/val/test set 

AR_train = np.zeros(shape=(8*144*100,256,256), dtype=np.float16)
OR_train = np.zeros(shape=(8*144*100,256,256), dtype=np.float16)

AR_val = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)
OR_val = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)

AR_test = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)
OR_test = np.zeros(shape=(1*144*100,256,256), dtype=np.float16)

f.write("Starting Dividing into training/val/test set" + '\n')
f.flush()

data_coord = [] #[7,144,[()]] Nested list of coordinate sets of extracted patches

train_sum, val_sum, test_sum = 0, 0, 0
for i in range(10):
    data_coord.append([])

    if i == 1:    # Val set
        f.write('Val set begin' + '\n')
        f.flush()
        for j in range(144):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()

            mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64) 
            data_len = mixed_data[0]
            data_coord[i].append(mixed_data[1])
            val_sum += data_len
            AR_val[val_sum-data_len:val_sum,:,:], OR_val[val_sum-data_len:val_sum,:,:] = mixed_data[2], mixed_data[3]

    elif i == 6:  # Test set
        f.write('Test set begin' + '\n')
        f.flush()
        for j in range(144):           
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush() 

            mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64) 
            data_len = mixed_data[0]
            data_coord[i].append(mixed_data[1])
            test_sum += data_len
            AR_test[test_sum-data_len:test_sum,:,:], OR_test[test_sum-data_len:test_sum,:,:] = mixed_data[2], mixed_data[3]

    else:  #Train set
        f.write('Train set begin' + '\n')
        f.flush()     
        for j in range(144):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()
            if i == 2:
                mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 1567, 1567, 256, 256, 70) # 8*8 patches
            else:
                mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64)
            data_len = mixed_data[0]
            data_coord[i].append(mixed_data[1])
            train_sum += data_len
            AR_train[train_sum-data_len:train_sum,:,:], OR_train[train_sum-data_len:train_sum,:,:] = mixed_data[2], mixed_data[3]

# save data_coord
coord_file = open('../data_coord.txt', 'w+')
for i in range(10):
    for j in range(144):
        for coord in data_coord[i][j]:
            coord_file.write(str(coord) + ',')
        coord_file.write('\n')
coord_file.close()

set_len = [train_sum, val_sum, test_sum]
len_str =','.join(str(e) for e in set_len)
f.write("length of train/val/test: " + len_str + '\n')
f.flush()

# Save tain/val/test set 
np.save('../Data_aug/AR_train.npy', AR_train)
np.save('../Data_aug/AR_val.npy', AR_val)
np.save('../Data_aug/AR_test.npy', AR_test)

np.save('../Data_aug/OR_train.npy', OR_train)
np.save('../Data_aug/OR_val.npy', OR_val)
np.save('../Data_aug/OR_test.npy', OR_test)

f.write("All training/val/test set saved!")
f.flush()
f.close()





