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

'''
Given AR/OR aug of size (10,144,2000,2000), through 144 diierent combinations of 5 transforms
To Obtain AR/OR train/val/test set of size (N,256,256) and dtype is np.float16
'''

f = open('../Obtain dataset record.txt','w+')

AR_aug =  np.load('../Data_aug/AR_aug.npy') #(10,144,2000,2000)
OR_aug =  np.load('../Data_aug/OR_aug.npy') #(10,144,2000,2000)

f.write("size of AR_aug: " + str(AR_aug.shape) + '\n')
f.write("size of OR_aug: " + str(OR_aug.shape) + '\n')
f.write ("Read augmented data done" + '\n')
f.flush()

# Image patch extraction based on vessel density sceening result
def Extract_Patch(img1, img2, rows, cols, patchH, patchW, overlap):
    """
    img: ndarray
    patch_size: touple, size of image patch 
    """
    # threshld segmentation
    img_uint8 = img2[0:rows, 0:cols].copy()
    img_uint8 = np.rint(255 * img_uint8).astype(np.uint8)
    img_thr = cv2.adaptiveThreshold(img_uint8, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 201, -4)  # thresholding

    # remove small connected areas less than min_size
    img_remove = morphology.remove_small_objects(img_thr.astype(np.bool), min_size=10, connectivity=2)
    img_remove = img_remove.astype(np.int) #binary image
   
    #calculate  vessel density
    density_mean = np.mean(img_remove)
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

data_coord = [] #[10,144,[()]] Nested list of coordinate sets of extracted patches

parms, parm = [], {}

for i in range(4):
    for j in range(3):
        for k in range(3):
            for m in range(2):
                for n in range(2):
                    parm = {'flip_option':i, 'rot_option':j, 
                                'ela_parm':{'sigma':6, 'seed':10*(k-1)+50}, 'gamma_option':m, 'noise_option':n}
                    parms.append(parm)

train_lookup, val_lookup, test_lookup = [], [], []
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
            for k in data_len:
                val_lookup[val_sum-data_len+k]['']
                val_lookup[val_sum-data_len+k] = parms[j]
                
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
            if i == 2: # 3rd pair of images sized (1567,1567)
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

del AR_aug, OR_aug
gc.collect()
   
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






