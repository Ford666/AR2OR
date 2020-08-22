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
from math import fabs, sin, cos, radians
from utils import *

'''
Given AR/OR aug of size (8,288,2000,2000), through 288 different combinations of 5 transforms
To Obtain AR/OR train/val/test set of size (N,256,256) and dtype is np.float16
'''

f = open('../Obtain dataset record.txt','w+')

AR_aug =  np.load('../Data_aug/AR_aug.npy') #(8,288,2000,2000)
OR_aug =  np.load('../Data_aug/OR_aug.npy') #(8,288,2000,2000)

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
    if np.mean(img2[0:rows, 0:cols]) < np.mean(img2[0:cols, 0:rows]):
        rows, cols = cols, rows
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
            if density < 3/5*density_mean: # threshold for screening
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

f.write("Starting Dividing into training/val/test set" + '\n')
f.flush()

data_coord = [] #[8,288,[()]] Nested list of coordinate sets of extracted patches

parms, parm = [], {}

for i in range(4):
    for j in range(3):
        for k in range(4):
            for m in range(3):
                for n in range(2):
                    parm = {'flip_option':i, 'rot_option':j, 
                                'ela_parm':{'sigma':6, 'seed':20*(k-1)+50}, 'gamma_option':m, 'noise_option':n}
                    parms.append(parm)

train_lookup, val_lookup, test_lookup = [], [], []
train_sum, val_sum, test_sum = 0, 0, 0
for i in range(8):
    data_coord.append([])
                
    if i == 5:  # val set
        f.write('Val set (8th pair) begin' + '\n')
        f.flush()
        for j in range(288):           
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush() 

            mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64) 
            data_len = mixed_data[0]
            data_coord[i].append(mixed_data[1])
            val_sum += data_len
            for k in range(data_len):
                test_lookup.append({'NO.':val_sum-data_len+k+1, 'Origin': i+1})
                test_lookup[val_sum-data_len+k].update(parms[j])
                test_lookup[val_sum-data_len+k]['Coord'] = mixed_data[1][k]

    elif i == 6:  # Test set
        f.write('Test set (9th pair) begin' + '\n')
        f.flush()
        for j in range(288):           
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush() 

            mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64) 
            data_len = mixed_data[0]
            data_coord[i].append(mixed_data[1])
            test_sum += data_len
            for k in range(data_len):
                val_lookup.append({'NO.':test_sum-data_len+k+1, 'Origin': i+1})
                val_lookup[test_sum-data_len+k].update(parms[j])
                val_lookup[test_sum-data_len+k]['Coord'] = mixed_data[1][k]
                
    else:  #Train set
        if i <= 2:
            train_str = "Train set (%dth pair) begin!\n" % (i+1)
        elif i == 3:
            train_str = "Train set (%dth pair) begin!\n" % (i+2)
        else:
            train_str = "Train set (%dth pair) begin!\n" % (i+3)
  
        f.write(train_str)
        f.flush()     
        for j in range(288):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()
            if i == 0: # 1st pair
                mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 1427, 256, 256, 64)  
            elif i == 2: # 3rd pair of images sized (1500,1500)
                mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 1500, 1500, 256, 256, 80) 
            elif i == 3: # 5th pair
                mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 1988, 1980, 256, 256, 80)
            else:
                mixed_data = Extract_Patch(AR_aug[i,j,:,:], OR_aug[i,j,:,:], 2000, 2000, 256, 256, 64)
            data_len = mixed_data[0]
            data_coord[i].append(mixed_data[1])
            train_sum += data_len
     
            # save training image patches
            for k in range(data_len):
                x_dir, y_dir = '../datasplit/training/x/'+str(train_sum-data_len+k+1).zfill(6)+'.npy', \
                                '../datasplit/training/y/'+str(train_sum-data_len+k+1).zfill(6)+'.npy'
                np.save(x_dir, np.squeeze(mixed_data[2][k]))
                np.save(y_dir, np.squeeze(mixed_data[3][k]))
                train_lookup.append({'NO.':train_sum-data_len+k+1, 'Origin': i+1})
                train_lookup[train_sum-data_len+k].update(parms[j])
                train_lookup[train_sum-data_len+k]['Coord'] = mixed_data[1][k]


# #选取弹性变换后的第9张大图裁出图像块，构成测试集
extract_patch(AR_aug[6][0], OR_aug[6][0], 2000, 2000, 256, 256, 64, '../datasplit/test/1/')
extract_patch(AR_aug[6][6], OR_aug[6][6], 2000, 2000, 256, 256, 64, '../datasplit/test/2/')
extract_patch(AR_aug[6][12], OR_aug[6][12], 2000, 2000, 256, 256, 64, '../datasplit/test/3/')

del AR_aug, OR_aug
gc.collect()

# save data_coord
coord_file = open('../data_coord.txt', 'w+')
for i in range(8):
    for j in range(288):
        for coord in data_coord[i][j]:
            coord_file.write(str(coord) + ',')
        coord_file.write('\n')
coord_file.close()

# save lookup tables

train_json = open('../lookup_train.json', 'w+')
for info in train_lookup:
    train_json.write(json.dumps(info)+'\n')
train_json.close()

val_json = open('../lookup_val.json', 'w+')
for info in val_lookup:
    val_json.write(json.dumps(info)+'\n')
val_json.close()

test_json = open('../lookup_test.json', 'w+')
for info in test_lookup:
    test_json.write(json.dumps(info)+'\n')
test_json.close()

set_len = [train_sum, val_sum, test_sum]
len_str =','.join(str(e) for e in set_len)
f.write("length of train/val/test: " + len_str + '\n')
f.flush()


f.write("All training/val/test set saved!")
f.flush()
f.close()






