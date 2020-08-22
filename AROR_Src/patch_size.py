# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import cv2 
from skimage import color, measure, morphology
import json
import gc  #garbage collector
from utils import *


'''Given AR/OR aug of size (8,288,2000,2000), through 288 different combinations of 5 transforms.
To obtain the distribution of patch vessel density given a certain size.'''

f = open('../Patch size record.txt','w+')

AR_aug =  np.load('../Data_aug/AR_aug.npy') #(8,288,2000,2000)

f.write("size of AR_aug: " + str(AR_aug.shape) + '\n')
f.flush()


def calculate_patch_density(img, rows, cols, patch_size, overlap):

    # threshld segmentation
    if np.mean(img[0:rows, 0:cols]) < np.mean(img[0:cols, 0:rows]):
        rows, cols = cols, rows #for rotated image
    img_uint8 = img[0:rows, 0:cols].copy()
    img_uint8 = np.rint(255 * img_uint8).astype(np.uint8)
    img_thr = cv2.adaptiveThreshold(img_uint8, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 201, -4)  # thresholding

    # remove small connected areas less than min_size
    img_remove = morphology.remove_small_objects(img_thr.astype(np.bool), min_size=10, connectivity=2)
    img_remove = img_remove.astype(np.int) #binary image
   
    # calculate vessel density
    density_mean = np.mean(img_remove)
    f.write ("mean of AR image vessel density is: " + str(density_mean) + '\n')
    f.flush()

    # extract AR image patch and calculate its density
    patchH, patchW = (rows-overlap)//(patch_size-overlap), (cols-overlap)//(patch_size-overlap)
    patch_density = []
    
    for i in range(patchH):
        for j in range(patchW):
            density = np.mean(img_remove[i*(patch_size-overlap):(i+1)*patch_size-i*overlap,
                                       j*(patch_size-overlap):(j+1)*patch_size-j*overlap])
            patch_density.append(round(density*1000)/1000.0) 

    DataLen = sum((np.array(patch_density) >= 3/5*density_mean).astype(np.int))
    f.write("Number of patches: " + str(DataLen) + '\n')
    f.flush()

    return DataLen, patch_density


# extract patches of different sizes
patch_size = [192, 256, 384, 480]
overlap = [64, 64, 64, 100]
density_sum = []

for k in range(len(patch_size)):

    # Record patch vessel density
    patch_num, density_patch = 0, []
    f.write("patch size: (%d, %d)\n" % (patch_size[k],patch_size[k]))
    f.flush()
    for i in range(8):
        for j in range(288):
            f.write(str(i+1) + '|' + str(j+1) + '\n')
            f.flush()
            if i == 0: # 1st pair of images sized (2000,1427)
                mixed_data = calculate_patch_density(AR_aug[i,j,:,:], 2000, 1427, patch_size[k], overlap[k])  
            elif i == 2: # 3rd pair of images sized (1500,1500)
                mixed_data = calculate_patch_density(AR_aug[i,j,:,:], 1500, 1500, patch_size[k], overlap[k]) 
            elif i == 3: # 5th pair of images sized (1988,1980)
                mixed_data = calculate_patch_density(AR_aug[i,j,:,:], 1988, 1980, patch_size[k], overlap[k]+2)
            else:
                mixed_data = calculate_patch_density(AR_aug[i,j,:,:], 2000, 2000, patch_size[k], overlap[k])
        
            patch_num += mixed_data[0]
            density_patch += mixed_data[1]
    density_sum.append(density_patch)
    f.write("Useful patch NO.: %d\n" % patch_num)
    f.flush()

f.write("Recording done!")
f.flush()
f.close()

del AR_aug
gc.collect()

# Plot distribution of patch vessel density
density_prob = np.zeros((4,101))
for num in range(len(patch_size)):
    for elem in density_sum[num]:
        density_prob[num, int(np.rint(100*elem))] = density_prob[num, int(np.rint(100*elem))]+1
    density_prob[num,:] /= len(density_sum[num])
np.rint
density = np.arange(0,1.01,0.01)
plt.figure()
plt.plot(density, density_prob[0], density, density_prob[1],
            density, density_prob[2], density, density_prob[3])
plt.xlabel('patch vessel density')
plt.ylabel('Probability')
plt.title(" PDF of patch vessel density of different size")
plt.legend(labels = ['192x192', '256x256', '384x384', '480x480'], loc = 'upper right')
plt.show()

