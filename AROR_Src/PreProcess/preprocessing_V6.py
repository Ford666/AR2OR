# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.image as mpimg
import scipy.io
import cv2 
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndi
from skimage import color, measure, morphology
import json
from math import fabs, sin, cos, radians
from matplotlib.pyplot import cm
from utils import *


AR_npy = np.load('../PAaror/AR_npy.npy') 
OR_npy = np.load('../PAaror/OR_npy.npy')
print("Load data done: AR_npy(10,2000,2000), OR_npy(10,2000,2000)\n")

for i in range(AR_npy.shape[0]):
    AR_max, OR_max = (np.squeeze(AR_npy[i])).max(), (np.squeeze(OR_npy[i])).max()
    print("%d pair, AR_max: %.4f, OR_max: %.4f" % (i+1, AR_max, OR_max))

# ## For the 5th pair to be registered
# AR_4 = AR_npy[4,12:2000,0:1980].copy()
# OR_4 = OR_npy[4,0:1988,20:2000].copy()

# AR_npy[4], OR_npy[4] = 0, 0
# AR_npy[4,0:AR_4.shape[0],0:AR_4.shape[1]], OR_npy[4,0:OR_4.shape[0],0:OR_4.shape[1]] = AR_4, OR_4

# AR, OR = np.squeeze(AR_npy[4]), np.squeeze(OR_npy[4])

# # plot error map
# for i in range(AR_npy.shape[0]):
#     AR_dir = "../Data_aug/AR_train/%d.png" % (i+1)
#     OR_dir = "../Data_aug/OR_train/%d.png" % (i+1)
#     error_map_dir = "../Data_aug/error_map/%d.png" % (i+1)  
#     error_map = np.squeeze(AR_npy[i]) - np.squeeze(OR_npy[i])
#     error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
#     print("error map%d| mean=%.4f, var=%.4f" % (i+1, error_map.mean(),error_map.var()) )
#     plt.imsave(AR_dir, np.squeeze(AR_npy[i]))
#     plt.imsave(OR_dir, np.squeeze(OR_npy[i]))
#     plt.imsave(error_map_dir, error_map, cmap=cm.gray)

# # plot the fusion of AROR image
# for i in range(AR_npy.shape[0]):
#     AR, OR = np.rint(np.squeeze(AR_npy[i]) * 255), np.rint(np.squeeze(OR_npy[i]) * 255)
#     AR, OR = AR.astype(np.uint8), OR.astype(np.uint8)

#     AR_RGB = cv2.cvtColor(AR, cv2.COLOR_GRAY2RGB)
#     AR_RGB[OR>10, :] = 255
#     AROR_map_dir = "../Data_aug/AROR_map/%d.png" % (i+1) 
#     plt.imsave(AROR_map_dir, AR_RGB)

for i in range(AR_npy.shape[0]):
    AR, OR = np.rint(np.squeeze(AR_npy[i]) * 255), np.rint(np.squeeze(OR_npy[i]) * 255)
    AR, OR = AR.astype(np.uint8), OR.astype(np.uint8)

    AR_RGB, OR_RGB = np.zeros([2000,2000,3], np.uint8), np.zeros([2000,2000,3], np.uint8)
    OR_RGB[OR>3, 0] = 255
    AR_RGB[AR>10, 2] = 255
    AROR_map = AR_RGB +  OR_RGB
    AROR_map_dir = "../Data_aug/AROR_map/%d.png" % (i+1) 
    plt.imsave(AROR_map_dir, AROR_map)


# for k in range(4):

#     ela_AR, ela_OR = Elastic_Deformation(AR, 6, 20*(k-1)+50), Elastic_Deformation(OR, 6, 20*(k-1)+50)

#     plt.imsave("../ela_AR%d.png" % k, ela_AR)
#     plt.imsave("../ela_OR%d.png" % k, ela_OR)

# gamma_value  = [0,1] # Compare brightness shift of image for different gamma values 
# plt.figure
# plt.tight_layout()
# for i in range(2):
#    plt.subplot(2,2,i+1)
#    plt.imshow(Add_Gaussian_Noise(AR, gamma_value[i]))
#    plt.title("ARPAM, gamma="+str(gamma_value[i]))

# for i in range(2):
#    plt.subplot(2,2,i+3)
#    plt.imshow(Add_Gaussian_Noise(OR, gamma_value[i]))
#    plt.title("ORPAM, gamma="+str(gamma_value[i]))
# plt.show()

# 4 flipping * 3 rotation * 3 elastic deformation * 2 gamma trasform * 2 add noise = 288
# AR_aug = np.zeros(shape=(8,288,2000,2000), dtype=np.float32)
# OR_aug = np.zeros(shape=(8,288,2000,2000), dtype=np.float32)
# parms, parm = [], {}
# for i in range(4):
#     for j in range(3):
#         for k in range(4):
#             for m in range(3):
#                 for n in range(2):
#                     parm = {'flip_option':i, 'rot_option':j, 
#                                 'ela_parm':{'sigma':6, 'seed':20*(k-1)+50}, 'gamma_option':m, 'noise_option':n}
#                     parms.append(parm)

# # save parms list as json file
# parms_json = open('../parms.json', 'w+')
# for i in parms:
#     parms_json.write(json.dumps(i)+'\n')
# parms_json.close()

# k = 0
# for i in range(10):
#     if k < 8:
#         if i==3 or i ==5: # Abandon 4h and 6th pair
#                 continue
#         elif i == 0:  # (2000,1427)
#             for j in range(288):                             
#                 AR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(AR_npy[0,0:2000,0:1427], 
#                         parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                         parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#                 AR_aug[k,j,0:AR.shape[0], 0:AR.shape[1]] = AR
#                 OR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(OR_npy[0,0:2000,0:1427], 
#                         parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                         parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#                 OR_aug[k,j,0:OR.shape[0], 0:OR.shape[1]] = OR
#             k += 1
#         elif i == 2:  #(1500,1500)
#             for j in range(288): 
#                 AR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(AR_npy[2,0:1500,0:1500], 
#                         parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                         parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#                 AR_aug[k,j,0:AR.shape[0], 0:AR.shape[1]] = AR
#                 OR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(OR_npy[2,0:1500,0:1500], 
#                         parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                         parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#                 OR_aug[k,j,0:OR.shape[0], 0:OR.shape[1]] = OR
#             k += 1
#         elif i == 4:  #(1988,1980)
#             for j in range(288): 
#                 AR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(AR_npy[4,0:1988,0:1980], 
#                         parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                         parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#                 AR_aug[k,j,0:AR.shape[0], 0:AR.shape[1]] = AR
#                 OR = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(OR_npy[4,0:1988,0:1980], 
#                         parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                         parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#                 OR_aug[k,j,0:OR.shape[0], 0:OR.shape[1]] = OR
#             k += 1
#         else:
#             for j in range(288): 
#                 AR_aug[k, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(AR_npy[i], 
#                             parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                             parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#                 OR_aug[k, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(OR_npy[i], 
#                         parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                         parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#             k += 1
#         print("Finish %d AROR pairs!" % k)
                 
# np.save('../Data_aug/AR_aug.npy', AR_aug) 
# np.save('../Data_aug/OR_aug.npy', OR_aug)

# ## To see the results of data augumentation of 3rd pair AR/OR images.
# #for i in range(288):
# #    path1 = '../PAaror/AR3_144/'+str(i+1)+'.png'
# #    path2 = '../PAaror/OR3_144/'+str(i+1)+'.png'
# #    mpimg.imsave(path1, AR_aug[2,i,:,:])
# #    mpimg.imsave(path2, OR_aug[2,i,:,:])

print("Data augmentation done: AR_aug(8,288,2000,2000), OR_aug(8,288,2000,2000)")






