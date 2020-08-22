import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import scipy.io as scio
from os import listdir
from os.path import join
from utils import *
from matplotlib.pyplot import cm


# #Split dataset(N, H, W) into image slides 
# def split_data(stack_data, path):
#     N = stack_data.shape[0]
#     for i in range(N):
#         np.save(path+str(i+1).zfill(5)+'.npy', stack_data[i])

#AR_train, OR_train = np.load('../Data_aug/AR_train.npy'), np.load('../Data_aug/OR_train.npy')
#print(AR_train.shape, OR_train.shape)

#split_data(AR_train, '../datasplit/training/x/')
#split_data(OR_train, '../datasplit/training/y/')

# AR = np.load("../datasplit/test/1/AR.npy")
# OR = np.load("../datasplit/test/1/OR.npy")
# AR_dir, OR_dir = '../datasplit/test/1/AR.png', '../datasplit/test/1/OR.png'
# plt.imsave(AR_dir, AR, cmap=cm.hot)
# plt.imsave(OR_dir, OR, cmap=cm.hot)

# scio.savemat(AR_dir, {'AR':AR)
# scio.savemat(OR_dir, {'OR':OR)

# def Gamma_Transform1(img, gamma):
#    """
#    img: ndarray
#    gamma: coefficient of gamma transform
#    """
#    img_gamma = np.power(img, gamma)

#    return img_gamma

# gamma_value = [0.8, 0.9, 1] # Compare brightness shift of image for different gamma values 
# plt.figure
# plt.tight_layout()
# for i in range(3):
#    plt.subplot(1,3,i+1)
#    plt.imshow(Gamma_Transform1(OR, gamma_value[i]), cmap=cm.hot)
#    plt.title("OR_gamma, gamma="+str(gamma_value[i]))   
# plt.show()