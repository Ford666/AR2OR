#!/usr/bin/python

from __future__ import print_function
import numpy as np
# import cupy as cp
from unet1 import unet, sampling
from deeplab import DeepLabV3Plus, classifier
from deeplabv3dense import DeepLabV3PlusDense
# from model import DMnet
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as D
import time
from torch.autograd import Variable
import h5py
from apex import amp
import Matfolder as mf

def dataloading(tuning=False, root=r'H:\LHH\PAimaging\AROR\datasplit\training', phase=None, BATCH_S=1, shuffle=True):
	print('====> loading ' + phase + ' datasets')
	dataset = mf.MatFolder(root=root, batch_size=BATCH_S)
	dataset_num = np.int(len(dataset))

	loader = D.DataLoader(
		dataset=dataset, pin_memory=True,
		batch_size=BATCH_S,
		shuffle=shuffle,
		num_workers=4,
	)

	print('number of' + phase + ' samples:', dataset_num)

	return loader, dataset_num
	
	
'''
root: data path
'''
train_loader, num_train = dataloading(
	root=r'H:\LHH\PAimaging\AROR\datasplit\training', phase='train', BATCH_S=BATCH_SIZE)
test_loader, num_val = dataloading(
	root=r'H:\LHH\PAimaging\AROR\datasplit\test', phase='test', BATCH_S=BATCH_SIZE)