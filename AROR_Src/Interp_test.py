# -*- coding: utf8 -*-
# Perceptual-driven SRGAN outputs produces SR images with sharp edges 
# and richer textures but with some unpleasant artifacts.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import numpy as np
import pandas as pd
from tqdm import tqdm

import pytorch_ssim
from utils import *
from SRGAN_V2 import *

 
if __name__ == "__main__":
    # SetUp
    #dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor
    BATCH_SIZE = 2

    #Load data
    test_set = DatasetFromFolder('../datasplit/test/1/x', '../datasplit/test/1/y')

    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, 
                                    shuffle=False, num_workers=BATCH_SIZE)

    AR = np.load("../datasplit/test/1/AR.npy")
    OR = np.load("../datasplit/test/1/OR.npy")
    AR, OR = AR[0:6*384-5*64, 0:6*384-5*64], OR[0:6*384-5*64, 0:6*384-5*64]

    # Import SRGAN model
    InterpNet = Generator().cuda() 
    alpha = [0.02,0.98,1]
    # k1, k2 = 4, 9
    for k in range(9,10): 
        print("Use InterpNet model %d to test!" % k)
        for t in range(len(alpha)):
            # weighted network parameters    
            # InterpNet_dir = "../datasplit/new_train/InterpNet_model%d%d.pkl" % (k1, k2)
            G_dir = "../datasplit/new_train/G_model%d.pkl" % k
            G_par = torch.load(G_dir) 
            SRCNN_par = torch.load("../datasplit/SRCNN_train/model8.pkl")
            # G_par = torch.load("../datasplit/new_train/G_model%d.pkl" % k1) #OrderedDict
            # SRCNN_par = torch.load("../datasplit/new_train/G_model%d.pkl" % k2)
            par_name = [name for name in G_par]  #OrderedDict
            for i in range(len(G_par)):
                G_par[par_name[i]] = alpha[t]*G_par[par_name[i]] + (1-alpha[t])*SRCNN_par[par_name[i]]
            # torch.save(G_par, InterpNet_dir)
            InterpNet.load_state_dict(G_par)
 

            with torch.no_grad():
                SR_imgs = torch.Tensor([]).cuda()
                InterpNet.eval()
                for x, y in test_loader:

                    AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(2,1,384,384)
                    SR_img = InterpNet(AR_img)

                    SR_img = SR_img.squeeze(1)
                    SR_imgs = torch.cat((SR_imgs, SR_img),0)

                SR_path = "../datasplit/test/1/InterpNet/GAN%d/alpha%.2f" % (k, alpha[t])
                SR = stitch_patch(SR_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64)

            mean, var = SR.mean(), SR.var()
            psnr, ssim, pcc = calculate_psnr(SR, OR), calculate_ssim(SR, OR), calculate_pcc(SR, OR)
            print("SR%d| Mean: %.4f, Var: %.4f, PSNR: %.4f, SSIM: %.4f, PCC: %.4f\n" % (k, mean, var, psnr, ssim, pcc))

    print("Finish test!")


            
                
    

    

            



