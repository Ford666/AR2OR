import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from SRGAN_V6 import *
from utils import *
# import h5py
import scipy.io


#Read tumor data
# DataName = ['AR1','AR2','AR3','AR4','day2.1','day2.2'] #skin tumor
DataName = ['AR_brain_1','AR_brain_2','AR_brain_3'] #Mouse Brain
# DataName = ['AR_1','AR_2','AR_3','AR_4'] #brain
# DataName = ['depth-0','depth-300','depth-700','depth-1000','depth-1300','depth-1700'] #depth
Num = len(DataName)
# ARnpy = np.zeros(shape=(Num,2000,2000), dtype=np.float64)
# for i in range(Num):
#     # TumorMat = h5py.File('../PAaror/MouseTumor/brain/%s' % DataName[i] + '.mat', 'r')
#     # ARdata = TumorMat.get('C2')[()]
#     # ARnpy[i] = np.transpose(ARdata)
#     TumorMat = scipy.io.loadmat('../PAaror/MouseTumor/brain/Mouse Brain/%s' % DataName[i] + '.mat')
#     ARnpy[i] = TumorMat['C2']

# #0~1 Normalization
# ARmin = (np.amin(np.amin(ARnpy,axis=1),axis=1)).reshape(Num,1,1)
# ARmax = (np.amax(np.amax(ARnpy,axis=1),axis=1)).reshape(Num,1,1)
# ARnpy = (ARnpy-ARmin)/(ARmax-ARmin)

# # # Zscore
# # ARmean = (np.mean(ARnpy, axis=1)).reshape(Num,1,-1) #(Num,1,2000)
# # ARstd = (np.std(ARnpy, axis=1, ddof=1)).reshape(Num,1,-1) #(Num,1,2000)
# # ARnpy = np.divide((ARnpy-ARmean), ARstd)

# #Extract image patches
# for i in range(Num):
#     extract_TestPatch(ARnpy[i],2000,2000,384,384,64,'../datasplit/test/Tumor_ARdata/brain/Mouse Brain/%s/' % DataName[i])

if __name__ == "__main__":

    #dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor
    BATCH_SIZE = 2

    #Load and initialize GAN 
    G = Generator().cuda()   
    G.load_state_dict(torch.load("../datasplit/new_train/train6/G_model12.pkl" ))

    for j in range(Num):

        #Load data
        test_set = TestDataFromFolder('../datasplit/test/Tumor_ARdata/brain/Mouse Brain/%s/x' % DataName[j])

        test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, 
                                        shuffle=False, num_workers=BATCH_SIZE)
        print("Test on %s" % DataName[j])

        with torch.no_grad():

            SR_imgs = torch.Tensor([]).cuda()
            G.eval()
            for x in test_loader:

                AR_img = (x.unsqueeze(1)).type(dtype) #(BATCH_SIZE,1,384,384)
                SR_img = G(AR_img)

                SR_img = SR_img.squeeze(1)
                SR_imgs = torch.cat((SR_imgs, SR_img),0)

            SR_path = '../datasplit/test/Tumor_ARdata/brain/Mouse Brain/%s/SR' % DataName[j]
            patch_path = '../datasplit/test/Tumor_ARdata/brain/Mouse Brain/%s/SR patch/' % DataName[j]
            SR = stitch_SavePatch(SR_path, patch_path, SR_imgs.data.cpu().numpy(), 2000, 2000, 384, 384, 64)
 

    print("Finish test!\n")


# #Display with colormap='hot'
# for j in range(4):
#     SR_path = '../datasplit/test/Tumor_ARdata/brain/Mouse Brain/%s/SR.jpg' % DataName[j]
#     SR = plt.imread(SR_path)
#     ImgMean, ImgStd = np.mean(SR), np.std(SR, ddof=1)
#     plt.imsave(SR_path, SR, cmap=cm.hot) # , vmin=0, vmax=ImgMean-ImgStd

