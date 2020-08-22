# coding: utf-8
import sys
sys.path.append("..")
from utils import *
import re

#Read .mat data
path = "../PAaror"
MatDirs = []  #"../PAaror/OR_10.mat"
for x in os.listdir(path):  #遍历指定路径下的文件
    if os.path.splitext(x)[1] == '.mat':  #筛选.mat文件
        MatDirs.append(os.path.join(path,x))

#Preprocessing
MatNum = len(MatDirs)
ARNpy = np.zeros(shape=(int(MatNum/2),2000,2000), dtype=np.float32)
ORNpy = np.zeros(shape=(int(MatNum/2),2000,2000), dtype=np.float32)
ARIdx, ORIdx = 0, 0
for x in MatDirs:
    # mkdir(os.path.splitext(x)[0])  
    if re.findall('AR', x):
        Mat, Str = scio.loadmat(x)['AR'], 'AR'
    elif re.findall('OR', x):
        Mat, Str = scio.loadmat(x)['OR'], 'OR'

    #(0,1) standardization
    Mat = Mat.astype(np.float32)
    Matmin = np.amin(np.amin(Mat,axis=0),axis=0)
    Matmax = np.amax(np.amax(Mat,axis=0),axis=0)
    NormMat = (Mat-Matmin) / (Matmax-Matmin)  
    
    #denoising
    #Not universal but individual analysis
    if re.findall('OR_02', x) or re.findall('OR_05', x) or re.findall('OR_ear_4', x):
        DenMat = DenoiseHist(NormMat)
    elif re.findall('Ali', x):
        DenMat = NormMat
    else:
        DenMat = Denoise(NormMat, TempSize=5)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(NormMat, cmap=cm.hot)
    # plt.subplot(1,2,2)
    # plt.imshow(DenMat, cmap=cm.hot)
    # plt.show()
    
    #Save all mat files 
    Matdir,Imgdir = list(os.path.splitext(x)[0]),list(os.path.splitext(x)[0])
    ProImgdir, OriImgdir = Imgdir[:], Imgdir[:]
    Matdir.insert(len(path),'/ProMat')
    ProImgdir.insert(len(path),'/ProImg')
    OriImgdir.insert(len(path),'/OriImg')
    Matdir, ProImgdir, OriImgdir = ''.join(Matdir), ''.join(ProImgdir), ''.join(OriImgdir)
    if Str == 'AR':
        ARNpy[ARIdx,0:DenMat.shape[0],0:DenMat.shape[1]] = DenMat
        ARIdx = ARIdx+1
    elif Str =='OR':
        ORNpy[ORIdx,0:DenMat.shape[0],0:DenMat.shape[1]] = DenMat
        ORIdx = ORIdx+1
    scio.savemat(Matdir+'.mat', {'%s' % Str: DenMat})
    plt.imsave(ProImgdir + '.png', DenMat, cmap=cm.hot)
    # plt.imsave(OriImgdir + '.png', NormMat, cmap=cm.hot)  

np.save("../PAaror/ProNpy/ARNpy.npy", ARNpy)
np.save("../PAaror/ProNpy/ORNpy.npy", ORNpy)
