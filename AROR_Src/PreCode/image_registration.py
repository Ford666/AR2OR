# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import cv2
from skimage import measure

# image registration for AR/OR 3rd iamge
AR_npy = np.load('../Data_aug/AR_npy.npy') 
OR_npy = np.load('../Data_aug/OR_npy.npy')

# iamge match
AR3_match = AR_npy[2,177:2000,100:2000]
OR3_match = OR_npy[2,0:1823,0:1900]

#for i in range(10):
#    if i == 2:
#        AO_ssim = measure.compare_ssim(AR3_match, OR3_match)
#    else:
#        AO_ssim = measure.compare_ssim(AR_npy[i], OR_npy[i])
#    print('%d AO_ssim: %f' % (i, AO_ssim))

AO3_ssim = measure.compare_ssim(AR3_match, OR3_match)
AO7_ssim = measure.compare_ssim(AR_npy[6], OR_npy[6])
print('AO3_ssim: ', AO3_ssim)
print('AO7_ssim: ', AO7_ssim)


# Extracting patches with high SSIM
def Extract_Patch(img1, img2, patchH, patchW, overlap, ssim):
    """
    img: ndarray
    patchH, patchW: int,size of image patch 
    overlap: int, the overlap size for extracting patches 
    """
    rows, cols = img1.shape
    AR_copy, OR_copy = img1.copy(), img2.copy()
    x = 0
    for i in range((rows-overlap)//(patchH-overlap)):
        for j in range((cols-overlap)//(patchW-overlap)):
            x += 1
            AR_patch = img1[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                       j*(patchW-overlap):(j+1)*patchW-j*overlap]
            OR_patch = img2[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                       j*(patchW-overlap):(j+1)*patchW-j*overlap]
            AO_ssim = measure.compare_ssim(AR_patch, OR_patch)
            if AO_ssim < ssim * 17/24:
                AR_copy[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                       j*(patchW-overlap):(j+1)*patchW-j*overlap] = 0
                OR_copy[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                       j*(patchW-overlap):(j+1)*patchW-j*overlap] = 0

                export_path1 = '../registration/AR3_diss/'+str(x)+'.png'
                export_path2 = '../registration/OR3_diss/'+str(x)+'.png'
                mpimg.imsave(export_path1, AR_patch)
                mpimg.imsave(export_path2, OR_patch)
            
    return AR_copy, OR_copy

AR3_ext, OR3_ext = Extract_Patch(AR3_match, OR3_match, 256, 256, 64, AO3_ssim) #(1823,1900)
np.save('../PAaror/AR3_ext.npy', AR3_ext) 
np.save('../PAaror/OR3_ext.npy', OR3_ext)


#'''
#1. Image match based on image grayscale (Normalized Cross Correlation method, NCC) 
#ther are some distortions between AR and OR images, even the common blocks are not exactly same.
#'''

#AR_npy = np.load('../Data_aug/AR_npy.npy') 
#OR_npy = np.load('../Data_aug/OR_npy.npy')

#AR_src = AR_npy[2,:,:]
#OR_temp = OR_npy[2,0:1823,0:1900] # template
#tempH, tempW = OR_temp.shape
#srcH, srcW = AR_src.shape

## calculate mean and stdd of temp: muT and SigmaT
#muT = np.mean(OR_temp)
#dSigmaT = tempH * tempW * np.var(OR_temp)
#temp_norm = OR_temp - muT

## Find optimal coordinate of the top-left point of submap
#Rmax = 0
#OptX, OptY = 0, 0
#for x in np.arange(srcH-tempH):
#    for y in np.arange(srcW-tempW):
#        dSigmaST, dSigmaS =0, 0      
#        muS = np.mean(AR_src[x:x+tempH, y:y+tempW])
#        src_norm = AR_src[x:x+tempH, y:y+tempW] - muS
#        dSigmaST = np.sum(temp_norm * src_norm)  # claculate dSigmaST
#        dSigmaS = tempH * tempW * np.var(AR_src[x:x+tempH, y:y+tempW])
#        R = dSigmaST / np.sqrt(dSigmaS * dSigmaT)
#        if R > Rmax:
#            OptX, OptY = x, y

#print('optimal coordinate of the top-left point of AR:', OptX, OptY)
#AR_sub = AR_src[OptX:OptX+tempH, OptY:OptY+tempW]

#plt.figure
#plt.tight_layout()
#plt.subplot(1,2,1)
#plt.imshow(AR_sub)
#plt.axis('off')
#plt.title('ARPAM submap')
#plt.subplot(1,2,2)
#plt.imshow(OR_temp)
#plt.axis('off')
#plt.title('ORPAM template')
#plt.show()

#"""
#2. Image alignment based on feature match, use SIFT feature detector
#Lowe, David G. "Object recognition from local scale-invariant features." iccv. Vol. 99. No. 2. 1999. 

#Four STeps:
#(1)Detect Features (2)Match Features  (3)Calculate Homography  (4)Warping image
#"""

#def sift_kp(image):
#    image = np.rint(255*image).astype(np.uint8)
#    if image.ndim == 3:
#        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    else:
#        gray_image = image
#    sift = cv2.xfeatures2d_SIFT.create() #feature detector
#    kp,des = sift.detectAndCompute(image,None)
#    kp_image = cv2.drawKeypoints(gray_image,kp,None)
#    return kp_image,kp,des
 
#def get_good_match(des1,des2):
#    bf = cv2.BFMatcher()
#    matches = bf.knnMatch(des1, des2, k=2)
#    good = []
#    for m, n in matches:
#        if m.distance < 0.75 * n.distance:
#            good.append(m)
#    return matches, good
 
#def siftImageAlignment(img1,img2):
#   _,kp1,des1 = sift_kp(img1)
#   _,kp2,des2 = sift_kp(img2)
   
#   _, goodMatch = get_good_match(des1,des2)
#   #match_img = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
#   #cv2.imshow('image match', match_img)
#   if len(goodMatch) > 4:
#       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
#       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
#       ransacReprojThreshold = 4
#       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
#       #其中H为求得的单应性矩阵
#       #status则返回一个列表来表征匹配成功的特征点。
#       #ptsA,ptsB为关键点
#       #cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关
#       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#   return imgOut,H,status
 

#img2 = AR_npy[2]
#img1 = OR_npy[2]
##while img1.shape[0] >  1000 or img1.shape[1] >1000:
##    img1 = cv2.resize(img1,None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
##while img2.shape[0] >  1000 or img2.shape[1] >1000:
##    img2 = cv2.resize(img2,None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
    
    
#result,_,_ = siftImageAlignment(img1,img2)
#export_name = '../registration/aligned_AR3.png'
#mpimg.imsave(export_name, result)

#cv2.namedWindow('1',cv2.WINDOW_NORMAL)
#cv2.namedWindow('2',cv2.WINDOW_NORMAL)
#cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
#cv2.imshow('1',img1)
#cv2.imshow('2',img2)
#cv2.imshow('Result',result)

##allImg = np.concatenate((img1,img2,result),axis=1)
##cv2.imshow('together',allImg)
#if cv2.waitKey(2000) & 0xff == ord('q'):
#    cv2.destroyAllWindows()
#    cv2.waitKey(1) 
