import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
import scipy.ndimage as ndi
from skimage import color, measure, morphology
from PIL import Image

ARPAM_mat, ORPAM_mat = scipy.io.loadmat('../PAaror/AR_1.mat'), scipy.io.loadmat('../PAaror/OR_1.mat')
ARPAM_npy, ORPAM_npy = ARPAM_mat['C532'], ORPAM_mat['C532']


def pre_processing(img, TempH, TempW):
    
    # (0,1) normalizatin
    rows, cols = img.shape
    img_max, img_min = np.amax(img.reshape(-1,1)), np.amin(img.reshape(-1,1))
    #img_norm = np.rint(255*((img.reshape(-1,1) - img_min) / (img_max - img_min)))
    img_norm = (img.reshape(-1,1) - img_min) / (img_max - img_min)
    img_norm = img_norm.reshape(rows, cols).astype(np.float32)

    # recognize isolated pixel points
    img_denoise = img_norm.copy()
    mean, std = np.mean(img_norm.reshape(-1,1)), np.std(img_norm.reshape(-1,1))
    print("mean, std: ", mean, std)
    threshold1, threshold2 = mean+std,  mean+3*std 

    for i in range(rows):
        for j in range(cols):
            TempValue = []
            nCount = 0
            if (img_norm[i,j] < threshold2):
                continue
            for m in range(-int((TempH-1)/2),int((TempH+1)/2)): #孤立亮点尺寸大概为(5,5）
                for n in range(-int((TempW-1)/2),int((TempW+1)/2)):
                    if (i+m)<0 or (i+m)>=2000 or (j+n)<0 or (j+n)>=2000:
                        continue
                    TempValue.append(img_denoise[i+m,j+n]) #一般列表有25个元素
            lenTemp = len(TempValue)
            nCount1 = sum(k < threshold1 for k in TempValue) 
            nCount2 = sum(k > threshold2 for k in TempValue)
            TempValue.sort()
            if nCount1 > lenTemp/5 : #孤立亮点较边缘点而言，邻域内所含像素值陡低的点要多（一般占1/5以上）
                if lenTemp % 2 == 1: #中值滤波
                    img_denoise[i,j] = TempValue[int((lenTemp-1)/2)]
                else:
                    img_denoise[i,j] = (TempValue[int(lenTemp/2)]+TempValue[int(lenTemp/2-1)])/2
            if img_norm[i,j] > mean+8*std:
                img_denoise[i,j] = 0

    return img_norm, img_denoise


AR_norm, AR_denoise = pre_processing(ARPAM_npy,5,5)
OR_norm, OR_denoise = pre_processing(ORPAM_npy,5,5)


plt.figure 
plt.subplot(221), plt.imshow(AR_norm)
plt.title("Normalized image"), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.hist(AR_norm.flatten(), bins=256) #histogram
plt.title("Histogram before denoising")
plt.subplot(223), plt.imshow(AR_denoise)
plt.title("Denoising image ")
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(AR_denoise.flatten(), bins=256) 
plt.title("Histogram after denoising")
plt.tight_layout()
plt.show()

plt.figure 
plt.subplot(221), plt.imshow(OR_norm)
plt.title("Normalized image"), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.hist(OR_norm.flatten(), bins=256) #histogram
plt.title("Histogram before denoising")
plt.subplot(223), plt.imshow(OR_denoise)
plt.title("Denoising image")
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(OR_denoise.flatten(), bins=256) 
plt.title("Histogram after denoising")
plt.tight_layout()
plt.show()


def vessel_segment(img):     
    
    # threshld segmentation
    rows, cols = img.shape
    img = np.rint(255 * img)
    img = img.astype(np.uint8)
    img_thr = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 201, -4)  #Otsu thresholding
    print('mean of thresholded image is:',np.mean(img_thr.reshape(-1,1)))

    # 8-connected area labeling
    #img_labels = measure.label(img_thr, connectivity=2)
    #skimage.measure.regionprops(label_image)

    # remove small connected areas less than min_size
    img_remove = morphology.remove_small_objects(img_thr.astype(np.bool), min_size=10, connectivity=2)
    img_remove = img_remove.astype(np.int) #binary image
   
    #calculate  vessel density
    density_mean = np.mean(img_remove)
    density_std = np.std(img_remove, ddof=1)
    print('mean of vessel density:', density_mean)
    print('std of vessel density:', density_std)

    # show vessel segmentation result
    img_label = np.zeros((rows,cols,3), np.uint8)
    img_label[img_remove == 1, 0] = 255
    img_RGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    img_RGB[img_remove == 1, :] = 0
    img_seg = cv2.addWeighted(img_label, 1, img_RGB, 1, 0)

    plt.figure 
    plt.subplot(221), plt.imshow(img, cmap = plt.cm.gray)
    plt.title("Denoising image"), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(img_thr, cmap = plt.cm.gray)
    plt.title("Thresholded result ")
    plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_remove, "gray")
    plt.title("Remove small connected areas")
    plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img_seg, "gray")
    plt.title("Show vessel segmentation")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

    return img_remove

AR_seg = vessel_segment(AR_denoise)
OR_seg = vessel_segment(OR_denoise) #binary image

