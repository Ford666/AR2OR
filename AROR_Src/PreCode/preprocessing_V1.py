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
from PIL import Image


f = open('../preprocessing_record.txt','w+')
# Read the data
AR_npy = np.zeros(shape=(10,2000,2000), dtype=np.float32)
OR_npy = np.zeros(shape=(10,2000,2000), dtype=np.float32)
#ARPAM_mat, ORPAM_mat = scipy.io.loadmat('../PAaror/AR_1.mat'), scipy.io.loadmat('../PAaror/OR_1.mat')
#ARPAM_npy, ORPAM_npy = ARPAM_mat['C532'], ORPAM_mat['C532']

for i in np.arange(10):
    #dict_keys(['__header__', '__version__', '__globals__', 'C532'])
    ARPAM_mat = scipy.io.loadmat('../PAaror/AR_'+str(i+1)+'.mat')
    ORPAM_mat = scipy.io.loadmat('../PAaror/OR_'+str(i+1)+'.mat')
    if i < 6:
        AR_npy[i], OR_npy[i] = ARPAM_mat['C532'], ORPAM_mat['C532']
    elif i == 6: 
        AR_npy[i], OR_npy[i] = ARPAM_mat['C2'], ORPAM_mat['C1']
    else: 
        AR_npy[i], OR_npy[i] = ARPAM_mat['C1'], ORPAM_mat['C1']
  
f.write ("Read data done" + '\n')
f.flush()

# preprocessing 1: (0,1) standardization
AR_min = (np.amin(np.amin(AR_npy,axis=1),axis=1)).reshape(10,1,1)
AR_max = (np.amax(np.amax(AR_npy,axis=1),axis=1)).reshape(10,1,1) 
OR_min = (np.amin(np.amin(OR_npy,axis=1),axis=1)).reshape(10,1,1) 
OR_max = (np.amax(np.amax(OR_npy,axis=1),axis=1)).reshape(10,1,1)
AR_npy,OR_npy = (AR_npy - AR_min) / (AR_max - AR_min), (OR_npy - OR_min) / (OR_max - OR_min)

#AR_max, AR_min = np.amax(ARPAM_npy.reshape(-1,1)), np.amin(ARPAM_npy.reshape(-1,1))
#AR_norm = (ARPAM_npy.reshape(-1,1) - AR_min) / (AR_max - AR_min)
#AR_norm = AR_norm.reshape(2000, 2000).astype(np.float32)
#OR_max, OR_min = np.amax(ORPAM_npy.reshape(-1,1)), np.amin(ORPAM_npy.reshape(-1,1))
#OR_norm = (ORPAM_npy.reshape(-1,1) - OR_min) / (OR_max - OR_min)
#OR_norm = OR_norm.reshape(2000, 2000).astype(np.float32)

f.write ("Data standardization done" + '\n')
f.flush()

for i in range(10):
    export_path1 = '../PAaror/AR_norm/'+str(i+1)+'.png'
    mpimg.imsave(export_path1, AR_npy[i])

    export_path2 = '../PAaror/OR_norm/'+str(i+1)+'.png'
    mpimg.imsave(export_path2, OR_npy[i])

# preprocessing 2: recognize isolated pixel points, apply median filtering
def Denoising(img, TempH, TempW):

    rows, cols = img.shape
    img_denoise = img.copy()
    mean, std = np.mean(img.reshape(-1,1)), np.std(img.reshape(-1,1))
    print("mean, std: ", mean, std)
    threshold1, threshold2 = mean+std,  mean+3*std 

    for i in range(rows):
        for j in range(cols):
            TempValue = []
            nCount = 0
            if (img[i,j] < threshold2):
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
            if img[i,j] > mean+8*std:
                img_denoise[i,j] = 0
    return img_denoise

def Denoising1(img, TempH, TempW):

    rows, cols = img.shape
    img_denoise = img.copy()
    mean, std = np.mean(img.reshape(-1,1)), np.std(img.reshape(-1,1))
    print("mean, std: ", mean, std)
    threshold1, threshold2 = mean+std,  mean+3*std 

    for i in range(rows):
        for j in range(cols):
            TempValue = []
            nCount = 0
            if (img[i,j] < threshold2):
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
            if img[i,j] > mean+4*std and nCount2 < lenTemp*2/3: #孤立亮点不如过曝血管区域的整体像素值都高
                img_denoise[i,j] = 0
    return img_denoise

for i in np.arange(10):
    if i <= 6:
        AR_npy[i] = Denoising(AR_npy[i],5,5)
        OR_npy[i] = Denoising(OR_npy[i],5,5)
    else:
        AR_npy[i] = Denoising1(AR_npy[i],5,5)
        OR_npy[i] = Denoising1(OR_npy[i],5,5)


for i in np.arange(10):
    export_path1 = '../PAaror/AR_denoise/'+str(i+1)+'.png'
    mpimg.imsave(export_path1, AR_npy[i])

    export_path2 = '../PAaror/OR_denoise/'+str(i+1)+'.png'
    mpimg.imsave(export_path2, OR_npy[i])


#AR_denoise = Denoising(AR_norm,5,5)
#OR_denoise = Denoising(OR_norm,5,5)

f.write ("Recognize isolated pixels done" + '\n')
f.flush()

# Data argumentation functions
def Flip(img, options):
    """
    img: ndarry
    options: 0, 1, 2, 3 for different flipping
    p: probability, [0,1]
    """
    if options == 0:
        image = img  #no operation
    elif options == 1:
        image = cv2.flip(img, 0, dst=None) #vertical flipping
    elif options == 2:
        image = cv2.flip(img, 1, dst=None) #horizontal flipping(mirroring)
    elif options == 3:
        image = cv2.flip(img, -1, dst=None) #diagonal flipping
        
    return image


def Rotation(img, options):
    """
    img: ndarray
    angle: rotation angle: 0,90,180,270
    """
    rows, cols = img.shape
    if options == 0:
        angle = 0
    elif options == 1:
        angle = 90
    elif options == 2:
        angle = 270

    rot_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    image = cv2.warpAffine(img,rot_matrix,(rows, cols))
    return image
 


def Elastic_Deformation(img, sigma, seed):
    """
    img: ndarray
    alpha: scaling factor to control the deformation intensity
    sigma: elasticity coefficient
    seed: random integer from 1 to 100
    """  
    if seed > 40:
        alpha = 34
        random_state = np.random.RandomState(seed) 
        #image = np.pad(img, pad_size, mode="symmetric")
        center_square, square_size = np.float32(img.shape)//2, min(img.shape)//3

        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-img.shape[1]*0.08, img.shape[1]*0.08, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1,pts2)
        image = cv2.warpAffine(img, M, img.shape, borderMode=cv2.BORDER_REFLECT_101) #Random affine transform

        #Generate random displacement fields by gaussian conv
        dx = dy = gaussian_filter((random_state.rand(*image.shape)*2 - 1),
                             sigma, mode="constant", cval=0) * alpha 
        x, y =np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])) #Grid coordinate matrix
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        img_ela = map_coordinates(image, indices, order=1).reshape(*image.shape) #preform bilinear interpolation
    else:
        img_ela = img
    return img_ela

    #return Cropping(map_coordinates(image, indices, order=1).reshape(rows,cols), 
    #                 rows, pad_size, pad_size)


def Crop_Pad(img, options):
    """
    img: ndarray
    crop_size: size of cropped image
    options: 1, 2, 3, 4
    """
    crop_size0, crop_size1 = int(img.shape[0] * 0.7), int(img.shape[1] * 0.7)
    if options == 1: #left top
        cropped_img = img[0:crop_size0, 0:crop_size1]
    elif options == 2: #left bottom
        cropped_img = img[img.shape[0]-crop_size0:img.shape[0], 0:crop_size1]
    elif options == 3: #right top
        cropped_img = img[0:crop_size0, img.shape[1]-crop_size1:img.shape[1]]
    elif options == 4: #right bottom
        cropped_img = img[img.shape[0]-crop_size0:img.shape[0], img.shape[1]-crop_size1:img.shape[1]]

    pad_size = int(img.shape[0] * 0.15)
    return np.pad(cropped_img, pad_size, mode="symmetric")


def Add_Gaussian_Noise(img, options):
    """
        image : numpy array of image
        standard deviation : pixel standard deviation of gaussian noise
    """
    if options:
        std = 1e-3
        img_mean = np.mean(img.reshape(-1,1))
        gaus_noise = np.random.normal(img_mean, std, img.shape)
        noise_img = img + gaus_noise
        noise_img[noise_img<0] = 0
        noise_img[noise_img>1] = 1
    else:
        noise_img = img
    return noise_img


def Gamma_Transform(img, options):
    """
    img: ndarray
    gamma: coefficient of gamma transform
    """
    if options:
        gamma = 0.8
        img_gamma = np.power(img, gamma)
        img_gamma[img>1] = 1
    else:
        img_gamma = img       
    return img_gamma


## 4 flipping * 3 rotation * 3 elastic deformation * 2 gamma trasform * 2 add noise = 144 
#AR_aug = np.zeros(shape=(7,144,2000,2000), dtype=np.float32)
#OR_aug = np.zeros(shape=(7,144,2000,2000), dtype=np.float32)
#parms, parm = [], {}
#for i in range(4):
#    for j in range(3):
#        for k in range(3):
#            for m in range(2):
#                for n in range(2):
#                    parm = {'flip_option':i, 'rot_option':j, 
#                                'ela_parm':{'sigma':6, 'seed':10*(k-1)+50}, 'gamma_option':m, 'noise_option':n}
#                    parms.append(parm)

#for i in range(7):
#    for j in range(144):
#        AR_aug[i, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(AR_npy[i], 
#                      parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                       parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
#        OR_aug[i, j, :, :] = Add_Gaussian_Noise(Gamma_Transform(Elastic_Deformation(Rotation(Flip(OR_npy[i], 
#                      parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
#                       parms[j]['ela_parm']['seed']), parms[j]['gamma_option']), parms[j]['noise_option'])
    
#np.save('../Data_aug/AR_aug.npy', AR_aug) 
#np.save('../Data_aug/OR_aug.npy', OR_aug)

#f.write ("Data augmentation done: AR_aug(7,144,2000,2000), OR_aug(7,144,2000,2000)")
#f.flush()

#AR_aug =  np.load('../Data_aug/AR_aug.npy') #(7,144,2000,2000)
#OR_aug =  np.load('../Data_aug/OR_aug.npy') #(7,144,2000,2000)

#f.write ("Read augmented data done" + '\n')
#f.flush()

# Vessel segmentation and image patch screening 
def Screening_Patch(img, patch_size):     
    
    # threshld segmentation
    img = np.rint(255 * img).astype(np.uint8)
    img_thr = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 201, -4)  # thresholding

    # remove small connected areas less than min_size
    img_remove = morphology.remove_small_objects(img_thr.astype(np.bool), min_size=10, connectivity=2)
    img_remove = img_remove.astype(np.int) #binary image
   
    #calculate  vessel density
    density_mean = np.mean(img_remove)
    density_std = np.std(img_remove, ddof=1)
    print('mean of vessel density:', density_mean)
    print('std of vessel density:', density_std)

    #Screeing image patch based on vessel density
    ScreenPatch = []
    patch_num = img.shape[0]//patch_size[0] * img.shape[1]//patch_size[1]
    x = 0
    for i in range(img.shape[0]//patch_size[0]):
        for j in range(img.shape[1]//patch_size[1]):
            #export_path1 = '../Vessel_densiy/image_patch/'+str(x+1)+'.png'
            #mpimg.imsave(export_path1, img[i*patch_size[0]:(i+1)*patch_size[0],
            #                           j*patch_size[1]:(j+1)*patch_size[1]])

            density = np.mean(img_remove[i*patch_size[0]:(i+1)*patch_size[0],
                                       j*patch_size[1]:(j+1)*patch_size[1]])
            if density > density_mean/6: # threshold for screening
                #export_path2 = '../Vessel_densiy/screening_patch/'+str(x+1)+'.png'
                #mpimg.imsave(export_path2, img[i*patch_size[0]:(i+1)*patch_size[0],
                #                       j*patch_size[1]:(j+1)*patch_size[1]])
                ScreenPatch.append(img[i*patch_size[0]:(i+1)*patch_size[0],
                                       j*patch_size[1]:(j+1)*patch_size[1]])
            x += 1
    return ScreenPatch.shape[0], np.array(ScreenPatch)

#OR_patch = Screening_Patch(OR_denoise,(200,200))
#print(OR_patch.shape)


#Image patch extraction based on vessel density sceening result
def Extract_Patch(img, patch_size, rand_row):
    """
    img: ndarray
    patch_size: touple, size of image patch 
    """
    # threshld segmentation
    img_uint8 = img.copy()
    img_uint8 = np.rint(255 * img_uint8).astype(np.uint8)
    img_thr = cv2.adaptiveThreshold(img_uint8, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 201, -4)  # thresholding

    # remove small connected areas less than min_size
    img_remove = morphology.remove_small_objects(img_thr.astype(np.bool), min_size=10, connectivity=2)
    img_remove = img_remove.astype(np.int) #binary image
   
    #calculate  vessel density
    density_mean = np.mean(img_remove)
    density_std = np.std(img_remove, ddof=1)
    f.write ("mean of vessel density is: " + str(density_mean) + '\n')
    f.flush()

    #extract image patch given vessel density
    patch_num = img.shape[0]//patch_size[0] * img.shape[1]//patch_size[1]
    TrainSet, ValSet, TestSet = [], [], []
    for i in range(img.shape[0]//patch_size[0]):
        for j in range(img.shape[1]//patch_size[1]):
            density = np.mean(img_remove[i*patch_size[0]:(i+1)*patch_size[0],
                                       j*patch_size[1]:(j+1)*patch_size[1]])
            if density < density_mean/6: # threshold for screening
                continue
            else:
                if (i == rand_row ): 
                    if (j % 2) == 0:   # patches from one random row and its even-numbered columns => Val set
                        ValSet.append(img[i*patch_size[0]:(i+1)*patch_size[0],
                                            j*patch_size[1]:(j+1)*patch_size[1]])
                    else: # patches from one random row and its odd-numbered columns => Test set
                        TestSet.append(img[i*patch_size[0]:(i+1)*patch_size[0],
                                            j*patch_size[1]:(j+1)*patch_size[1]])
                else:
                    TrainSet.append(img[i*patch_size[0]:(i+1)*patch_size[0],
                                            j*patch_size[1]:(j+1)*patch_size[1]])

    TrainSet, ValSet, TestSet = np.array(TrainSet), np.array(ValSet), np.array(TestSet)
    set_len = [len(TrainSet),len(ValSet),len(TestSet)]
    len_str =','.join(str(e) for e in set_len)
    f.write("length of train/val/test: " + len_str + '\n')
    f.flush()

    return set_len, TrainSet, ValSet, TestSet



## Divide into training/val/test set 
#def Divide_Dataset(Aug_data):
#    train_set = np.zeros(shape=(7*144*90,200,200), dtype=np.float16)
#    train_label = np.zeros(shape=(7,144,1), dtype=np.uint8)
#    val_set = np.zeros(shape=(7*144*5,200,200), dtype=np.float16)
#    val_label = np.zeros(shape=(7,144,1,1), dtype=np.uint8)
#    test_set = np.zeros(shape=(7*144*5,200,200), dtype=np.float16)
#    test_label = np.zeros(shape=(7,144,1,1), dtype=np.uint8)

#    train_sum, val_sum, test_sum = 0, 0, 0
#    for i in range(7):
#        for j in range(144):
#            f.write(str(i+1) + '|' + str(j+1) + '\n')
#            f.flush()
#            rand_row = np.random.randint(0,10)
#            mixed_data = Extract_Patch(Aug_data[i,j,:,:],(200,200), rand_row)
#            train_len, val_len, test_len = mixed_data[0][0], mixed_data[0][1], mixed_data[0][2]
#            train_sum += train_len
#            val_sum += val_len
#            test_sum += test_len

#            train_set[train_sum-train_len:train_sum,:,:], val_set[val_sum-val_len:val_sum,:,:], \
#               test_set[test_sum-test_len:test_sum,:,:] = mixed_data[1], mixed_data[2], mixed_data[3]
#            val_label[i,j,0,:], test_label[i,j,0,:] = rand_row, rand_row
#            train_label[i,j,0], val_label[i,j,:,0], test_label[i,j,:,0] =  \
#               mixed_data[0][0], mixed_data[0][1], mixed_data[0][2]

#    set_sum = [train_sum,val_sum,test_sum]
#    sum_str =','.join(str(e) for e in set_sum)
#    f.write("sum of length of train/val/test: " + sum_str + '\n')
#    f.flush()
#    return set_sum,train_set,train_label,val_set,val_label,test_set,test_label

#AR_set_sum, OR_set_sum = [],[]
#AR_train = np.zeros(shape=(7*144*90,200,200), dtype=np.float32)
#AR_train_label = np.zeros(shape=(7,144,1), dtype=np.uint8)
#OR_train = np.zeros(shape=(7*144*90,200,200), dtype=np.float32)
#OR_train_label = np.zeros(shape=(7,144,1), dtype=np.uint8)

#AR_val = np.zeros(shape=(7*144*5,200,200), dtype=np.float32)
#AR_val_label = np.zeros(shape=(7,144,1,1), dtype=np.uint8)
#OR_val = np.zeros(shape=(7*144*5,200,200), dtype=np.float32)
#OR_val_label = np.zeros(shape=(7,144,1,1), dtype=np.uint8)

#AR_test = np.zeros(shape=(7*144*5,200,200), dtype=np.float32)
#AR_test_label = np.zeros(shape=(7,144,1,1), dtype=np.uint8)
#OR_test = np.zeros(shape=(7*144*5,200,200), dtype=np.float32)
#OR_test_label = np.zeros(shape=(7,144,1,1), dtype=np.uint8)

#f.write("AR: Divide into training/val/test set" + '\n')
#f.flush()
#AR_set_sum,AR_train,AR_train_label,AR_val,AR_val_label,AR_test,AR_test_label = Divide_Dataset(AR_aug)
#f.write("OR: Divide into training/val/test set" + '\n')
#f.flush()
#OR_set_sum,OR_train,OR_train_label,OR_val,OR_val_label,OR_test,OR_test_label = Divide_Dataset(OR_aug)


## Save tain/val/test set 
#np.save('../Data_aug/AR_train.npy', AR_train)
#np.save('../Data_aug/AR_train_label.npy', AR_train_label)
#np.save('../Data_aug/OR_train.npy', OR_train)
#np.save('../Data_aug/OR_train_label.npy', OR_train_label)

#np.save('../Data_aug/AR_val.npy', AR_val)
#np.save('../Data_aug/AR_val_label.npy', AR_val_label)
#np.save('../Data_aug/OR_val.npy', OR_val)
#np.save('../Data_aug/OR_val_label.npy', OR_val_label)

#np.save('../Data_aug/AR_test.npy', AR_test)
#np.save('../Data_aug/AR_test_label.npy', AR_test_label)
#np.save('../Data_aug/OR_test.npy', OR_test)
#np.save('../Data_aug/OR_test_label.npy', OR_test_label)

#f.write("All training/val/test set saved!")
#f.flush()
#f.close()





