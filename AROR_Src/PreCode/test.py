import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os 
from os import listdir
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from SRGAN import Generator, Discriminator, generator_loss, MSE_loss, SSIM_loss
from train import DatasetFromFolder, show_images

#AR_aug =  np.load('../Data_aug/AR_aug.npy') #(8,144,2000,2000)
#OR_aug =  np.load('../Data_aug/OR_aug.npy') #(8,144,2000,2000)

# Image patch extraction 
def Extract_Patch(img1, img2, rows, cols, patchH, patchW, overlap, path):

    plt.imsave(path+'AR.png', img1)
    plt.imsave(path+'OR.png', img2)

    count = 0
    for i in range((rows-overlap)//(patchH-overlap)):
        for j in range((cols-overlap)//(patchW-overlap)):
            count += 1
            np.save(path+'AR/'+str(count).zfill(3)+'.npy', img1[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                            j*(patchW-overlap):(j+1)*patchW-j*overlap])
            np.save(path+'OR/'+str(count).zfill(3)+'.npy', img2[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                j*(patchW-overlap):(j+1)*patchW-j*overlap])

    return 

#Image patch stitch
def Stitch_patch(path1, path2, rows, cols, patchH, patchW, overlap):
      
    imgs = np.zeros([rows, cols])
    patchs = np.load(path1)

    numW, numH = (rows-overlap)//(patchH-overlap), (cols-overlap)//(patchW-overlap)
    count = 0
    for i in range(numW):
        for j in range(numH):
            imgs[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                      j*(patchW-overlap):(j+1)*patchW-j*overlap] = patchs[count]
            count += 1
    imgs = imgs[0:(i+1)*patchH-i*overlap, 0:(j+1)*patchW-j*overlap]
    np.save(path2+'SR.npy', imgs)
    plt.imsave(path2+'SR.png', imgs)

##选取弹性变换后的大图裁出图像块，构成测试集
#Extract_Patch(AR_aug[4][0], OR_aug[4][0], 2000, 2000, 256, 256, 64, '../datasplit/test/1/')
#Extract_Patch(AR_aug[4][4], OR_aug[4][4], 2000, 2000, 256, 256, 64, '../datasplit/test/2/')
#Extract_Patch(AR_aug[4][8], OR_aug[4][8], 2000, 2000, 256, 256, 64, '../datasplit/test/3/')
#Extract_Patch(AR_aug[5][0], OR_aug[5][0], 2000, 2000, 256, 256, 64, '../datasplit/test/4/')


if __name__ == "__main__":
    #SetUp
    f = open('../AROR_test.txt','w+')

    #dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
    BATCH_SIZE = 4

    #Load data
    test_set = DatasetFromFolder('../datasplit/test/1/AR', '../datasplit/test/1/OR')

    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, 
                                    shuffle=False, num_workers=BATCH_SIZE)

    #Load model parameter
    G = Generator().cuda()
    D = Discriminator().cuda()

    #Test the SRGAN
    G.load_state_dict(torch.load('../G_model.pkl'))
    D.load_state_dict(torch.load('../D_model.pkl'))

    f.write('Use trained GAN model to test on AR images!' + '\n')
    f.flush()
    print("Use trained GAN model to test on AR images!")

    iter_count = 0
    iter_per_test = int(100/BATCH_SIZE)

    with torch.no_grad():
        test_results = {'batch_sizes': 0, 'g_loss': 0, 'mse': 0, 'ssim': 0}
        SR_imgs = torch.Tensor([]).cuda()
        G.eval()
        D.eval()
        for x, y in test_loader:
            iter_count += 1
            test_results['batch_sizes'] += BATCH_SIZE

            AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(2,1,256,256)
            SR_img = G(AR_img)
            #logits_SR, logits_OR = D(SR_img), D(OR_img)
            #G_loss = generator_loss(logits_OR,logits_SR,OR_img,SR_img)
            logits_SR = D(SR_img)
            G_loss = generator_loss(logits_SR, OR_img, SR_img)

            if iter_count % 1 == 0:
                img_path = '../datasplit/test/1/'
                show_images(AR_img, img_path+'AR_iter/'+str(iter_count)+'.png')
                show_images(SR_img, img_path+'SR_iter/'+str(iter_count)+'.png')
                show_images(OR_img, img_path+'OR_iter/'+str(iter_count)+'.png')

            test_results['g_loss'] += BATCH_SIZE * G_loss.item()
            test_results['mse'] += BATCH_SIZE * MSE_loss(OR_img, SR_img).item()
            test_results['ssim'] += BATCH_SIZE * SSIM_loss(OR_img, SR_img).item()
            f.write('Iter/iter_per_test: '+str(iter_count)+ '/'+ str(iter_per_test) + '\t' + 'Loss_G: ' 
                            + str(test_results['g_loss']/test_results['batch_sizes']) + '\t' + 'MSE: '+
                             str( test_results['mse']/test_results['batch_sizes']) + '\t' + 'SSIM: ' +
                             str(test_results['ssim']/test_results['batch_sizes']) + '\n')
            f.flush()

            SR_img = SR_img.squeeze(1)
            SR_imgs = torch.cat((SR_imgs, SR_img),0)
        np.save('../datasplit/test/1/SR_patch.npy', SR_imgs.data.cpu().numpy())
        f.write("Finish test!\n")
        f.flush()
        print("Finish test!\n")


    f.close()
    Stitch_patch('../datasplit/test/1/SR_patch.npy', '../datasplit/test/1/', 2000, 2000, 256, 256, 64)