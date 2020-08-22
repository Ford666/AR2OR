import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np

from os import listdir
from os.path import join
import pytorch_ssim
from SRGAN import Generator, Discriminator, initialize_weights, \
                    generator_loss, discriminator_loss
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#Custom Dataset class
class DatasetFromFolder(Dataset):
    def __init__(self, x_dir, y_dir):
        super(DatasetFromFolder, self).__init__()
        self.x_filenames = [join(x_dir, x) for x in listdir(x_dir)]
        self.y_filenames = [join(y_dir, y) for y in listdir(y_dir)]

    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.x_filenames[index])).float()
        y = torch.from_numpy(np.load(self.y_filenames[index])).float()
        return x, y

    def __len__(self):
        return len(self.x_filenames)

#Optimizing our loss
def get_optimizer(model, lr):
    """
    Construct and return an Adam optimizer for the model with learning rate,
    beta1=0.5, and beta2=0.999.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    return optimizer

#Show images during training
def show_images(images, path):
    images = images.data.cpu().numpy()
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    imgH, imgW = int(images.shape[2]), int(images.shape[3])

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn) 
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.squeeze(img)) 
    if path != None:
        plt.savefig(path) #save plot instead of image itself
    plt.close()

if __name__ == "__main__":
    #SetUp
    f = open('../AROR_train.txt','w+')

    #dtype = torch.FloatTenso
    dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
    BATCH_SIZE = 4
    NUM_EPOCHS = 10

    #Load data
    train_set = DatasetFromFolder('../datasplit/training/x', '../datasplit/training/y')
   
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=BATCH_SIZE)
    
    #Import SRGAN model
    G = Generator().cuda()
    print('# generator parameters:', sum(param.numel() for param in G.parameters()))
    D = Discriminator().cuda()
    print('# discriminator parameters:', sum(param.numel() for param in D.parameters()))
    
    #Vertify model
    data, target = next(enumerate(train_loader))[-1][0], next(enumerate(train_loader))[-1][1] #(BATCH_SIZE,256,256)
    data, target = (data.unsqueeze(1)).type(dtype), (target.unsqueeze(1)).type(dtype)
    show_images(data, '../AR_img.png')
    show_images(target, '../OR_img.png')
    print(data.size())
    outG, outD = G(data), D(data)   
    print(outG.size(), outD.size())

    #Training
    G.apply(initialize_weights)
    D.apply(initialize_weights)
    #G.load_state_dict(torch.load('../datasplit/training/G_model3.pkl'))
    #D.load_state_dict(torch.load('../datasplit/training/D_model3.pkl'))
    
    G_solver, D_solver = get_optimizer(G, 1e-3), get_optimizer(D, 1e-4)

    iter_count = 0
    iter_per_epoch = int(78518/BATCH_SIZE)
    K = 1
    print("Initializing Training!")
    f.write("Initializing Training!\n")
    f.flush()

    for epoch in range(1, NUM_EPOCHS+1):
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        G.train()
        D.train()
        for x, y in train_loader:
            running_results['batch_sizes'] += BATCH_SIZE
            iter_count += 1

            #train Discriminator
            D_solver.zero_grad()
            AR_img, OR_img = (x.unsqueeze(1)).type(dtype), (y.unsqueeze(1)).type(dtype) #(4,1,256,256)
            SR_img = G(AR_img).detach()
            logits_SR, logits_OR = D(SR_img), D(OR_img)
            D_loss = discriminator_loss(logits_OR, logits_SR)
            D_loss.backward()
            D_solver.step()

            #train Generator
            for k in range(K):                
                G_solver.zero_grad()
                SR_img = G(AR_img)
                #logits_SR, logits_OR = D(SR_img), D(OR_img)
                #G_loss = generator_loss(logits_OR,logits_SR,OR_img,SR_img)
                logits_SR = D(SR_img)
                G_loss = generator_loss(logits_SR, OR_img, SR_img)
                G_loss.backward()
                G_solver.step()

            #loss for current batch before optimization 
            running_results['g_loss'] += G_loss.item() * BATCH_SIZE
            running_results['d_loss'] += D_loss.item() * BATCH_SIZE
            #running_results['d_score'] += ((torch.sigmoid(logits_OR-torch.mean(logits_SR))).mean()).item() * BATCH_SIZE
            #running_results['g_score'] += ((torch.sigmoid(logits_SR-torch.mean(logits_OR))).mean()).item() * BATCH_SIZE
            running_results['d_score'] += (torch.sigmoid(logits_OR).mean()).item() * BATCH_SIZE
            running_results['g_score'] += (torch.sigmoid(logits_SR).mean()).item() * BATCH_SIZE

            #Show training results
            if iter_count % 20 == 0:
                img_path = '../datasplit/training/'
                show_images(AR_img, img_path+'AR_iter/'+str(iter_count)+'.png')
                show_images(SR_img, img_path+'SR_iter/'+str(iter_count)+'.png')
                show_images(OR_img, img_path+'OR_iter/'+str(iter_count)+'.png')
                f.write('Epoch: '+str(epoch)+ '\t' + 'Iter/iter_per_epoch: '+str(iter_count)+ '/'+ str(iter_per_epoch)
                         + '\t' + 'Loss_D: ' + str(running_results['d_loss']/running_results['batch_sizes']) +  '\t' + 
                         'Loss_G: ' + str(running_results['g_loss']/running_results['batch_sizes']) + '\t' + 'D(x): '+
                         str( running_results['d_score']/running_results['batch_sizes']) + '\t' + 'D(G(z)): ' +
                         str(running_results['g_score']/running_results['batch_sizes']) + '\n')
                f.flush()
 
        #Save model per epoch
        torch.save(D.state_dict(), '../D_model.pkl')  
        torch.save(G.state_dict(), '../G_model.pkl')                 
        print('Discriminator and Generator saved!')
        f.write('Discriminator and Generator saved!\n')
        f.flush()

    print("Training Done!")
    f.write("Training Done!\n")
    f.close()

            
                
    

    

            



