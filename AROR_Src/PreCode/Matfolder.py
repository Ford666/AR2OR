from __future__ import print_function
import os.path
import torch
import numpy as np
#import cupy as cp
import torch.utils.data as D

def make_dataset(dir):
    pathx_ind = []
    pathy_ind = []
    path_ind = []
    dir = os.path.expanduser(dir) #dir - main path
    # main path - 1. x; 2.y

    d = os.path.join(dir, 'x')#path combination

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            path_x = os.path.join(root, fname)#path combination
            item = (path_x)
            pathx_ind.append(item)
    d = []       
    d = os.path.join(dir, 'y')#path combination

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            path_y = os.path.join(root, fname)#path combination
            item = (path_y)
            pathy_ind.append(item)

    for i in range(len(pathx_ind)):
        item = (pathx_ind[i], pathy_ind[i], i+1)
        path_ind.append(item)
    return path_ind


class DatasetFolder(D.Dataset):

    def __init__(self, root, loader, batch_size):
        sampling = make_dataset(root)                
        self.root = root
        self.loader = loader
        self.bs = batch_size
        self.sampling = sampling
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        pathx, pathy, label = self.sampling[index]
        x = self.loader(pathx)
        y = self.loader(pathy)
        '''
        debug:
        motion = m_ind ----> cause error: cannot unsqueeze empty tensor
        m_ind is a scalar value. 
        Converting m_ind to a list can solve the problem:
        motion = [m_ind]
        '''

        #motion[m_ind]=1
        #y = torch.from_numpy(np.kron(self.num_image[:, :, n_ind-1], np.ones((16,16)))).float()

        return x, y, label

    def __len__(self):
        return len(self.sampling)


def npy_loader(path):
    npy_file = np.load(path)
    Tensor_out = torch.from_numpy(npy_file.astype(float)).float() #dimention to be checked
    #background = torch.from_numpy(mat_file['Background'].astype(float)).float()
    return Tensor_out #(valset_x-background)/torch.max(valset_x-background)


class MatFolder(DatasetFolder):
    def __init__(self, root=None, loader=npy_loader, num_image=None, batch_size=1):
        super(MatFolder, self).__init__(root, loader, batch_size)
        self.path = self.sampling


'''
if __name__ == '__main__':
    num_image = imageloading()
    BATCH_S = 1
    dataset = MatFolder(root=r'F:\movingdataset', num_image=num_image, batch_size=BATCH_S)
    dataloader = D.DataLoader(
        dataset=dataset,
        batch_size=BATCH_S,
        shuffle=True,
        num_workers=0,
    )
    print(len(dataloader))
    for batch_ind, (speckles, motion, image) in enumerate(dataloader):
        t_s = time.time()
        x = speckles
        y1 = motion
        y2 = image
        t = time.time()-t_s
        print(t)
'''