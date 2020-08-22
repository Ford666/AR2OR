import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data analysis

# csv = pd.read_csv('../datasplit/esemble_train/train7/SRGAN_esemble_train(1-5 cycle).csv', sep = ',')
# Iters = csv[:]['Iters'].values[:]
# Loss_D = csv[:]['Loss_D'].values[:]
# Loss_G = csv[:]['Loss_G'].values[:]
# PSNR = csv[:]['PSNR'].values[:]
# G_lr = csv[:]['G_lr'].values[:]
# D_lr = csv[:]['D_lr'].values[:]

# iter_num = np.max(Iters)
# ITER_PER_EPOCH = 25569
# NUM_CYCLES = 5
# T_mult = 2
# EPOCH_PER_CYCLE = [T_mult**ele for ele in range(NUM_CYCLES)]
# EPOCH_SUM = [sum(EPOCH_PER_CYCLE[:ele+1]) for ele in range(len(EPOCH_PER_CYCLE))]
# ITER_SUM = np.array([0]+[ITER_PER_EPOCH*ele for ele in EPOCH_SUM])   


# # Plot cyclic annealing scheduler
# plt.figure()
# plt.plot(Iters, D_lr, 'b-', label="D_lr", linewidth=2)
# plt.plot(Iters, G_lr, 'r-', label="G_lr", linewidth=2)
# plt.xticks(ITER_SUM , ("0", "1", "3", "7", "15", "31"))
# plt.grid(True, linestyle='-')
# plt.xlabel('Epochs')
# plt.ylabel('Learning Rate')
# plt.title("Cyclic Cosine Annealing ")
# plt.legend(loc = 'upper right') 

# # Plot loss curve
# plt.figure()
# plt.plot(Iters, Loss_D, 'b-', label="Loss_D", linewidth=0.5)
# plt.plot(Iters, Loss_G, 'r-', label="Loss_G", linewidth=0.5)
# plt.xticks(ITER_SUM , ("0", "1", "3", "7", "15", "31"))

# plt.grid(True, linestyle='-')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title("SRGAN Loss curve of esemble_train")
# plt.legend(loc = 'upper right')

# plt.figure()
# plt.plot(Iters, Loss_G, 'r-', linewidth=0.5)
# plt.ylim([0, 0.02])
# plt.xticks(ITER_SUM , ("0", "1", "3", "7", "15", "31"))
# plt.grid(True, linestyle='-')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title("Generator Loss curve of esemble_train")

# #plot PSNR 
# plt.figure()
# plt.plot(Iters,  PSNR, 'g-')
# plt.xticks(range(0,iter_num+1,100000), ("0", "100k", "200k", "300k", "400k", "500k", "600k","700k"))

# # plt.xlim([])
# plt.ylim([0,30])
# plt.grid(True, linestyle='-')
# plt.xlabel('Iterations')
# plt.ylabel('PSNR')
# plt.title("esemble_train: Average PSNR of 384x384 image patch")
# plt.show()


Loss_G1 = [0.182464,0.154460,0.154384,0.156436,0.151329,0.151666,
0.148999,0.144204,0.143406,0.140247,0.138683,0.138445]

Loss_G2 = [0.446761,0.408333,0.405068,0.406923,0.403889,0.406599,
0.395124,0.399339,0.401075,0.404917,0.404167,0.404644]

Loss_G3 = [0.448329,0.408718,0.413490,0.411981,0.407564,0.408991,
0.404845,0.402900,0.404077,0.410163,0.408820,0.409315]

plt.figure()
plt.plot(range(1,12+1),Loss_G1,'k-',label='Test on 7th(train set)')
plt.plot(range(1,12+1),Loss_G2,'b-',label='Test on 8th(val set)')
plt.plot(range(1,12+1),Loss_G3,'r-',label='Test on 10th(test set)')

plt.grid(True, linestyle='-')
plt.legend(loc = 'lower right')
plt.xlabel('Epochs')
plt.ylabel('Loss_G')
plt.title("Generator Loss during test")
plt.show()
