from utils import *
import re


#Load processed AR-OR data(FOV Matched)
AR_npy = np.load('../PAaror/ProNpy/ARNpy.npy') 
OR_npy = np.load('../PAaror/ProNpy/ORNpy.npy') 

#Read .mat directories
Matpath, Augpath = "../PAaror", "../Augdata"
AugARDirs, AugORDirs = [], []
for x in os.listdir(Matpath):  #遍历指定路径下的文件
    if os.path.splitext(x)[1] == '.mat':  #筛选.mat文件
        if re.findall('AR', x):
            AugARDirs.append(os.path.join(Augpath+"/AR",os.path.splitext(x)[0]))
        elif re.findall('OR', x):
            AugORDirs.append(os.path.join(Augpath+"/OR",os.path.splitext(x)[0]))

# Data argumentation
# #Pre-experiment
# AR,OR = AR_npy[7], OR_npy[7]
# TransNum = 11
# AugAR, AugOR = np.zeros([TransNum, AR.shape[0], AR.shape[1]], dtype= np.float32), \
#                 np.zeros([TransNum, OR.shape[0], OR.shape[1]], dtype= np.float32)

# AugAR[0], AugOR[0] = Add_Gaussian_Noise(AR,0), Add_Gaussian_Noise(OR,0)
# AugAR[1], AugOR[1] = Add_Gaussian_Noise(AR,1), Add_Gaussian_Noise(OR,1)
# AugAR[2], AugOR[2] = Gamma_Transform(AR,0), Add_Gaussian_Noise(OR,0)
# AugAR[3], AugOR[3] = Gamma_Transform(AR,2), Add_Gaussian_Noise(OR,2)
# AugAR[4], AugOR[4] = Elastic_Deformation(AR,sigma=6, seed=50), Elastic_Deformation(OR,sigma=6, seed=50)
# AugAR[5], AugOR[5] = Elastic_Deformation(AR,sigma=6, seed=60), Elastic_Deformation(OR,sigma=6, seed=60)
# AugAR[6], AugOR[6] = New_Rotation(AR,1), New_Rotation(OR,1)
# AugAR[7], AugOR[7] = New_Rotation(AR,2), New_Rotation(OR,2)
# AugAR[8], AugOR[8] = Flip(AR,1), Flip(OR,1)
# AugAR[9], AugOR[9] = Flip(AR,2), Flip(OR,2)
# AugAR[10], AugOR[10] = Flip(AR,3), Flip(OR,3)

# for i in range(TransNum):
#     ARdir, ORdir = "../Augdata/AR%d.png" % i, "../Augdata/OR%d.png" % i
#     plt.imsave(ARdir, AugAR[i], cmap=cm.hot)
#     plt.imsave(ORdir, AugOR[i], cmap=cm.hot)



#4 flipping * 3 rotation * 2 elastic deformation * 3 gamma trasform = 72
parms, parm = [], {}
for i in range(4):
    for j in range(3):
        for k in range(1,2+1):
            for m in range(3):
                parm = {'flip_option':i, 'rot_option':j, 
                            'ela_parm':{'sigma':6, 'seed':10*(k-1)+50}, 'gamma_option':m}
                parms.append(parm)

# save parms list as json file
parms_json = open('../Augdata/parms.json', 'w+')
for i in parms:
    parms_json.write(json.dumps(i)+'\n')
parms_json.close()

TransNum = len(parms)
for i in range(AR_npy.shape[0]):
    if i==1:  #AROR_03 size of (1664,1344)
        H, W = 1664, 1344 
    elif i==7:  #AROR_ear_1 size of (2000,1471)
        H, W = 2000, 1471
    else:
        H, W = 2000, 2000
    for j in range(TransNum):
                ARdir, ORdir = AugARDirs[i], AugORDirs[i] 
                ARdir, ORdir =  ARdir+"_"+str(j).zfill(2)+".npy", ORdir+"_"+str(j).zfill(2)+".npy"   

                AR = Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(AR_npy[i,0:H,0:W], 
                        parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
                        parms[j]['ela_parm']['seed']), parms[j]['gamma_option'])
                
                OR = Gamma_Transform(Elastic_Deformation(New_Rotation(Flip(OR_npy[i,0:H,0:W], 
                        parms[j]['flip_option']), parms[j]['rot_option']), parms[j]['ela_parm']['sigma'],
                        parms[j]['ela_parm']['seed']), parms[j]['gamma_option'])
                np.save(ARdir, AR.astype(np.float32))
                np.save(ORdir, OR.astype(np.float32))
            