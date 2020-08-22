import re
import numpy as np
import matplotlib.pyplot as plt

# Read training records from txt
#Ref: Python--re模块 https://www.cnblogs.com/zjltt/p/6955965.html
Loss_Gs, Loss_Ds, L1Loss, Dxs, DGzs, PSNRs = [], [], [], [], [], []

with open('../datasplit/new_train/train8/PretrainG5（失败）/RecordPretrainG.txt', 'r') as f:
    for line in f.readlines():
        if line.find('Epoch') != -1:
            linestr = list(line.split(','))
            #-?表示匹配1或多个负号，\d+表示匹配0-9数字1或多个，\.?表示匹配小数点0次或1次
            Loss_Ds.append(float(re.findall(r'(-?\d+\.?\d+)',linestr[2])[0])) #re.findall()返回list，加[0]取其中元素
            Loss_Gs.append(float(re.findall(r'(-?\d+\.?\d+)',linestr[3])[0]))
            L1Loss.append(float(re.findall(r'(-?\d+\.?\d+)',linestr[4])[0]))
            Dxs.append(float(re.findall(r'(-?\d+\.?\d+)',linestr[5])[0]))
            DGzs.append(float(re.findall(r'(-?\d+\.?\d+)',linestr[6])[0]))
            PSNRs.append(float(re.findall(r'(-?\d+\.?\d+)',linestr[7])[0]))


fig1, fig2, fig3 = plt.figure(1), plt.figure(2), plt.figure(3)
fig4, fig5, fig6 = plt.figure(4), plt.figure(5), plt.figure(6)
ax1, ax2, ax3 = fig1.add_subplot(111), fig2.add_subplot(111), fig3.add_subplot(111)
ax4, ax5, ax6 = fig4.add_subplot(111), fig5.add_subplot(111), fig6.add_subplot(111)

Recordlen = len(Loss_Gs)
ax1.plot(np.arange(1,Recordlen+1),Loss_Gs)
ax2.plot(np.arange(1,Recordlen+1), L1Loss)
ax3.plot(np.arange(1,Recordlen+1),Loss_Ds)
ax4.plot(np.arange(1,Recordlen+1), Dxs)
ax5.plot(np.arange(1,Recordlen+1), DGzs)
ax6.plot(np.arange(1,Recordlen+1), PSNRs)

Xtick = [1]+list(range(Recordlen//6,Recordlen+1,Recordlen//6))+[Recordlen]
Xlabel = [1000*x for x in Xtick]
Xlabel = tuple([str(a) for a in Xlabel])
ax1.grid(True, linestyle='-')
ax1.set_xticks(Xtick)
ax1.set_xticklabels(Xlabel)
ax1.set_xlabel("Iter")
ax1.set_ylabel("Loss_G")

ax2.grid(True, linestyle='-')
ax2.set_xticks(Xtick)
ax2.set_xticklabels(Xlabel)
ax2.set_xlabel("Iter")
ax2.set_ylabel("L1Loss")

ax3.grid(True, linestyle='-')
ax3.set_xticks(Xtick)
ax3.set_xticklabels(Xlabel)
ax3.set_xlabel("Iter")
ax3.set_ylabel("Loss_D")

ax4.grid(True, linestyle='-')
ax4.set_xticks(Xtick)
ax4.set_xticklabels(Xlabel)
ax4.set_xlabel("Iter")
ax4.set_ylabel("D(x)")

ax5.grid(True, linestyle='-')
ax5.set_xticks(Xtick)
ax5.set_xticklabels(Xlabel)
ax5.set_xlabel("Iter")
ax5.set_ylabel("D(G(z))")

ax6.grid(True, linestyle='-')
ax6.set_xticks(Xtick)
ax6.set_xticklabels(Xlabel)
ax6.set_xlabel("Iter")
ax6.set_ylabel("PSNR")
plt.show()



'''python
#Method(1) 使用readline方法来逐行读取文件
with open(file_path, encoding='utf-8') as file_obj:
    line = file_obj.readline() #末尾带'\n'
    while line != '':
        print(line)
        line = file_obj.readline()

#Method(2) 一次性将文件逐行读取存入一个列表中
with open(file_path, encoding='utf-8') as file_obj:
    lines = file_obj.readlines()

for line in lines:
    print(line)
'''