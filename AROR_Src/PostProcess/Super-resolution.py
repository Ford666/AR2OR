import os
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import ConnectionPatch
from PIL import Image
import numpy as np

#Supre-resolution
#Method1：对比某根线段上的像素灰度分布

SRdir, ORdir = "../datasplit/test/SR_cycle5.tiff", "../datasplit/test/OR.tiff"
SRimg = np.asarray(Image.open(SRdir).convert('L'))
ORimg = np.asarray(Image.open(ORdir).convert('L'))
if ORimg.shape != SRimg.shape:
    ORimg = ORimg[0:6*384-5*64, 0:6*384-5*64]

#Get the line segment endpoints
fig1 = plt.figure(1, figsize=(14,7))  
ax11, ax12 = fig1.add_subplot(1,2,1), fig1.add_subplot(1,2,2)
ax11.imshow(SRimg, cmap=cm.hot)
ax12.imshow(ORimg, cmap=cm.hot)
ax11.set_xticks([])  
ax11.set_yticks([]) 
ax12.set_xticks([])  
ax12.set_yticks([]) 
pos=plt.ginput(2)

#Repolt the line segment, X：rightwards, Y: downwards
PointX, PointY = [pos[0][0],pos[1][0]], [pos[0][1],pos[1][1]]
ax11.imshow(SRimg, cmap=cm.hot)
ax12.imshow(ORimg, cmap=cm.hot)
ax11.plot(PointX, PointY, 'w--', linewidth=2)
ax12.plot(PointX, PointY, 'w--', linewidth=2)

#Get values of the pixels within the line segment
MinH, MaxH = int(min(pos[0][1], pos[1][1])), int(max(pos[0][1], pos[1][1]))
MinW, MaxW = int(min(pos[0][0], pos[1][0])), int(max(pos[0][0], pos[1][0]))
k = (pos[1][1]-pos[0][1])/(pos[1][0]-pos[0][0]) 
LineX = [x for x in range(MinW, MaxW+1)]
LineY=  [int(k*(x-pos[0][0])+pos[0][1]) for x in LineX]
SRPixValue = [SRimg[x,y] for x, y in zip(LineX, LineY)]
ORPixValue = [ORimg[x,y] for x, y in zip(LineX, LineY)]

#Plot the distribution of pixel value 
ax13 = fig1.add_axes([0.02,0.05,0.1,0.1],facecolor='none')
ax14 = fig1.add_axes([0.52,0.05,0.1,0.1],facecolor='none')
ax13.plot(SRPixValue,'w-',linewidth=2)
ax14.plot(ORPixValue,'w-',linewidth=2)
ax13.set_xticks([])  
ax13.set_yticks([])  
ax14.set_xticks([])  
ax14.set_yticks([]) 
axPoss = ['top','bottom','left','right']
for i in axPoss:  #axes aren't shown
    ax13.spines[i].set_visible(False)
    ax14.spines[i].set_visible(False)

fig1.tight_layout()
plt.show()

#Method2: 标出ROI，放大构成subimage附在原图上
fig2 = plt.figure(2, figsize=(14,7))
ax21, ax22 = fig2.add_subplot(1,2,1), fig2.add_subplot(1,2,2)

ax21.imshow(SRimg, cmap=cm.hot)
ax22.imshow(ORimg, cmap=cm.hot)
BoxX = [int(pos[0][0]), int(pos[0][0]), int(pos[1][0]), int(pos[1][0]), int(pos[0][0])]
BoxY = [int(pos[0][1]), int(pos[1][1]), int(pos[1][1]), int(pos[0][1]), int(pos[0][1])]
ax21.plot(BoxX, BoxY, 'w--', linewidth=2)
ax22.plot(BoxX, BoxY, 'w--', linewidth=2)
ax21.set_xticks([])  
ax21.set_yticks([]) 
ax22.set_xticks([])  
ax22.set_yticks([]) 

#嵌入局部放大图的坐标系
ax23 = ax21.inset_axes((0.6, 0, 0.4, 0.4)) #左下角坐标(x0,y0,width,height)
ax24 = ax22.inset_axes((0.6, 0, 0.4, 0.4)) 
ax23.imshow(SRimg[MinH:MaxH+1, MinW:MaxW+1],cmap=cm.hot)
ax24.imshow(ORimg[MinH:MaxH+1, MinW:MaxW+1],cmap=cm.hot)
ax23.set_xticks([])  
ax23.set_yticks([]) 
ax24.set_xticks([])  
ax24.set_yticks([])
axPoss = ['top','bottom','left','right']
for i in axPoss:  #axes aren't shown
    ax23.spines[i].set_color('white')
    ax23.spines[i].set_linestyle('--')
    ax23.spines[i].set_linewidth('2')
    
    ax24.spines[i].set_color('white')
    ax24.spines[i].set_linestyle('--')
    ax24.spines[i].set_linewidth('2')

#画两条连接线
SubH, SubW = MaxH - MinH, MaxW - MinW
xy = (0,SubH) #子图ax23的点
xy2 = (MinW,MaxH) #主图ax21的点
con1 = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data",coordsB="data",
                        axesA = ax23, axesB=ax21)
con1.set_color([1, 1, 1])
con1.set_linewidth(2)
con1.set_linestyle('--')
ax23.add_artist(con1)

con2 = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data",coordsB="data",
                        axesA = ax24, axesB=ax22)
con2.set_color([1, 1, 1])
con2.set_linewidth(2)
con2.set_linestyle('--')                        
ax24.add_artist(con2)


xy = (SubW,0) #子图ax23的点
xy2 = (MaxW,MinH) #主图ax21的点
con1 = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data",coordsB="data",
                        axesA = ax23, axesB=ax21)
con1.set_color([1, 1, 1])
con1.set_linewidth(2)
con1.set_linestyle('--')
ax23.add_artist(con1)

con2 = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data",coordsB="data",
                        axesA = ax24, axesB=ax22)
con2.set_color([1, 1, 1])
con2.set_linewidth(2)
con2.set_linestyle('--') 
ax24.add_artist(con2)

fig2.tight_layout()

plt.show()

