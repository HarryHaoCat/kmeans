# -*- coding: utf-8 -*-

import kMeans
from numpy import*
from matplotlib.pyplot import*
import matplotlib.pyplot as plt
import cv2
from skimage import io,data,color
import time

st = time.time()
my_ima = imread('city.jpg')
fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
imshow(my_ima)
lab = color.rgb2lab(my_ima)
ab = double(lab[:,:,1:3])
nrows = ab.shape[0]
ncols = ab.shape[1]
X = ab.reshape(nrows*ncols,2)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X[:,0],X[:,1])

k = 10
centList, clusterAssment = kMeans.biKmeans(X,k)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
centroids = array(centList)
datMat = array(X)
colorList = ['b','c','g','k','r','y','m','w']
makerList = ['.','^','*','+','o']
for i in range(k):
    ax2.scatter(datMat[kMeans.find_all_index(clusterAssment[:,0], i),0],datMat[kMeans.find_all_index(clusterAssment[:,0], i),1],c = colorList[i%8],marker = makerList[3])
    ax2.scatter(centroids[:,0],centroids[:,1],marker = makerList[4] ,c=colorList[4])
plt.title('Graph of k_Means ')
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
Y = (30*(clusterAssment[:,0].T)[0]).reshape(nrows,ncols)
#Yrgb = color.lab2rgb(Y)
et = time.time()
print 'The total time is:', et-st
plt.imshow(Y)
plt.axis('off')
show()



