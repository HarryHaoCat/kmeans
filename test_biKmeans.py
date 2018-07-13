# -*- coding: utf-8 -*-

import kMeans
from numpy import*
import matplotlib
import matplotlib.pyplot as plt
import time

st = time.time()
k = 4
datMat = mat(kMeans.loadDataSet('testSet.txt'))
centroids, clusterAssment = kMeans.biKmeans2(datMat,k)
fig = plt.figure()
ax = fig.add_subplot(111)
centroids = array(centroids)

datMat = array(datMat)
colorList = ['b','c','g','k','r','y']
makerList = ['.','^','*','+','o']
for i in range(k):    # c = 15*array(tile([i],(1,len(kMeans.find_all_index(clusterAssment[:,0], i))))[0])
    ax.scatter(datMat[kMeans.find_all_index(clusterAssment[:,0], i),0],datMat[kMeans.find_all_index(clusterAssment[:,0], i),1], c = colorList[i % 6], marker = makerList[i % 5] )
    ax.scatter(centroids[:,0],centroids[:,1],marker = makerList[4] ,c=colorList[4])
plt.title('Graph of k_Means ',)
plt.xlabel('x')
plt.ylabel('y')
print 'The centers of the points is: ',centroids
et = time.time()
print 'The total time is:',et - st
plt.show()


