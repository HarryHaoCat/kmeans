
import kMeans
from numpy import*
import matplotlib
import matplotlib.pyplot as plt
import time

k = 3
datMat = mat(kMeans.loadDataSet('testSet2.txt'))
centroids, clusterAssment = kMeans.kMeans3(datMat,k)
print centroids,clusterAssment
fig = plt.figure()
ax = fig.add_subplot(111)
centroids = array(centroids)

datMat = array(datMat)
colorList = ['b','c','g','k','r','y']
makerList = ['.','^','*','+','o']
for i in range(k):
    ax.scatter(datMat[kMeans.find_all_index(clusterAssment[:,0], i),0],datMat[kMeans.find_all_index(clusterAssment[:,0], i),1],c = colorList[i%6],marker = makerList[i%5] )
    ax.scatter(centroids[:,0],centroids[:,1],marker = makerList[4] ,c=colorList[4])
plt.title('Graph of k_Means ',)
plt.xlabel('x')
plt.ylabel('y')
print 'The centers of the points is: ',centroids
plt.show()
