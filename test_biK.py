import kMeans
from numpy import*
import matplotlib
import matplotlib.pyplot as plt

k = 4
datmat = array(kMeans.loadDataSet('testSet.txt'))
centerList, clusterAssment = kMeans.biKmeans(datmat,k)
print 'The cendroids is:',centerList
fig = plt.figure()
fig.add_subplot(111)
colorList = ['b','c','g','k','r','y']
makerList = ['.','^','*','o','+']
for i in range(k):
    ax = plt.scatter(datmat[nonzero(clusterAssment[:,0].A == i)[0],0],datmat[nonzero(clusterAssment[:,0].A == i)[0],1],
                     c = colorList[i],marker=makerList[i])
    ax = plt.scatter(array(centerList[:,0]),array(centerList[:,1]),c = colorList[4],marker=makerList[3])
plt.title('Graph of k_Means ',)
plt.xlabel('x')
plt.ylabel('y')
plt.show()