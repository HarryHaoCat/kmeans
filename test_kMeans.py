# -*-coding: utf-8 -*-

# 算法实现流程：
# 创建k个点作为起始质心（经常是随机选择）
# 当任意一个点的簇分配发生改变时
#   对数据集中的每个数据点
#       对每个质心
#           计算质心与数据点之间的距离
#       将数据点分配到距其最近的簇
#   对每一个簇，计算簇中所有点的均值并将均值作为质心

import kMeans
from numpy import*
import matplotlib
import matplotlib.pyplot as plt
import time
st = time.time()
k = 3
dataMat = mat(kMeans.loadDataSet('testSet2.txt'))
oldClassLabel = zeros(len(dataMat),int)
newClassLabel = ones(len(dataMat),int)
center = kMeans.randCent(dataMat, k)
dist = []
m = 0
while newClassLabel.tolist().__eq__(oldClassLabel.tolist()) != True:     # 所有的点的新分类的标签和旧分类的标签不一致时就继续进行划分
      m += 1                                                             # 迭代次数
      for di in range(len(dataMat)):                                     # 对数据集中的每个数据点
          dist = []
          for ci in range(len(center)):                                  # 对每个质心，计算某个点到质心的距离
              dist.append(kMeans.distEclud(dataMat[di], center[ci]))     # dist 用于记录一个点到所有簇点的距离
          distsort = array(dist).argsort()                               # 对距离排序，返回从小到大的索引
          oldClassLabel = newClassLabel.copy()                           # ！！！！注意这里是引用不能直接用等号，否则将会使得两个值一起变
          newClassLabel[di] = distsort[0]                                # 取出索引的最小值，就是距离最近的点
      for j in range(k):
          x = mean(array(dataMat)[kMeans.find_all_index(newClassLabel, j), 0])  # 重新计算簇心的坐标
          y = mean(array(dataMat)[kMeans.find_all_index(newClassLabel, j), 1])
          center[j] = [x, y]
fig = plt.figure()
ax = fig.add_subplot(111)
center = array(center)
ax.scatter(center[:,0],center[:,1],marker = '+' ,c='r')
print center
print m
dataMat = array(dataMat)
et = time.time()
print 'The total time is:',et-st
ax.scatter(dataMat[kMeans.find_all_index(newClassLabel, 0),0],dataMat[kMeans.find_all_index(newClassLabel, 0),1],c = 'b')
ax.scatter(dataMat[kMeans.find_all_index(newClassLabel, 1),0],dataMat[kMeans.find_all_index(newClassLabel, 1),1],c = 'g')
ax.scatter(dataMat[kMeans.find_all_index(newClassLabel, 2),0],dataMat[kMeans.find_all_index(newClassLabel, 2),1],c = 'c')
ax.scatter(dataMat[kMeans.find_all_index(newClassLabel, 3),0],dataMat[kMeans.find_all_index(newClassLabel, 3),1],c = 'k')
plt.show()
