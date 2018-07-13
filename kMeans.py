# -*-coding: utf-8 -*-

# 算法实现流程：
# 创建k个点作为起始质心（经常是随机选择）
# 当任意一个点的簇分配发生改变时
#   对数据集中的每个数据点
#       对每个质心
#           计算质心与数据点之间的距离
#       将数据点分配到距其最近的簇
#   对每一个簇，计算簇中所有点的均值并将均值作为质心

from numpy import*
import  time
import matplotlib
import matplotlib.pyplot as plt

# 加载数据集
def loadDataSet(filename):
    dataMat = [map(float,lines.strip().split('\t')) for lines in open(filename).readlines()]
    return mat(dataMat)

# 计算距离,输入为array类型，只有array才能实现对应元素相减的操作
def distEclud(vecA, vecB):
    vecA = array(vecA)
    vecB = array(vecB)
    return sqrt(sum(power(vecA - vecB,2)))

# 构建k个随机簇中心
def randCent2(dataSet , k):
    dataSet = array(dataSet)
    center = mat(zeros((k,dataSet.shape[1])))
    for j in range(dataSet.shape[1]):
        center[:,j] = max(dataSet[:,j]) * random.rand(k,1)+ (1.0-random.rand(k,1)) * min(dataSet[:,j])   # 保证所取到的点一定在相应的数据范围内
    return center

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item]

# k-均值算法
def kMeans2(dataSet,k,distMeans = distEclud,creatCent= randCent):
    m = shape(dataSet)[0]                        # 返回数据的行
    clusterAssment = mat(zeros((m,2)))          # 拿来存储每个数据点的类别和到最近簇点的距离
    centroids = creatCent(dataSet,k)
    clusterChanged = True
    iterations = 0
    while clusterChanged:
        iterations += 1
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        for cent in range(k):
            ptsInClust = dataSet[[index for index,item in enumerate(clusterAssment[:,0]) if int(item) == cent],:]
            #ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust , axis=0)
    #fig = plt.figure()
    #print 'The iterations is:',iterations
    #ax = fig.add_subplot(111)
    #centroids = array(centroids)
    #ax.scatter(centroids[:,0],centroids[:,1], c = 'r', marker = '.')
    #ax.scatter(array(dataSet)[[index for index, item in enumerate(clusterAssment[:, 0]) if item == 0], 0],
     #          array(dataSet)[[index for index, item in enumerate(clusterAssment[:, 0]) if item == 0], 1], c = 'b', marker = '^')
    #ax.scatter(array(dataSet)[[index for index, item in enumerate(clusterAssment[:, 0]) if item == 1], 0],
    #           array(dataSet)[[index for index, item in enumerate(clusterAssment[:, 0]) if item == 1], 1], c = 'g', marker = '*')
    #ax.scatter(array(dataSet)[[index for index, item in enumerate(clusterAssment[:, 0]) if item == 2], 0],
     #          array(dataSet)[[index for index, item in enumerate(clusterAssment[:, 0]) if item == 2], 1], c = 'c', marker = 'o')
    #ax.scatter(array(dataSet)[[index for index, item in enumerate(clusterAssment[:, 0]) if item == 3], 0],
     #          array(dataSet)[[index for index, item in enumerate(clusterAssment[:, 0]) if item == 3], 1], c = 'y', marker = '+')
    #plt.show()
    return centroids, clusterAssment

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    #st = time.time()
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    et = time.time()
    #print 'The total time is:', et - st
    return centroids, clusterAssment


# 将所有点看成一个簇
# 当簇数目小于k时
#   对于每一个簇
#       计算总误差
#       在给定的簇上面进行k-均值聚类（k = 2）
#       计算将该簇一分为二之后的总误差
#   选择使得误差最小的那个簇进行划分操作

def biKmeans2(dataSet, k):
    centroids, clusterAssment = kMeans(dataSet,1)
    centroids = centroids.tolist()
    iterations = 0
    while len(centroids) < k:
        iterations += 1
        lowSse = inf
        classIndex = 0
        for i in range(len(centroids)):
            subDatMat = mat(dataSet[[index for index,item in enumerate(clusterAssment[:,0]) if int(item) == i],:])
            #sse = sum(clusterAssment[[index for index, item in enumerate(clusterAssment[:, 0]) if int(item) == i], 1])\
            #     /float(len(clusterAssment[[index for index, item in enumerate(clusterAssment[:, 0]) if int(item) == i], 1]))
            subCentroids, subClusterAssment = kMeans(subDatMat,2)
            subCentroids = subCentroids.tolist()
            subSse = sum(subClusterAssment[:,1], axis = 0) + sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1], axis = 0)    # /float(len(subClusterAssment))
            if subSse < lowSse:
                lowSse = subSse
                classIndex = i
        subDatMat = dataSet[[index for index, item in enumerate(clusterAssment[:, 0]) if int(item) == classIndex], :]
        subCentroids, subClusterAssment = kMeans(subDatMat, 2)
        subCentroids = subCentroids.tolist()
        centroids[classIndex] = subCentroids[0]
        centroids.append(subCentroids[1])
        #centroids = row_stack((centroids,subCentroids[1,:]))
        subClusterAssment[[index for index,item in enumerate(subClusterAssment[:,0]) if int(item) == 1],0] = len(centroids) - 1
        subClusterAssment[[index for index, item in enumerate(subClusterAssment[:, 0]) if int(item) == 0], 0] = classIndex

        clusterAssment[[index for index, item in enumerate(clusterAssment[:, 0]) if int(item) == classIndex],:] = subClusterAssment.copy()
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))                   # 存储每一个样本点的类标签和距离簇中心的距离
    centroid0 = mean(dataSet, axis=0).tolist()[0]        # 初始化簇中心
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2   # 初始化clusterAssment的第二列距离
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans3(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        #print 'the bestCentToSplit is: ',bestCentToSplit
        #print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

def kMeans3(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    newcentroids = createCent(dataSet, k)
    clusterChanged = True
    #i = 0
    while clusterChanged:
        clusterChanged = False
        #i += 1
        dataSet1 = dataSet.T
        g = tile(sum((multiply(centroids,centroids)),axis = 1),(1,m))-2*centroids*dataSet1
        clusterAssment[:,0] = argsort(g,axis = 0).T[:,0]
        clusterAssment[:,1] = (g.max(axis=0) + sum((multiply(dataSet1,dataSet1)),axis = 0)).T
        #print shape(clusterAssment)
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]                # get all the point in this cluster
            newcentroids[cent,:] = mean(ptsInClust, axis=0)                                    # assign centroid to mean
        if newcentroids.tolist().__eq__(centroids.tolist()) != True:
            centroids = newcentroids.copy()
            clusterChanged = True
    return centroids, clusterAssment


#% % 采用K - means + +算法初始化聚类中心
#centroids = X(:, 1 + round(rand * (size(X, 2) - 1)));
#labels = ones(1, size(X, 2));
#for i = 2:k
#D = X - centroids(:, labels);
#D = cumsum(sqrt(dot(D, D, 1)));
#if D(end) == 0, centroids(:,
#    i:k) = X(:, ones(1, k - i + 1)); return;
#end
#centroids(:, i) = X(:, find(rand < D / D(end), 1));
#[~, labels] = max(bsxfun( @ minus, 2 * real(centroids'*X),dot(centroids,centroids,1).'));
#end

#% % 标准Kmeans算法
#for iter = 1:max_iter
#for i = 1:k, l = labels == i; centroids(:, i) = sum(X(:, l), 2) / sum(l);
#end
#[~, labels] = max(bsxfun( @ minus, 2 * real(centroids'*X),dot(centroids,centroids,1).'), [], 1);
#end
####kmeans++和kmeans,这里的dataSet是列是样本
def kMeans4(dataSet,k):
    centroids = dataSet[:,1 + int(round(((random.rand() * (dataSet.shape[1] - 1)))))]   # 随机取一个簇的中心
    labels = ones((1,dataSet.shape[1]))                                                 # 初始化类标签
    for i in range(1,k):
        D = dataSet - centroids[:, labels]
