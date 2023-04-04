# @Version: python3.10
# @Time: 2023/3/28 0:15
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: kmeans.py.py
# @Software: PyCharm
# @User: chent

# import kmeans
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt




'''
	度量距离，此处采用欧式距离
'''
def eucDistance(vec1, vec2):
	return sqrt(sum(pow(vec2 - vec1, 2)))



'''
	初始化聚类中心，随机选择k个点作为初始聚类中心
'''
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = np.zeros((k, dim))
	for i in range(k):
		index = int(np.random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids

def kmeanss(dataSet, k):
	numSamples = dataSet.shape[0]
	clusterAssment = np.mat(np.zeros((numSamples, 2)))
	clusterChanged = True

	## init centroids
	centroids = initCentroids(dataSet, k)

	while clusterChanged:
		clusterChanged = False
		for i in range(numSamples):
			minDist = 100000.0
			minIndex = 0
			## step2: find the centroid who is closest
			for j in range(k):
				distance = eucDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist = distance
					minIndex = j

			## step3: update its cluster
			clusterAssment[i, :] = minIndex, minDist ** 2
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
			## step4: update centroids
			for j in range(k):
				pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
				centroids[j, :] = np.mean(pointsInCluster, axis=0)
	print('Congratulations, cluster complete!')
	return centroids, clusterAssment



'''
	加载数据集
'''
dataSet = []
fileIn = open('kmeans.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split(',')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])

'''
	调用K-Means算法
'''
dataSet = np.mat(dataSet)
k = 4
centroids, clusterAssment = kmeanss(dataSet, k)


def showCluster(dataSet, k, centroids, clusterAssement):
	numSamples, dim = dataSet.shape
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', '<r', 'pr']
	if k > len(mark):
		print('Sorry')
		return 1
	for i in np.xrange(numSamples):
		markIndex = int(clusterAssement[i, 0])
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
	plt.show()



























