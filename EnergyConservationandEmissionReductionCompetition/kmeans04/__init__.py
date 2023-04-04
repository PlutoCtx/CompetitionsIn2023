# @Version: python3.10
# @Time: 2023/3/28 14:42
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: kmeans.py.py
# @Software: PyCharm
# @User: chent

import numpy as np
import pandas as pd
import pylab as pl
import random as rd

step = 0
color = ['.r', '.g', '.b', '.y']
dcolor = ['*r', '*g', '*b', '*y']
frames = []

def distance(a, b):
	return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

# K均值算法
def k_means(x, y, k_count):
	count = len(x) # 点的个数
	# 随机选择K个点
	k = rd.sample(range(count), k_count)
	k_point =[[x[i], [y[i]]] for i in k] # 保证有序
	k_point.sort()
	global firames
	global step
	while True:
		km =[[] for i in range(k_count)] # 储每的索
		# 遍历所有点
		for i in range(count):
			cp =[x[i],y[i]] # 当前点
			# 计算cp点到所有质心的距离
			_sse =[distance(k_point[j], cp) for j in range(k_count)]
			# cp点到那个质心最近
			min_index = _sse.index(min(_sse))
			# 把cp点并入第i簇
			km[min_index].append(i)
		# 更换质心
		step += 1
		k_new = []
		for i in range(k_count):
			_x = sum([x[i] for j in km[i]]) / len(km[i])
			_y = sum([y[i] for j in km[i]]) / len(km[i])
			k_new.append([_x, _y])
		k_new.sort()  # 排
		# 使用Matplotlab画图
		pl.figure()
		# 图片显示中文
		pl.rcParams['font.sans-serif'] = ['SimHei']
		pl.rcParams['axes.unicode_minus'] = False  # 减号unicode编码
		pl.title('聚类分析图 类别=%d 迭代次数=%d' %(k_count, step), fontsize=13)
		pl.xlabel("使用意愿 1-》5 意愿逐渐增加", fontsize=13)
		pl.ylabel("了解程度", fontsize=13)

		for j in range(k_count):
			pl.plot([x[i] for i in km[j]], [y[i] for i in km[j]], color[j%4], markersize=15)
			pl.legend('1234')
		pl.savefig("K-means聚类.jpg")
		if (k_new != k_point):  # 一直循环直到聚类中心没有变化
			k_point = k_new
		else:
			return km
		print(km[j])

# x, y = np.loadtxt(julei3.csv, delimiter=',', unpack = True)
x, y = pd.read_excel('data.xlsx', header=0)
k_count = 3  # 聚类个数
km = k_means(x, y, k_count)
print(step)


































