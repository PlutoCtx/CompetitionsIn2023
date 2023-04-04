# @Version: python3.10
# @Time: 2023/3/27 19:15
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: kmeans.py.py
# @Software: PyCharm
# @User: chent

import random as rd

import numpy as np
import pylab as pl

step = 0
color = ['.r', '.g', '.b', '.y']
dcolor = ['*r', '*g', '*b', '*y']
frames = []

def distance(a, b):
	return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# K均值
def k_means(x, y, k_count):
	count = len(x)

	k = rd.sample(range(count), k_count)
	k_point = [[x[i], [y[i]]] for i in k]
	k_point.sort()
	global frames
	global step
	while True:
		km = [[] for i in range(k_count)]

		for i in range(count):
			cp = [x[i], y[i]]
			_sse = [distance(k_point[j], cp) for j in range(k_count)]
			min_index = _sse.index(min(_sse))
			km[min_index].append(i)

		step += 1
		k_new = []
		for i in range(k_count):
			_x = sum([x[j] for j in km[i]]) / len(km[i])
			_y = sum([y[j] for j in km[i]]) / len(km[i])
			k_new.append([_x, _y])
		k_new.sort()


		pl.figure()
		pl.rcParams['font.sans-serif'] = ['SimHei']
		pl.rcParams['axes.unicode_minus'] = False
		pl.title('聚类分析图 类别=%d 迭代次数：%d' %(k_count, step), fontsize=13)
		pl.xlabel('1->5 逐渐增加', fontsize=13)
		pl.xlabel('1->4 逐渐增加', fontsize=13)
		for j in range(k_count):
			pl.plot([x[i] for i in km[j]], [y[i] for i in km[j]], color[j % 4], markersize=1)
			pl.legend('1234')
		pl.savefig('KMEANS.jpg')
		if(k_new != k_point):
			k_point = k_new
		else:
			return km
		print(km[j])

x, y = np.loadtxt('kmeans.txt', delimiter=',', unpack=True)
print(x, y)
k_count = 4
km = k_means(x, y, k_count)
print(step)
























