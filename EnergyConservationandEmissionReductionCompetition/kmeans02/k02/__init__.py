# @Version: python3.10
# @Time: 2023/3/27 20:07
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: kmeans.py.py
# @Software: PyCharm
# @User: chent


import math
import random

f = open("kmeans.txt")
a = []
b = []
tag = 1
for k in f.read().split(','):
	if tag == 1:
		a.append(float(k))
		tag = 0
	else:
		b.append(float(k))
		tag = 1
# 到此已经将x坐标和y坐标分别放在了a[]和b[]中

p = len(a)
k = 4
print("数据表的长度为", p, "已知聚类数k =", k)
center = []
for i in range(k):
	while (1):
		j = random.randint(0, p - 1)
		if j not in center:
			center += [j]
			break
	# 到此已经选取了四个0~104的不重复下标
cen_x = [a[i] for i in center]  # 存放种子点横坐标
cen_y = [b[i] for i in center]  # 存放种子点纵坐标

belong = [-1 for i in range(p)]
# print belong
tag = 1
count = 0
while (tag == 1):  # tag==1表示在循环里进行过种子点的移动
	tag = 0
	# 先更新聚类belong[]
	for i in range(p):  # 对每一个点
		index_maxlen = -1  # 存放到谁最近
		maxlen = -1  # 存放最近距离
		for j in range(k):  # 对k个种子点
			numlen = (a[i] - cen_x[j]) ** 2 + (b[i] - cen_y[j]) ** 2  # 存放当前距离
			if maxlen < 0 or numlen < maxlen:  # 如果比maxlen小
				index_maxlen = j  # 记录种子下标
				maxlen = numlen  # 更新更小值
		belong[i] = index_maxlen  # 记录它的种子点下标
	# 再移动种子点
	sum_x = [0 for i in range(4)]
	sum_y = [0 for i in range(4)]  # 分别保存k个种子点对应聚类的横纵坐标和
	numofclass = [0 for i in range(4)]  # 用来保存k个聚类的点数
	for i in range(p):  # 对每一个点
		sum_x[belong[i]] += a[i]
		sum_y[belong[i]] += b[i]  # 将横纵坐标加到它的聚类中去
		numofclass[belong[i]] += 1  # 维护聚类的点数
	for i in range(k):  # 对k个种子点
		if sum_x[i] / numofclass[i] != cen_x[i]:
			tag = 1
			cen_x[i] = sum_x[i] / numofclass[i]
		if sum_y[i] / numofclass[i] != cen_y[i]:
			tag = 1
			cen_y[i] = sum_y[i] / numofclass[i]
	count += 1

x = [[], [], [], []]
y = [[], [], [], []]
for i in range(k):  # 对k个种子点
	for j in range(p):  # 对每个点
		if belong[j] == i:  # 如果是第i个聚类
			x[i].append(a[j])
			y[i].append(b[j])
import numpy
import matplotlib
import pylab as pb

for i in range(k):
	if i == 0:
		pb.plot(x[i], y[i], 'or')
	elif i == 1:
		pb.plot(x[i], y[i], 'ob')
	elif i == 2:
		pb.plot(x[i], y[i], 'og')
	else:
		pb.plot(x[i], y[i], 'ok')
pb.show()