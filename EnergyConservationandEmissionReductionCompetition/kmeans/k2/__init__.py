# @Version: python3.10
# @Time: 2023/3/27 17:52
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: kmeans.py.py
# @Software: PyCharm
# @User: chent



# # coding:utf-8
# import numpy as np
# import pylab as pl
# import random as rd
# import imageio
# #计算平面两点的欧氏距离
# step=0
# color=['.r','.g','.b','.y']#颜色种类
# dcolor=['*r','*g','*b','*y']#颜色种类
# frames = []
# def distance(a, b):
#     return (a[0]- b[0]) ** 2 + (a[1] - b[1]) ** 2
# #K均值算法
# def k_means(x, y, k_count):
#     count = len(x)      #点的个数
#     #随机选择K个点
#     k = rd.sample(range(count), k_count)
#     k_point = [[x[i], [y[i]]] for i in k]   #保证有序
#     k_point.sort()
#     global frames
#     global step
#     while True:
#         km = [[] for i in range(k_count)]      #存储每个簇的索引
#         #遍历所有点
#         for i in range(count):
#             cp = [x[i], y[i]]                   #当前点
#             #计算cp点到所有质心的距离
#             _sse = [distance(k_point[j], cp) for j in range(k_count)]
#             #cp点到那个质心最近
#             min_index = _sse.index(min(_sse))
#             #把cp点并入第i簇
#             km[min_index].append(i)
#         #更换质心
#         step+=1
#         k_new = []
#         for i in range(k_count):
#             _x = sum([x[j] for j in km[i]]) / len(km[i])
#             _y = sum([y[j] for j in km[i]]) / len(km[i])
#             k_new.append([_x, _y])
#         k_new.sort()        #排序
#
#         #使用Matplotlab画图
#         pl.figure()
#         pl.title("N=%d,k=%d  iteration:%d"%(count,k_count,step))
#         for j in range(k_count):
#             pl.plot([x[i] for i in km[j]], [y[i] for i in km[j]], color[j%4])
#             pl.plot(k_point[j][0], k_point[j][1], dcolor[j%4])
#         pl.savefig("1.jpg")
#         frames.append(imageio.imread('1.jpg'))
#         if (k_new != k_point):#一直循环直到聚类中心没有变化
#             k_point = k_new
#         else:
#             return km
#
# #计算SSE
# # def calc_sse(x, y, k_count):
# #     count = len(x)                              #点的个数
# #     k = rd.sample(range(count), k_count)        #随机选择K个点
# #     k_point = [[x[i], [y[i]]] for i in k]
# #     k_point.sort()                              #保证有序
# #     #centroid
# #     sse = [[] for i in range(k_count)]
# #     while True:
# #         ka = [[] for i in range(k_count)]      #存储每个簇的索引
# #         sse = [[] for i in range(k_count)]
# #         #遍历所有点
# #         for i in range(count):
# #             cp = [x[i], y[i]]                   #当前点
# #             #计算cp点到所有质心的距离
# #             _sse = [distance(k_point[j], cp) for j in range(k_count)]
# #             #cp点到那个质心最近
# #             min_index = _sse.index(min(_sse))
# #             #把cp点并入第i簇
# #             ka[min_index].append(i)
# #             sse[min_index].append(min(_sse))
# #         #更换质心
# #         k_new = []
# #         for i in range(k_count):
# #             _x = sum([x[j] for j in ka[i]]) / len(ka[i])
# #             _y = sum([y[j] for j in ka[i]]) / len(ka[i])
# #             k_new.append([_x, _y])
# #         k_new.sort()        #排序
# #         #更换质心
# #         if (k_new != k_point):
# #             k_point = k_new
# #         else:
# #             break
# #     s =0
# #     for i in range(k_count):
# #         s += sum(sse[i])
# #     return s
# x, y = np.loadtxt('kmeans.csv', delimiter=',', unpack=True)
# k_count = 4
#
# km = k_means(x, y, k_count)
# print(step)
# imageio.mimsave('k-means.gif', frames, 'GIF', duration = 0.5)
#



'''
@Author: your name
@Date: 2020-04-07 10:52:14
@LastEditTime: 2020-04-07 13:51:27
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \learn\kmeans.py
'''

import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# 从Excel中读取数据存入数组
rawData = xlrd.open_workbook('kmeans.xlsx')
table = rawData.sheets()[0]
data = []
for i in range(table.nrows):
    if i == 0:
        continue
    else:
        data.append(table.row_values(i)[1:])
featureList = ['责任主体', '意愿支付']
mdl = pd.DataFrame.from_records(data, columns=featureList)
# 聚类
mdl_new = np.array(mdl[['责任主体', '意愿支付']])   # 将其转化为数组
seed = 9    # 设置随机数
clf = KMeans(n_clusters=3, random_state=seed)   # 构造k-means聚类器
clf.fit(mdl_new)    # 拟合模型
print(clf.cluster_centers_)     # 以数组形式查看KMeans聚类后的质心点，即聚类中心。
mdl['label'] = clf.labels_  # 对原数据表进行类别标记
c = mdl['label'].value_counts()
print(mdl.values)   # 以数组形式打印结果
#图形化展示
label_pred = clf.labels_    # 获取聚类标签
centroids = clf.cluster_centers_    # 获取聚类中心
inertia = clf.inertia_ # 获取聚类准则的总和
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
# 这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
color = 0
j = 0
for i in label_pred:
    plt.plot([mdl_new[j:j+1,0]], [mdl_new[j:j+1,1]],
     mark[i], markersize = 5)
    j +=1
plt.show()  # 画出聚类结果简易图

