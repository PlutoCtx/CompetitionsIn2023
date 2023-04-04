# @Version: python3.10
# @Time: 2023/3/27 18:32
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: kmeans.py.py
# @Software: PyCharm
# @User: chent

# 导入鸢尾花数据，并重构数据框
import pandas as pd

# iris = load_iris()
iris = pd.read_excel("kmeans.xlsx")

df = pd.DataFrame(iris, columns=iris[0, :])

# 根据前三个特征：利用K-means聚类将数据聚成四类

from sklearn.cluster import KMeans

estimator = KMeans(n_clusters=4)  # 构造聚类器
estimator.fit(df.iloc[:, 0:3])  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
df['label'] = label_pred  # 在原数据表显示聚类标签

# 根据鸢尾花数据前三个特征，绘制三维分类散点图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

# 设置x、y、z轴
x = df['sepal length (cm)']
y = df['sepal width (cm)']
z = df['petal length (cm)']
# 绘图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z, c=label_pred)  # c指颜色，c=label_pred刚好四个分类四个颜色。相比普通三维散点图只改了这里！！！
# 添加坐标轴
ax.set_xlabel('sepal length (cm)', fontdict={'size': 10, 'color': 'black'})
ax.set_ylabel('sepal width (cm)', fontdict={'size': 10, 'color': 'black'})
ax.set_zlabel('petal length (cm)', fontdict={'size': 10, 'color': 'black'})
plt.show()