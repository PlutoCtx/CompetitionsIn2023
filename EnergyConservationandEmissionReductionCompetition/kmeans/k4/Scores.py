# @Version: python3.10
# @Time: 2023/3/27 19:20
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: Scores.py
# @Software: PyCharm
# @User: chent

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df_features = pd.read_csv(r'julei.csv', encoding='gbk')

Scores = []
for k in range(2, 9):
	estimator = KMeans(n_clusters=k)
	estimator.fit(df_features[['R', 'F']])
	Scores.append(silhouette_score(df_features[['R', 'F']], estimator.labels_, metric='euclidean'))
X = range(2, 9)
plt.xlabel('k')
plt.ylabel('轮廓系数')
plt.plot(X, Scores, 'o-')
plt.show()
