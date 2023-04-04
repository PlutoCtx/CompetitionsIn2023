# @Version: python3.10
# @Time: 2023/3/27 19:16
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: SSE.py
# @Software: PyCharm
# @User: chent

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

df_features = pd.read_csv(r'julei.csv', encoding='gbk')

SSE = []
for k in range(1, 12):
	estimator = KMeans(n_clusters=k)
	estimator.fit(df_features[['R', 'F']])
	SSE.append(estimator.inertia_)
X = range(1, 12)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()