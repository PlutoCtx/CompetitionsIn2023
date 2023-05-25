# @Version: python3.10
# @Time: 2023/5/22 7:13
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: points.py
# @Software: PyCharm
# @User: chent

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.randint(1, 5, size=(600, 4))

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

labels = kmeans.labels_

counts = [len(labels[labels==i]) for i in range(3)]

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()



