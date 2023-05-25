# @Version: python3.10
# @Time: 2023/5/22 6:53
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: hot.py
# @Software: PyCharm
# @User: chent

import matplotlib.pyplot as plt
import numpy as np

data = np.array([[100, 150, 120, 80], [100, 150, 120, 80]])  # 以二维数组的形式存储数据
fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.Blues)  # 绘制热力图
plt.colorbar(heatmap)
ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()

plt.show()
