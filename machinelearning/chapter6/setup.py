# @Version: python3.10
# @Time: 2023/5/2 19:41
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: setup.py
# @Software: PyCharm
# @User: chent

# import numpy as np

# a = np.array([1, 2, 3, 4])
# b = np.array((5, 6, 7, 8))
# c = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])

# print(a.shape)
# print(b.shape)
# print(c.shape)

# c.shape = 4, 3
# print(c)

# c.shape = 2, -1
# print(c)

# print(a.dtype)

# e = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=np.float)

# print(e)

# f = np.arange(0, 1, 0.1)
# print(f)

# g = np.linspace(0, 1, 12)
# print(g)

# h = np.logspace(0, 2, 10)
# print(h)

# a = np.arange(10)
# print(a)

# import time
# import math
# import numpy as np
#
# x = [i * 0.001 for i in range(1000000)]
# start = time.clock()
# for i, t in enumerate(x):
#     x[i] = math.sin(t)
#
# print("math.sin:", time.clock() - start)
#
# x = [i * 0.001 for i in range(1000000)]
# x = np.array(x)
# start = time.clock()
# np.sin(x, x)
#
# print("numpy.sin: ", time.clock() - start)

import numpy as np

# a = np.arange(12).reshape(4, 3)
# b = np.arange(12, 24).reshape(3, 4)
# c = np.dot(a, b)
#
# print(c)

# a = np.random.rand(10, 10)
# b = np.random.rand(10)
# x = np.linalg.solve(a, b)
# print(np.sum(np.abs(np.dot(a, x)-b)))

a = np.arange(0, 12, 0.5).reshape(4, -1)
# np.savetxt("a.txt", a)  # 默认按照 %.18e 格式保存数据，以空格分隔
#
# print(np.loadtxt("a.txt"))

np.savetxt("a.txt", a, fmt="%d", delimiter=",")  # 改为保存为整数，以逗号分隔

print(np.loadtxt("a.txt", delimiter=","))