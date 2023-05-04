# @Version: python3.10
# @Time: 2023/5/4 14:40
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: words_pickle_open.py
# @Software: PyCharm
# @User: chent

import sys
sys.getdefaultencoding()
import pickle
import numpy as np
import pickle

pickle_data_path = "D:\ProgramingCodes\PycharmProjects\EnglishPal\\app\static\words_and_tests.p"
file = open(pickle_data_path, 'rb')   # pickle_data_path为.pickle文件的路径；
info = pickle.load(file)
print(info)
info = str(info)

obj_path = 'D:\ProgramingCodes\PycharmProjects\EnglishPal\\app\static\\res.txt'
ft = open(obj_path, 'w')
ft.write(info)

file.close()  # 别忘记close pickle文件


# np.set_printoptions(threshold=1000000000000000)
