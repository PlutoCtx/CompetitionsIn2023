# @Version: python3.10
# @Time: 2023/5/11 10:30
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: txt_to_pickle.py
# @Software: PyCharm
# @User: chent

import pickle
from itertools import chain

def load_record(pickle_fname):
    f = open(pickle_fname, 'rb')
    d = pickle.load(f)
    f.close()
    return d

def txt_to_pickle(path, tag):
    f = open(path, encoding='utf-8', errors='ignore')
    d = {}
    for line in f:
        if line.strip() != "":
            temp = line.strip().split()
            d[temp[0]] = [tag]

    file = open(tag + '.p', 'wb')
    pickle.dump(d, file)
    file.close()

    print("success")

def merge_two_dict(d1, d2):
    result = {}
    '''
    核心思路：
            1：遍历字典1和字典2的每一个键
            2：如果两个字典的键是一样的，就给新字典的该键赋值为空列表
                然后空列表依次添加字典1和字典2 的值，然后将最后的值赋值给原字典
            3：如果两个字典的键不同，则分别将键值对加到新列表中
    '''
    for k, v in d1.items():
        for m, n in d2.items():
            if k == m:
                temp = d1[k] + d2[m]
                result[k] = temp
            else:
                result[k] = d1[k]
                result[m] = d2[m]
    return result

def merge(d1, d2):
    for k in d2:
        if k in d1:
            temp = d1[k] + d2[k]
            d1[k] = temp
        else:
            d1[k] = d2[k]
    return d1

def merge_EnWords(d1, d2):
    for k in d2:
        if not k in d1:
            d1[k] = d2[k]
    return d1


# def txt_to_pickle02(path):
#     f = open(path, encoding='utf-8', errors='ignore')
#     d = {}
#     for line in f:
#         if line.strip() != "":
#             temp = line.strip().split()
#             d[temp[0]] = [tag]
#
#     file = open(tag + '.p', 'wb')
#     pickle.dump(d, file)
#     file.close()

    # print("success")


if __name__ == '__main__':

    # f = open("words_and_test_enlarged.txt", encoding='utf-8', errors='ignore')
    # d = {}
    # for line in f:
    #     if line.strip() != '':
    #         temp = line.strip().split()
    #         d[temp[0]] = [temp]

    # d_CET4 = load_record('CET4.p')
    # d_CET6 = load_record('CET6.p')
    # d_GRADUATE = load_record('GRADUATE.p')
    # d_IETLS = load_record('IETLS.p')
    # d_EnWords = load_record('EnWords.p')

    # result1 = merge(d_CET4, d_CET6)
    # result2 = merge(result1, d_GRADUATE)
    # result = merge(result2, d_IETLS)
    #
    # result = merge_EnWords(result, d_EnWords)
    #
    # file = open('words_and_tests01.p', 'wb')
    # pickle.dump(result, file)
    # file.close()
    # d = load_record('words_ad_tests.p')
    # i = 0
    # for k in d:
    #     if d[k] == ['EnWords'] and i < 15000:
    #         # d[k] = ['OTHER']
    #         print(i, k, d[k])
    #         i += 1

    # file = open('words_and_tests01.p', 'wb')
    # pickle.dump(d, file)
    # file.close()

    d = load_record('')





