# @Version: python3.10
# @Time: 2023/5/12 15:34
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: pickle.py
# @Software: PyCharm
# @User: chent

import pickle

def load_record(pickle_fname):
    """
    :param pickle_fname:    保存了3/8万个单词的pickle文件
    :return:    读取单词保存在一个字典 d 中
    """
    f = open(pickle_fname, 'rb')
    d = pickle.load(f)
    f.close()
    return d

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


if __name__ == '__main__':
    # d1 = load_record('words_and_tests (1).p')      # 原本的
    # d2 = load_record('words_and_tests.p')            # 打不开的原本的
    #
    # d3 = merge(d1, d2)


    # i = 0
    # for k in d3:
    #     if not str.isalpha(k):
    #         d3.pop(k)
    #
    #     # if 'BBC' in d3[k]:
    #     #     print(i, k, d3[k])
    #     #     i += 1
    #
    # file = open('words_final.p', 'wb')
    # pickle.dump(d3, file)
    # file.close()
    # print(len(d3))
    d = load_record('words_and_tests_enlarged__cleansing2.p')
    # if '***' in d:
    #     d.pop('***')
    #
    # if '英语四级单词表（H）' in d:
    #     d.pop('英语四级单词表（H）')
    #     print('h')
    #
    # if '英语四级单词表（J、K）' in d:
    #     d.pop('英语四级单词表（J、K）')
    #     print('jk')
    #
    # if '语四级单词表（L）' in d:
    #     print('语四级单词表（L）')
    #     d.pop('语四级单词表（L）')
    for k in d:
        d[k] = list(set(d[k]))

    file = open('words_and_tests_enlarged__cleansing.p', 'wb')
    pickle.dump(d, file)
    file.close()

    # i = 0
    # for k in d:
    #     if i < 20000:
    #         # print(i, k, d[k])
    #         i += 1
    #     elif i < 40000:
    #         print(i, k, d[k])
    #         i += 1
    #     elif i < 60000:
    #         # print(i, k, d[k])
    #         i += 1
    #     elif i < 80000:
    #         # print(i, k, d[k])
    #         i += 1
    #     elif i < 100000:
    #         # print(i, k, d[k])
    #         i += 1
    #     elif i < 120000:
    #         # print(i, k, d[k])
    #         i += 1

