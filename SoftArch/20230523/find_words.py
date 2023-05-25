# @Version: python3.10
# @Time: 2023/5/23 19:03
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: find_words.py
# @Software: PyCharm
# @User: chent

import pickle
import snowballstemmer

def load_pickle_file(pickle_name):
    """
    :param pickle_fname:    保存了35511万个单词的pickle文件
    :return:    读取单词保存在一个字典 d 中
    """
    f = open(pickle_name, 'rb')
    d = pickle.load(f)
    f.close()
    return d


def convert_test_type_to_difficulty_level(d):
    """
    对原本的单词库中的单词进行难度评级
    :param d: 存储了单词库pickle文件中的单词的字典
    :return:
    """
    result = {}
    L = list(d.keys())  # in d, we have test types (e.g., CET4,CET6,BBC) for each word

    for k in L:
        if 'CET4' in d[k]:
            result[k] = 4  # CET4 word has level 4
        elif 'OXFORD3000' in d[k]:
            result[k] = 5
        elif 'CET6' in d[k] or 'GRADUATE' in d[k]:
            result[k] = 6
        elif 'OXFORD5000' in d[k] or 'IELTS' in d[k]:
            result[k] = 7
        elif 'BBC' in d[k]:
            result[k] = 8

    return result  # {'apple': 4, ...}


if __name__ == '__main__':
    d = load_pickle_file('words_and_tests.p')

    # d2 =
    print(d['easy'])