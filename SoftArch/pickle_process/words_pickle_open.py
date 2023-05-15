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
import snowballstemmer
import math


def load_record(pickle_fname):
    """
    :param pickle_fname:    保存了3/8万个单词的pickle文件
    :return:    读取单词保存在一个字典 d 中
    """
    f = open(pickle_fname, 'rb')
    d = pickle.load(f)
    f.close()
    return d


def get_difficulty_level_for_words_and_tests(dic):
    """
    对原本的单词库中的单词进行难度评级
    :param dic: 存储了单词库pickle文件中的单词的字典
    :return:
    """
    # T1 = time.clock()
    d = {}
    L = list(dic.keys())  # in dic, we have test types (e.g., CET4,CET6,BBC) for each word
    # i = 0
    for k in L:
        if 'CET4' in dic[k]:
            d[k] = [4]  # CET4 word has level 4
        elif 'CET6' in dic[k]:
            d[k] = [6]
        elif 'BBC' in dic[k]:
            d[k] = [8]
        # print(i, k, d[k])
        # i += 1
    # T2 = time.clock()
    # print('给27000个单词评级所需时间:%s毫秒' % ((T2 - T1) * 1000))  # 196.46390000000002毫秒
    # print('get_difficulty_level_for_words_and_tests')
    return d  # {'apple': 4, ...}


def difficulty_level_from_frequency(word, d):
    """
    根据单词的频率进行难度的评级
    :param word:
    :param d:
    :return:
    """
    level = 1
    if not word in d:
        return level

    if 'what' in d:
        ratio = (d['what'] + 1) / (d[word] + 1)  # what is a frequent word
        level = math.log(max(ratio, 1), 2)

    level = min(level, 8)
    return level


def simplify_the_words_dict(dic):

    stem = snowballstemmer.stemmer('english')

    res = {}
    for j in dic:   # j 在字典中
        temp = stem.stemWord(j)     # 提取j得词根
        if not temp in res:         # 如果这个词根不在结果字典中，则以词根为键、以dic中的等级作为值添加
            res[temp] = dic[j]
        else:                   # 如果这个词在结果词典中，则比较一下单词的难度等级是否最小
            if res[temp] > dic[j]:
                res[temp] = dic[j]

    return res

if __name__ == '__main__':

    # d01 = load_record('words_and_tests01.p')  # 导入27000个已标记单词
    # d02 = load_record('words_and_tests01.p')  # 导入27000个已标记单词
    #
    # print(d01)
    # d03 = load_record('frequency_qwertyuiop.p')  # 用户的单词，可能包括了不规则变形
    # d04 = load_record('frequency_qwertyuiop.p')  # 用户的单词，可能包括了不规则变形
    #
    # d01 = get_difficulty_level_for_words_and_tests(d01)  # 根据标记评级
    #
    # d_res = simplify_the_words_dict(d01)
    #
    #
    # print('新的单词评级方式，对用户的单词进行评级，得到一个res01')
    # """
    #     新的单词评级方式，对用户的单词进行评级，得到一个res01{'apples': 4, 'abandon': 4, ...}
    # """
    # stem = snowballstemmer.stemmer('english')
    # res01 = {}
    # for k in d03:           # k是用户不会的词
    #     for l in d_res:       # l是已经完成评级的词库的词
    #         if stem.stemWord(k) == l:
    #             res01[k] = d_res[l]
    #             print(k, l, res01[k])
    #             break
    #         else:
    #             res01[k] = difficulty_level_from_frequency(k, d03)
    #
    #     print('test', k, res01[k])
    #     print()
    #
    #
    d = load_record('pickle/words_an_tests.p')


