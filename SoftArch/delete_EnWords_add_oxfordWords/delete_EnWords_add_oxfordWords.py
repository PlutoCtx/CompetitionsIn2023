# @Version: python3.10
# @Time: 2023/5/18 16:15
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: delete_EnWords_add_oxfordWords.py
# @Software: PyCharm
# @User: chent

import pickle
import snowballstemmer

def load_pickle_file(pickle_name):
    """
    :param pickle_fname:    保存了3/8万个单词的pickle文件
    :return:    读取单词保存在一个字典 d 中
    """
    f = open(pickle_name, 'rb')
    d = pickle.load(f)
    f.close()
    return d


if __name__ == '__main__':
    # 这个词库内包含了单词的词根，所以不需要对单词进行词根还原
    # 但其实有537个单词的词根不在这个词库里，大多数都是BBC的
    d = load_pickle_file('my_words_and_tests.pickle')
    stem = snowballstemmer.stemmer('english')

    print(len(d))

    i = 0
    for k in d:
        if stem.stemWord(k) not in d:
            print(i, k, d[k])
            i += 1
