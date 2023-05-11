# @Version: python3.10
# @Time: 2023/5/11 11:16
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: xml_read_and_write.py
# @Software: PyCharm
# @User: chent

import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
import pickle

def load_record(pickle_fname):
    f = open(pickle_fname, 'rb')
    d = pickle.load(f)
    f.close()
    return d
def generate_pickle_for_graduete(path='files/undergraduate.xml'):
    dom = minidom.parse(path)
    root = dom.documentElement
    names = root.getElementsByTagName('name')
    d = {}
    for name in names:
        # 它的第一个子节点是一个textnode，存取的是真正的节点值
        value = name.childNodes[0].nodeValue
        value = value.strip()
        d[value] = ['GRADUATE']

    file = open('GRADUATE.p', 'wb')
    pickle.dump(d, file)
    file.close()

    print("success")

    d1 = load_record('GRADUATE.p')
    print(d1)

def generate_pickle_for_cet46_toff_yasi(path='files/CET46_TOFF_AISI.xml'):

    # 读取文件
    dom = minidom.parse(path)
    # 获取文档元素对象
    elem = dom.documentElement
    # 获取 student
    items = elem.getElementsByTagName('item')
    for item in items:
        # 获取标签中内容
        word = item.getElementsByTagName('word')[0].childNodes[0].nodeValue
        tags = item.getElementsByTagName('tags')[0].childNodes[0].nodeValue
        # age = stu.getElementsByTagName('progress')[0].childNodes[0].nodeValue
        print('word:', word, ', tags:', tags)



    # # 使用minidom解析器打开 XML 文档
    # DOMTree = minidom.parse(path)
    # collection = DOMTree.documentElement
    # # if collection.hasAttribute("shelf"):
    # #     print
    # #     "Root element : %s" % collection.getAttribute("shelf")
    #
    # # 在集合中获取所有电影
    # items = collection.getElementsByTagName("item")
    #
    # # 打印每部电影的详细信息
    # for item in items:
    #     word = item.getElementsByTagName('word')
    #     tag = item.getElementsByTagName('tag')
    #     print(word, tag)




        # print
        # "*****Movie*****"
        # if movie.hasAttribute("title"):
        #     print
        #     "Title: %s" % movie.getAttribute("title")
        #
        # type = movie.getElementsByTagName('type')[0]
        # print
        # "Type: %s" % type.childNodes[0].data
        # format = movie.getElementsByTagName('format')[0]
        # print
        # "Format: %s" % format.childNodes[0].data
        # rating = movie.getElementsByTagName('rating')[0]
        # print
        # "Rating: %s" % rating.childNodes[0].data
        # description = movie.getElementsByTagName('description')[0]
        # print
        # "Description: %s" % description.childNodes[0].data






    # tree = ET.parse(path)
    # items = tree.findall('/item')
    #
    # for item in items:
    #     for child in item.getchildren():
    #         print(child['word'].tag, child.text)



    # root = tree.getroot()
    # items = root.findall('item')
    # # print(items)
    # print(items[0].tag)
    # # print(items.find('word').text)


    # for item in items:
    #     tag = item.find('tag')
    #     word = item.find('word')
        # print(word[1], tag[1])






    # dom = minidom.parse(path)
    # root = dom.documentElement
    # # names = root.getElementsByTagName('word')
    # items = root.getElementsByTagName('item')
    # i = 0
    # d = {}
    # for item in items:
    #     # # 它的第一个子节点是一个textnode，存取的是真正的节点值
    #     # value = name.childNodes[0].nodeValue
    #     # value = value.strip()
    #     # d[value] = 'GRADUATE'
    #     word = item.childNodes[0].nodeValue
    #     tag = item.childNodes[3].nodeValue
    #     print(word, ', ', tag)

    # file = open('GRADUATE.p', 'wb')
    # pickle.dump(d, file)
    # file.close()
    #
    # print("success")
    #
    # d1 = load_record('GRADUATE.p')
    # print(d1)
    # coding=utf-8


    # 打开xml文档
    # dom = minidom.parse(path)   # 用于打开一个xml文件，并将这个文件对象dom变量

    # # 得到文档元素对象
    # root = dom.documentElement      # 用于得到dom对象的文档元素，并把获得的对象给root
    # print(root.nodeName)        # 结点名字
    # print(root.nodeValue)       # 结点的值，只对文本结点有效
    # print(root.nodeType)        # 结点的类型
    # print(root.ELEMENT_NODE)    # ELEMENT_NODE类型


if __name__ == '__main__':
    generate_pickle_for_graduete()
    # generate_pickle_for_cet46_toff_yasi()
