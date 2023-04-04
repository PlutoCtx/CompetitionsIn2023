# @Version: python3.10
# @Time: 2023/2/28 17:31
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: kmeans.py.py
# @Software: PyCharm
# @User: chent


# @Version: python3.10
# @Time: 2023/2/28 17:14
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: kmeans.py.py
# @Software: PyCharm
# @User: chent




# import jieba
# import wordcloud
# # 读取文本
# with open("comments.txt", encoding="utf-8") as f:
#     s = f.read()
# print(s)
# ls = jieba.lcut(s) # 生成分词列表
# text = ' '.join(ls) # 连接成字符串
#
#
# stopwords = ["的","是","了"] # 去掉不需要显示的词
#
# wc = wordcloud.WordCloud(font_path="msyh.ttc",
#                          width = 1000,
#                          height = 700,
#                          background_color='white',
#                          max_words=100,stopwords=s)
# # msyh.ttc电脑本地字体，写可以写成绝对路径
# wc.generate(text) # 加载词云文本
# wc.to_file("pic.png") # 保存词云文件




'''
    词云有形状
'''
# 示例代码
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba
import pandas as pd

# x, y = pd.read_excel('data.xlsx')
# print(x)
# print(y)


# 将列表存储为txt格式文件
# def list2txt(list):
#     file = open('k.txt', 'w', encoding="utf-8")
#     for l in list:
#         l = str(l)  # 强制转换
#         if l[-1] != '\n':
#             l = l + '\n'
#         file.write(l)
#     file.close()
#     print(f"文件存储成功")



# df = pd.read_excel(r'data.xlsx')
#
# desc = df.loc[:, ['描述']].values
# eif = df.loc[:, ['频率']].values
#
# # print(desc)
# # print(eif)
# lis = []
# for i in range(len(desc)):
#     for j in range(int(eif[i])):
#         x = desc[i] 		# * int(eif[i])
#         lis.append(x)
# print(lis)

# list2txt(list)










# 打开文本
with open("k.txt", encoding="utf-8") as f:
    s = f.read()

# 中文分词
text = ' '.join(jieba.cut(s))

# 生成对象
img = Image.open("flower01.jpg")  # 打开遮罩图片
mask = np.array(img) #将图片转换为数组

stopwords = ["当", "寄", "春", '啊', '无', '1', '好', 'w', '。', "我","你","她","的","是","了","在","也","和","就","都","这","要",'年', '月','日','上','届']
wc = WordCloud(font_path="msyh.ttc",
               mask=mask,
               width=1000,
               height=700,
               background_color='white',
               max_words=200,
               relative_scaling=0.001,
               stopwords=stopwords,
                contour_width=3,
                contour_color='steelblue',
               scale=4).generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')# 用plt显示图片
plt.axis("off")  # 不显示坐标轴
plt.show() # 显示图片

# 保存到文件
wc.to_file("flower01_cover.jpg")

