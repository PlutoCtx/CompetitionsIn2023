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
import jieba
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 示例代码
from wordcloud import WordCloud

# 打开文本
with open("wordcloud/comments.txt", encoding="utf-8") as f:
    s = f.read()

# 中文分词
text = ' '.join(jieba.cut(s))

# 生成对象
img = Image.open("wordcloud/picc.png") # 打开遮罩图片
mask = np.array(img) #将图片转换为数组

stopwords = ["我","你","她","的","是","了","在","也","和","就","都","这"]
wc = WordCloud(font_path="msyh.ttc",
               mask=mask,
               width = 1000,
               height = 700,
               background_color='white',
               max_words=200,
               stopwords=stopwords).generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')# 用plt显示图片
plt.axis("off")  # 不显示坐标轴
plt.show() # 显示图片

# 保存到文件
wc.to_file("李焕英2.png")