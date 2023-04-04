# @Version: python3.10
# @Time: 2023/3/29 17:33
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: wordcloud02.py
# @Software: PyCharm
# @User: chent


import jieba  # 分词
from wordcloud import WordCloud  # 词云图相关
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # 处理图片相关内容
import numpy
from PIL import Image

import jieba
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def trans_ch(txt):
  words = jieba.lcut(txt)
  newtxt = ''.join(words)
  return newtxt


f = open('comment02.txt', 'r', encoding='utf-8')     # 将你的文本文件名与此句的'maozedong.txt'替换
txt = f.read()
f.close
txt = trans_ch(txt)
mask = np.array(Image.open("CLIMATE.png"))               # 将你的背景图片名与此句的"love.png"替换
print('mask')
colormaps = colors.ListedColormap(['#62bbb3'])
print('colormaps')
wordcloud = WordCloud(background_color="white",
                        width=800,
                        height=600,
                        max_words=200,
                        max_font_size=80,
                        mask=mask,
                        contour_width=4,
                        contour_color='steelblue',
                        font_path="msyh.ttf",
                        colormap=colormaps
                      ).generate(txt)
wordcloud.to_file('CLIMATE_COVER.png')

