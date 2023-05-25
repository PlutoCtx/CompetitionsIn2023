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










#
#
# '''
#     词云有形状
# '''
# import jieba
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import matplotlib.colors as colors  # 处理图片相关内容
# # 示例代码
# from wordcloud import WordCloud
#
# # 打开文本
# with open("01easy.txt", encoding="utf-8") as f:
#     s = f.read()
#
# # 中文分词
# text = ' '.join(jieba.cut(s))
#
# # 生成对象
# img = Image.open("01easy1.png") # 打开遮罩图片
# mask = np.array(img) #将图片转换为数组
#
# stopwords = ["我","你","她","的","是","了","在","也","和","就","都","这"]
# colormaps = colors.ListedColormap(['#FF0000','#FF7F50','#FFE4C4'])
# wc = WordCloud(font_path="msyh.ttc",
#                mask=mask,
#                width = 1000,
#                height = 700,
#
#                background_color='white',
#                max_words=200,
#                min_font_size=2,
#
#                stopwords=stopwords,
#                 colormap=colormaps
#                ).generate(text)
#
# # 显示词云
# plt.imshow(wc, interpolation='bilinear')# 用plt显示图片
# plt.axis("off")  # 不显示坐标轴
# plt.show() # 显示图片
#
# # 保存到文件
# wc.to_file("01easy2.png")









# import jieba  # 分词
# from wordcloud import WordCloud  # 词云图相关
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors  # 处理图片相关内容
# import numpy
# from PIL import Image
#
# # 生成词云方法(以庆余年小说为例)
# def get_wcloud():
# 	# 读取小说内容
#     with open(r'01easy.txt', 'r', encoding='utf-8') as txt:
#         data = txt.read()
#     # 加载词典
#     # jieba.load_userdict('01easy.txt')
#     # 也可以添加自定义词典
#     # jieba.add_word('范慎')
#     # jieba.add_word('范闲')
#     # 将文件中所有文字分词
#     words_list = jieba.lcut(data)
#     # 用空格分隔词语
#     tokenstr = ' '.join(words_list)
#     # 打开背景图片
#     color_mask = numpy.array(Image.open('01easy1.png'))
#     # 自定义文字颜色
#     colormaps = colors.ListedColormap(['#FF0000','#FF7F50','#FFE4C4'])
#     # 生成词云（默认样式）
#     # mywc1 = WordCloud().generate(tokenstr)
#     # 生成词云（自定义样式）
#     mywc1 = WordCloud(
#         mask=color_mask,  # 指定背景图形状
#         colormap=colormaps,  # 指定颜色
#         # font_path='C:/Windows/Fonts/simkai.ttf',  # 指定字体
#         background_color='white',  # 指定背景颜色，默认黑色
#         width=800,  # 指定宽度
#         height=600  # 指定高度
#     ).generate(tokenstr)
#     # 显示词云
#     plt.imshow(mywc1)
#     plt.axis('off')
#     plt.show()
#     mywc1.to_file('01easy2.png')  # 生成词云图片
#
# if __name__ == '__main__':
#     get_wcloud()











# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# mask = np.array(Image.open("01easy1.png"))
#
#
# f = open('01easy.txt','r',encoding = 'utf-8')
# txt = f.read()
# f.close
# wordcloud = WordCloud(background_color="white",
#                       width = 800,
#                       height = 600,
#                       max_words = 200,
#                       max_font_size = 80,
#                       mask = mask,
#                       contour_width = 3,
#                       contour_color = 'steelblue'
#                       ).generate(txt)
# wordcloud.inter_word_space = 1
# wordcloud.to_file('01easy2.png')






from wordcloud import WordCloud

# 读取文本文件
text = open("01easy.txt", encoding="utf-8").read()

# 配置词云图参数
wc = WordCloud(width = 800, height = 800,
               background_color ='white',
               min_font_size = 10)

# 将文本传入WordCloud对象
wc.generate(text)

# 调节词语间距
wc.inter_word_space = 1

# 调节字体大小和字体
wc.generate_from_frequencies(max_font_size=60,
                             font_path='STCAIYUN.TTF')

# 调节词频权重
frequencies = {'apple': 20, 'banana': 15, 'orange': 10}
wc.generate_from_frequencies(frequencies)

# 生成词云图并保存为图片
wc.to_file("wordcloud.png")
