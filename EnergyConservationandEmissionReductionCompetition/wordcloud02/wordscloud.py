# @Version: python3.10
# @Time: 2023/3/29 17:14
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: wordscloud.py
# @Software: PyCharm
# @User: chent


import matplotlib.pyplot as plt  # 数据可视化
import jieba  # 词语切割
import wordcloud  # 分词
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS  # 词云，颜色生成器，停止词
import numpy as np  # 科学计算
from PIL import Image  # 处理图片


def ciyun():
	with open('comment02.txt', 'r', encoding='utf-8') as f:  # 打开新的文本转码为gbk
		textfile = f.read()  # 读取文本内容
	wordlist = jieba.lcut(textfile)  # 切割词语
	space_list = ' '.join(wordlist)  # 空格链接词语
	# print(space_list)
	backgroud = np.array(Image.open('CLIMATE.png'))

	wc = WordCloud(width=1400, height=2200,
				   background_color='white',
				   mode='RGB',
				   mask=backgroud,  # 添加蒙版，生成指定形状的词云，并且词云图的颜色可从蒙版里提取
				   max_words=500,
				   stopwords=STOPWORDS.add('老年人'),  # 内置的屏蔽词,并添加自己设置的词语
				   font_path='C:\Windows\Fonts\STZHONGS.ttf',
				   max_font_size=150,
				   relative_scaling=0.6,  # 设置字体大小与词频的关联程度为0.4
				   random_state=50,
				   scale=2
				   ).generate(space_list)

	image_color = ImageColorGenerator(backgroud)  # 设置生成词云的颜色，如去掉这两行则字体为默认颜色
	wc.recolor(color_func=image_color)

	plt.imshow(wc)  # 显示词云
	plt.axis('off')  # 关闭x,y轴
	plt.show()  # 显示
	wc.to_file('test1_ciyun.jpg')  # 保存词云图


def main():
	ciyun()


if __name__ == '__main__':
	main()





#
# import jieba
# import stylecloud
# def ciyun():
#     with open('comment02.txt','r',encoding='utf-8') as f:
#         word_list = jieba.cut(f.read())
#         result = " ".join(word_list) #分词用空格隔开
#     stylecloud.gen_stylecloud(
#         text=result, # 上面分词的结果作为文本传给text参数
#         size=512,
#         font_path='msyh.ttc', # 字体设置
#         palette='cartocolors.qualitative.Pastel_7', # 调色方案选取，从palettable里选择
#         gradient='horizontal', # 渐变色方向选了垂直方向
#         icon_name='fab fa-weixin',  # 蒙版选取，从Font Awesome里选
#         output_name='test_ciyun.png') # 输出词云图
#
# def main():
#     ciyun()
#
# if __name__ == '__main__':
#     main()
