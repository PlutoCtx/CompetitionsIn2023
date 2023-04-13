# @Version: python3.10
# @Time: 2023/4/13 20:05
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: adaboost.py
# @Software: PyCharm
# @User: chent

import cv2
import os

img = cv2.imread('Aaron_Eckhart.jpg')       # 读取一张图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 将图片转化成灰度
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load('haarcascade_frontalface_alt2.xml')   # 一定要告诉编译器文件所在的位置

'''此文件是opencv的haar人脸特征分类器'''
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    # 图片， 左上角坐标， 右下角坐标， 边框线的颜色， 边框线的宽度
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('img', img)
    # cv2.imwirte('result/' + img,img)
    cv2.waitKey()