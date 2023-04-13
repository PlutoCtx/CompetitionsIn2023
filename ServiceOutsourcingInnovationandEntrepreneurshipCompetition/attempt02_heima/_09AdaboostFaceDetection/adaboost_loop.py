# @Version: python3.10
# @Time: 2023/4/13 20:06
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: adaboost_loop.py
# @Software: PyCharm
# @User: chent

import cv2
import os

datapath = 'data/'

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load('haarcascade_frontalface_alt2.xml')   # 一定要告诉编译器文件所在的位置

for img in os.listdir(datapath):
    frame = cv2.imread(datapath + img)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图片转化成灰度
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # cv2.imshow('img', img)
    cv2.imwrite('result/' + img, frame)
    # cv2.waitKey()
