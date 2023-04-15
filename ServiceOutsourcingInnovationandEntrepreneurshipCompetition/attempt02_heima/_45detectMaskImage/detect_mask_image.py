# @Version: python3.10
# @Time: 2023/4/15 1:06
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: detect_mask_image.py
# @Software: PyCharm
# @User: chent

import argparse
import os

import cv2
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to the image')
ap.add_argument('-f', '--face', type=str,
                default='face_detector')
ap.add_argument('-m', '--model', type=str,
                default='mask_detector.model',
                help='path to the model')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
                help='minimum probability to filter weak detections')
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args['face'],"deploy.prototxt"])
weightsPath = os.path.sep.join([args['face'],
                                "res10_300x30_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

print('[INFO] load face mask detector...')
model = load_model(args['model'])