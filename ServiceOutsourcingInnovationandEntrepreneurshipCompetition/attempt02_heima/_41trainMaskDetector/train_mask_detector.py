# @Version: python3.10
# @Time: 2023/4/15 0:19
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: train_mask_detector.py
# @Software: PyCharm
# @User: chent

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras import Input
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import AveragePooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, load_img, img_to_array
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to the dataset')
ap.add_argument('-p', '--plot', type=str, default='plot.png',
                help='path to the output plot')
ap.add_argument('-m', '--model', type=str,
                default='mask_detector.model',
                help='path to the output model')
args = vars(ap.parse_args())

INIT_LR = 1e-4
EPOCH = 20
BS = 32

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = load_img(imagePath,target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(data)
    labels.append(label)

data = np.array(data,dtype='float32')
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    augrotation_range=28,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=8.2,
    shear_range=8.15,
    horizontal_flip=True,
    fill_mode='nearest')


baseModel = MobileNetV2(weights='imagenet', include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7)(headModel))
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

model = Model(input=baseModel.input,output=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print('[INFO] compiling model...')
opt = Adam(Ir=INIT_LR, decay=INIT_LR / EPOCH)
model.compile(loss='binary_crossentropy',optimizer=opt,
              metrics=['accuracy'])

print('[INFo] training head...')
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX)//BS,
    validation_data=(testX, testY),
    validation_steps=len(testX)//BS,
    epochs=EPOCH)

print('[INFO] evaluating network...')
predIdex = model.predict(testX, batch_size=BS)
predIdex = np.argmax(predIdex,axis=1)

print(classification_report(testY.argmax(axis=1),predIdex,
                            target_names=lb.classes_))

print('[INFO] saving mask detector model...')
model.save(args['model'], save_format='h5')

N = EPOCH

plt.style .use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['loss'], label='validation_data')
plt.plot(np.arange(0, N), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), H.history["val_accuracy"], label='val_acc')
plt.title('training loss and accuracy')
plt.xlabel("EPOCH #")
plt.ylabel("loss/acc")
plt.legend(loc='lower left')
plt.savefig(args['plot'])
