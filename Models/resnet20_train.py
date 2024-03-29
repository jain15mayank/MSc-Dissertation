# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:26:42 2019

@author: Mayank Jain
"""

import os
import tensorflow as tf
import cv2
import numpy as np
import keras
from keras.utils import np_utils
from keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

# Assignment rather than import because direct import from within Keras
# doesn't work in tf 1.8
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
KerasModel = tf.keras.models.Model
BatchNormalization = tf.keras.layers.BatchNormalization
AveragePooling2D = tf.keras.layers.AveragePooling2D
Input = tf.keras.layers.Input
l2 = tf.keras.regularizers.l2

'''
LOAD DATA
'''
def load_images_from_folder(folder, maxImg=None):
  # Load 1 file to check image shape
  i = 0
  for filename in os.listdir(folder):
    if i==0:
      img = cv2.imread(os.path.join(folder,filename))
      i+=1
    else:
      break
  # Create empty numpy array for all the images to come
  images = np.empty((len(os.listdir(folder)), img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
  
  # Iterate over all filenames to fill in the images array
  for i, filename in enumerate(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
      images[i, ...] = img
  if maxImg is not None:
    maxImages = np.empty((maxImg, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
    indices = np.random.permutation(len(os.listdir(folder)))
    i = 0
    for idx in indices:
      maxImages[i, ...] = images[idx]
      i+=1
      if i>=maxImg:
        break
    return maxImages
  else:
    return images

'''
AUGMENT data further with multiple observations
'''
def makeObservations(imgList, mode='multi', farScale = 0.5, obliquePercentage = 10, obliqueDirection = 'left'):
    """
    Changes all the images in the provided list according to the given parameters.
        imgList: [ndarray]
            List of all images to be altered; All images must be of same dimensions
        mode: string ('single' or 'multi')
            If 'multi': ignores other parameters and generate 28 new observations for
                        each image. 7 different oblique patterns each at 4 different
                        distances (including original image too)
            If 'single': create 1 observation for each image in the list according to
                         the specified parameters
        farScale: scalar (float) (0-1]
            Parameter defining the closeness of the new image (1=same, ~0=farthest)
        obliquePercentage: scalar (int) (0-45)
            Defines how much the image is to be tilted
        obliqueDirection: string ('left' or 'right')
            Specifies the direction of tilt
    """
    if (len(imgList.shape)==3): #Only one image is provided
        numImgs = 1
        imgRows = imgList.shape[0]
        imgCols = imgList.shape[1]
        nChannels = imgList.shape[2]
        imgList = [imgList]
    else:
        numImgs = imgList.shape[0]
        imgRows = imgList.shape[1]
        imgCols = imgList.shape[2]
        nChannels = imgList.shape[3]
    if mode=='multi':
        obliquePercentages = [20, 30]
        farScales = [1, 0.25]
        newImgList = np.zeros((numImgs*((2*len(obliquePercentages))+1)*len(farScales), imgRows, imgCols, nChannels))
        i = 0
        for img in imgList:
            for fS in farScales:
                img1 = cv2.resize(img, (int(imgRows*fS), int(imgCols*fS)))
                img1 = cv2.resize(img1, (imgRows, imgCols))
                img1 = cv2.GaussianBlur(img1, (5,5), 0)
                newImgList[i, ...] = img1
                i+=1
            for oP in obliquePercentages:
                for fS in farScales:
                    #For Left Oblique
                    XobliquePixels = oP*imgRows/100
                    YobliquePixels = 0.75*XobliquePixels
                    src = np.array([
                            [0, 0],
                            [imgRows - 1, 0],
                            [imgRows - 1, imgCols - 1],
                            [0, imgCols - 1]
                            ], dtype = "float32")
                    dst = np.array([
                            [XobliquePixels, YobliquePixels],
                            [imgRows - XobliquePixels - 1, 0],
                            [imgRows - XobliquePixels - 1, imgCols - 1],
                            [XobliquePixels, imgCols - YobliquePixels - 1]
                            ], dtype = "float32")
                    # compute the perspective transform matrix and then apply it
                    M = cv2.getPerspectiveTransform(src, dst)
                    warped = cv2.warpPerspective(img, M, (imgRows, imgCols))
                    #warped = warped[:, int(XobliquePixels):int(imgRows - XobliquePixels - 1)]
                    warped = warped[int(oP*YobliquePixels/100):int(imgCols - (oP*YobliquePixels/100) - 1), 
                                int(XobliquePixels):int(imgRows - XobliquePixels - 1)]
                    warped = cv2.resize(warped, (imgRows, imgCols))
                    img1 = cv2.resize(warped, (int(imgRows*fS), int(imgCols*fS)))
                    img1 = cv2.resize(img1, (imgRows, imgCols))
                    img1 = cv2.GaussianBlur(img1, (5,5), 0)
                    newImgList[i, ...] = img1
                    i+=1
                    #For Right Oblique
                    XobliquePixels = oP*imgRows/100
                    YobliquePixels = 0.75*XobliquePixels
                    src = np.array([
                            [0, 0],
                            [imgRows - 1, 0],
                            [imgRows - 1, imgCols - 1],
                            [0, imgCols - 1]
                            ], dtype = "float32")
                    dst = np.array([
                            [XobliquePixels, 0],
                            [imgRows - XobliquePixels - 1, YobliquePixels],
                            [imgRows - XobliquePixels - 1, imgCols - YobliquePixels - 1],
                            [XobliquePixels, imgCols - 1]
                            ], dtype = "float32")
                    # compute the perspective transform matrix and then apply it
                    M = cv2.getPerspectiveTransform(src, dst)
                    warped = cv2.warpPerspective(img, M, (imgRows, imgCols))
                    warped = warped[int(oP*YobliquePixels/100):int(imgCols - (oP*YobliquePixels/100) - 1), 
                                int(XobliquePixels):int(imgRows - XobliquePixels - 1)]
                    warped = cv2.resize(warped, (imgRows, imgCols))
                    img1 = cv2.resize(warped, (int(imgRows*fS), int(imgCols*fS)))
                    img1 = cv2.resize(img1, (imgRows, imgCols))
                    img1 = cv2.GaussianBlur(img1, (5,5), 0)
                    newImgList[i, ...] = img1
                    i+=1
    else:
        newImgList = np.zeros((numImgs, imgRows, imgCols, nChannels))
        i = 0
        for img in imgList:
            XobliquePixels = obliquePercentage*imgRows/100
            YobliquePixels = 0.75*XobliquePixels
            if obliqueDirection=='left' or obliqueDirection == 'Left':
                src = np.array([
                    [0, 0],
                    [imgRows - 1, 0],
                    [imgRows - 1, imgCols - 1],
                    [0, imgCols - 1]
                    ], dtype = "float32")
                dst = np.array([
                    [XobliquePixels, YobliquePixels],
                    [imgRows - XobliquePixels - 1, 0],
                    [imgRows - XobliquePixels - 1, imgCols - 1],
                    [XobliquePixels, imgCols - YobliquePixels - 1]
                    ], dtype = "float32")
            else:
                src = np.array([
                    [0, 0],
                    [imgRows - 1, 0],
                    [imgRows - 1, imgCols - 1],
                    [0, imgCols - 1]
                    ], dtype = "float32")
                dst = np.array([
                    [XobliquePixels, 0],
                    [imgRows - XobliquePixels - 1, YobliquePixels],
                    [imgRows - XobliquePixels - 1, imgCols - YobliquePixels - 1],
                    [XobliquePixels, imgCols - 1]
                    ], dtype = "float32")
            # compute the perspective transform matrix and then apply it
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, (imgRows, imgCols))
            warped = warped[int(obliquePercentage*YobliquePixels/100):int(imgCols - (obliquePercentage*YobliquePixels/100) - 1), 
                        int(XobliquePixels):int(imgRows - XobliquePixels - 1)]
            warped = cv2.resize(warped, (imgRows, imgCols))
            img1 = cv2.resize(warped, (int(imgRows*farScale), int(imgCols*farScale)))
            img1 = cv2.resize(img1, (imgRows, imgCols))
            img1 = cv2.GaussianBlur(img1, (5,5), 0)
            newImgList[i, ...] = img1
            i+=1
    return newImgList

'''
CREATE Training and Test Data
'''
# Get Train Dataset
turnRight = makeObservations(load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/33', 3000))
print("shape of original turnRight", turnRight.shape)
turnLeft = makeObservations(load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/34', 3000))
print("shape of original turnLeft", turnLeft.shape)
goStraight = makeObservations(load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/35', 3000))
print("shape of original goStraight", goStraight.shape)

x_train = np.concatenate((turnLeft, turnRight, goStraight))
y_train = np.zeros((x_train.shape[0], 3))
y_train[0:turnLeft.shape[0], 0] = 1
y_train[turnLeft.shape[0]:turnLeft.shape[0] + turnRight.shape[0], 1] = 1
y_train[turnLeft.shape[0] + turnRight.shape[0]:, 2] = 1

def shuffle_in_unison(a, b):
  rng_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(rng_state)
  np.random.shuffle(b)
shuffle_in_unison(x_train, y_train)

# Get Test Dataset
turnRight = load_images_from_folder('../GTSRB Dataset/categorized&cropped_NA - Test/33')
print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder('../GTSRB Dataset/categorized&cropped_NA - Test/34')
print("shape of original turnLeft", turnLeft.shape)
goStraight = load_images_from_folder('../GTSRB Dataset/categorized&cropped_NA - Test/35')
print("shape of original goStraight", goStraight.shape)

x_test = np.concatenate((turnLeft, turnRight, goStraight))
y_test = np.zeros((x_test.shape[0], 3))
y_test[0:turnLeft.shape[0], 0] = 1
y_test[turnLeft.shape[0]:turnLeft.shape[0]+turnRight.shape[0], 1] = 1
y_test[turnLeft.shape[0]+turnRight.shape[0]:, 2] = 1
shuffle_in_unison(x_test, y_test)

'''
CREATE Model
'''

# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

FLAGS = flags.FLAGS

TRAIN_FRAC = 0.85
NB_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = .0000005
RESULTS_DIR = 'ResNet20 Batch Max - Data Random - LR 1e-6/'
LOAD_MODEL = False

# Define TF model graph

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = KerasModel(inputs=inputs, outputs=outputs)
    return model

model = resnet_v1((img_rows, img_cols, nchannels), depth=20, num_classes=nb_classes)
opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
'''
'''
#train the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=NB_EPOCHS, callbacks=[es])

# Plot training & validation accuracy values
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
fig.savefig(RESULTS_DIR + 'acc_plot.png')

# Plot training & validation loss values
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
fig.savefig(RESULTS_DIR + 'loss_plot.png')

#predict first 4 images in the test set
print(model.predict(x_test[:4]))
#actual results for first 4 images in test set
print(y_test[:4])

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s Test: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(x_train, y_train, verbose=0)
print("%s Train: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_temp = np.zeros((turnRight.shape[0], 3))
y_temp[:,1] = 1
scores = model.evaluate(turnRight, y_temp, verbose=0)
print("%s Test (Turn Right): %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_temp = np.zeros((turnLeft.shape[0], 3))
y_temp[:,0] = 1
scores = model.evaluate(turnLeft, y_temp, verbose=0)
print("%s Test (Turn Left): %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_temp = np.zeros((goStraight.shape[0], 3))
y_temp[:,2] = 1
scores = model.evaluate(goStraight, y_temp, verbose=0)
print("%s Test (Go Straight): %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save(RESULTS_DIR + "model.h5")
print("Saved model to disk")