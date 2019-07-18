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
from cleverhans_utils_keras import cnn_model, KerasModelWrapper
from tensorflow.keras.callbacks import EarlyStopping

'''import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
'''
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
  if maxImg == None:
    images = np.empty((len(os.listdir(folder)), img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
  else:
    images = np.empty((maxImg, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
  # Iterate over all filenames to fill in the images array
  for i, filename in enumerate(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
      images[i, ...] = img
    if maxImg is not None:
      if i>=maxImg-1:
        break
  # Return images
  return images

'''
CREATE Training and Test Data
'''
# Get Train Dataset
turnRight = load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/33', 8300)
print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/34', 8300)
print("shape of original turnLeft", turnLeft.shape)
goStraight = load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/35', 8300)
print("shape of original goStraight", goStraight.shape)

x_train = np.concatenate((turnLeft, turnRight, goStraight))
y_train = np.zeros((x_train.shape[0], 3))
y_train[0:turnLeft.shape[0], 0] = 1
y_train[turnLeft.shape[0]:turnRight.shape[0], 1] = 1
y_train[turnRight.shape[0]:, 2] = 1

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
y_test[turnLeft.shape[0]:turnRight.shape[0], 1] = 1
y_test[turnRight.shape[0]:, 2] = 1
shuffle_in_unison(x_test, y_test)

'''
CREATE Model
'''

# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

FLAGS = flags.FLAGS

TRAIN_FRAC = 0.85
NB_EPOCHS = 75
BATCH_SIZE = 128
LEARNING_RATE = .0000001
TRAIN_DIR = 'train_dir'
FILENAME = 'turnDirTest.ckpt'
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
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=NB_EPOCHS, callbacks=[es])

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
model.save("model.h5")
print("Saved model to disk")