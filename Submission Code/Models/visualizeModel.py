# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 23:20:33 2019

@author: Mayank Jain
"""

import os
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from keras import backend
from keras.utils import plot_model

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

def createModel_VGG16(inputShape, outClasses):
    model = Sequential([
                    Conv2D(64, (3, 3), input_shape=inputShape, padding='same', activation='relu'),
                    Conv2D(64, (3, 3), activation='relu', padding='same'),
                    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    Conv2D(128, (3, 3), activation='relu', padding='same'),
                    Conv2D(128, (3, 3), activation='relu', padding='same',),
                    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    Conv2D(256, (3, 3), activation='relu', padding='same',),
                    Conv2D(256, (3, 3), activation='relu', padding='same',),
                    Conv2D(256, (3, 3), activation='relu', padding='same',),
                    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    Conv2D(512, (3, 3), activation='relu', padding='same',),
                    Conv2D(512, (3, 3), activation='relu', padding='same',),
                    Conv2D(512, (3, 3), activation='relu', padding='same',),
                    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    Conv2D(512, (3, 3), activation='relu', padding='same',),
                    Conv2D(512, (3, 3), activation='relu', padding='same',),
                    Conv2D(512, (3, 3), activation='relu', padding='same',),
                    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    Flatten(),
                    Dense(4096, activation='relu'),
                    Dense(4096, activation='relu'),
                    Dense(outClasses, activation='softmax')
                    ])
    return model

inputShape = (100,100,3)
outClasses = 3

model = createModel_VGG16(inputShape, outClasses)
model.summary()
plot_model(model, to_file='model.png')