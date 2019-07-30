# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:45:49 2019

@author: Mayank Jain
"""
import os
import tensorflow as tf
import cv2
import numpy as np
from keras.utils import np_utils
from keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
#from cleverhans_utils_keras import cnn_model, KerasModelWrapper
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
NB_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = .0000001
RESULTS_DIR = 'VGG16 - Data Random - LR 1e-7/'
LOAD_MODEL = False

# Define TF model graph
#Instantiate an empty model
model = Sequential([
                    Conv2D(64, (3, 3), input_shape=(img_rows, img_cols, nchannels), padding='same', activation='relu'),
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
                    Dense(nb_classes, activation='softmax')
                    ])

opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
'''
'''
#train the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=NB_EPOCHS, callbacks=[es])

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