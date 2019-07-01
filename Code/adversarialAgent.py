# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:16:21 2019

@author: Mayank Jain
"""

import os
import cv2
from copy import deepcopy
import numpy as np
from utils_mudSlap import addMudSplat

from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

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

class mudSplat:
    """
    Creates a mudSplat object specified by the following properties:
        imgPath: string
            Directory path to the mudSplat image
        xOffset: scalar (int)
            Offset value in X-Dimension where the splat should be placed
        yOffset: scalar (int)
            Offset value in Y-Dimension where the splat should be placed
        scale: scalar (float) (0-100)
            Specifies how much big the mud splat should be in reference to the
            size of original image (e.g.: if scaleParam==100: the splat will be
            almost equal to the size of original image); values are by default
            clipped between 0 and 100
        rotate: scalar (float) (0-360)
            Specifies how much to rotate the original image by (in degrees)
    """
    def __init__(self, imgPath, xOffset, yOffset, scale, rotate):
        self.imgPath = imgPath
        self.xOffset = xOffset
        self.yOffset = yOffset
        self.scale   = scale
        self.rotate  = rotate

def predictModelMudSplat(originalImage, originalClass, targetClass,
                         trainedModelWeightsPath, mudSplatObjects = None):
    """
    Adds a mud splat on the original image according to the specified feature
    vector and then predict its class using the trained model.

    Arguments:
    -----------
        originalImage: np.ndarray
            Original Image (without any perturbations)
        originalClass: scalar (int) {0,1,2}
            Original Class label specified as follows:
                turnLeft: 0
                turnRight: 1
                goStraight: 2
        targetClass: scalar (int) {0,1,2}
            Target Class (aiming to misclassify) label specified as follows:
                turnLeft: 0
                turnRight: 1
                goStraight: 2
        trainedModelWeightsPath: string
            Path to the '*.h5' file with the trained model
        mudSplatObjects: Array of mud-splat objects
            Explains the features of all mud-splats.
            (If None: Predict the class and correctness of original image.)
    Returns:
    -----------
        predClass: Classify prediction w.r.t. original and target classes:
            2   : if predicted class == original class (correct prediction)
            0   : if predicted class == target class (incorrect prediction - 
                                                      target matched)
            1   : if predicted class == any other class (incorrect prediction - 
                                                         target mismatched)
        predOutput: list of predictions
            The transformed output image with mud-splat on it
    """
    newImage = deepcopy(originalImage)
    if mudSplatObjects is not None:
        for mudSplatObj in mudSplatObjects:
            mudSplat = cv2.imread(mudSplatObj.imgPath, cv2.IMREAD_UNCHANGED)
            newImage = addMudSplat(newImage, mudSplat, mudSplatObj.xOffset,
                                   mudSplatObj.yOffset, mudSplatObj.scale,
                                   mudSplatObj.rotate)
    # load model
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model(trainedModelWeightsPath)
    predOutput = model.predict(newImage)
    if np.argmax(predOutput) == originalClass:
        predClass = 2
    elif np.argmax(predOutput) == targetClass:
        predClass = 0
    else:
        predClass = 1
    return predClass, predOutput

'''
MAIN CODE
'''

folder = '../GTSRB Dataset/categorized&cropped_NA/'
turnRight = load_images_from_folder(folder + '33')
print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder(folder + '34')
print("shape of original turnLeft", turnLeft.shape)
goStraight = load_images_from_folder(folder + '35')
print("shape of original goStraight", goStraight.shape)

outputFolder = "CorrectlyClassifiedImages/"
#testImgs = []
i = 0
for img in turnRight:
    if predictModelMudSplat(np.expand_dims(img, axis=0), 1, 0, 'model.h5')[0] == 2:
        #testImgs.append((img, 1, 0))
        cv2.imwrite(outputFolder+'33/'+str(i)+'.png', img)
        i+=1
i = 0
for img in turnLeft:
    if predictModelMudSplat(np.expand_dims(img, axis=0), 0, 1, 'model.h5')[0] == 2:
        #testImgs.append((img, 0, 1))
        cv2.imwrite(outputFolder+'33/'+str(i)+'.png', img)
        i+=1
