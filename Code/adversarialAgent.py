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

from pyswarm import pso

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

def  predictModelMudSplat(originalImages, originalClass, targetClass,
                         model, mudSplatObjects = None):
    """
    Adds a mud splat on the original image according to the specified feature
    vector and then predict its class using the trained model.

    Arguments:
    -----------
        originalImages: np.ndarray
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
        trainedModelWeightsPath: Keras Model Object
            Previously Trained Model Object
        mudSplatObjects: Array of mud-splat objects
            Explains the features of all mud-splats.
            (If None: Predict the class and correctness of original image.)
    Returns:
    -----------
        predScore: Classify predictions w.r.t. original and target classes and
            give score according to the following scheme:
                2   : if predicted class == original class (correct prediction)
                0   : if predicted class == target class (incorrect prediction - 
                                                          target matched)
                1   : if predicted class == any other class (incorrect prediction - 
                                                             target mismatched)
            The scores for all images are then added to produce final score
        predOutput: list of predictions
            The transformed output image with mud-splat on it
    """
    newImages = deepcopy(originalImages)
    if len(newImages.shape) == 3:   # Only 1 image is provided
        newImages = np.expand_dims(newImages, axis=0)
    if mudSplatObjects is not None:
        numSplattedImages = newImages.shape[0]*len(mudSplatObjects)
        splattedImages = np.zeros(np.append(numSplattedImages,newImages.shape[1:]))
        for mudSplatObj in mudSplatObjects:
            mudSplat = cv2.imread(mudSplatObj.imgPath, cv2.IMREAD_UNCHANGED)
            for i, img in enumerate(newImages):
                splattedImages[i,...] = addMudSplat(img, mudSplat, mudSplatObj.xOffset,
                                                    mudSplatObj.yOffset, mudSplatObj.scale,
                                                    mudSplatObj.rotate)
        predOutput = model.predict(splattedImages)
    else:
        predOutput = model.predict(newImages)
    
    predScore = 0
    for outputs in predOutput:
        if np.any(np.argmax(outputs) == originalClass):
            predScore += 2
        elif np.any(np.argmax(outputs) == targetClass):
            predScore += 0
        else:
            predScore += 1
    return predScore, predOutput

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

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model.h5')

testTurnRightImgs = []
outputFolder = "Correctly Classified Images"
i = 0
if not os.path.exists(outputFolder+'/33/'):
    os.makedirs(outputFolder+'/33/')
for img in turnRight:
    if predictModelMudSplat(np.expand_dims(img, axis=0), 1, 0, model)[0] == 2:
        testTurnRightImgs.append((img, 1, 0))
        cv2.imwrite(outputFolder+'/33/'+str(i)+'.png', img)
        i+=1
print('here')

testTurnLeftImgs = []
i = 0
if not os.path.exists(outputFolder+'/34/'):
    os.makedirs(outputFolder+'/34/')
for img in turnLeft:
    if predictModelMudSplat(np.expand_dims(img, axis=0), 0, 1, model)[0] == 2:
        testTurnLeftImgs.append((img, 0, 1))
        cv2.imwrite(outputFolder+'/34/'+str(i)+'.png', img)
        i+=1
print('here 2')

'''

folder = "Correctly Classified Images/"
#turnRight = load_images_from_folder(folder + '33')
#print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder(folder + '34')
print("shape of original turnLeft", turnLeft.shape)
#testImgs = np.concatenate((turnRight, turnLeft), axis=0)
testImgs = turnLeft
'''

outputFolder = "Misclassified Images"
def pso_objective_1(features, *args):
    xOffset, yOffset, scale, rotate = features
    imgData, mudImgPath = args
    mudSplatObj = mudSplat(mudImgPath, xOffset, yOffset, scale, rotate)
    return predictModelMudSplat(imgData[0], imgData[1], imgData[2], model,
                                [mudSplatObj])[0]
'''
numSplats = [1]
i = 0
for imgData in testImgs:
    for num in numSplats:
        for mudImgPath in ['AdversaryImages/mudSplat1.png', 'AdversaryImages/mudSplat2.png']:
            args = (imgData, mudImgPath)
            lb = [0.2*args[0][0].shape[0], 0.2*args[0][0].shape[1], 15, 0]
            ub = [0.6*args[0][0].shape[0], 0.6*args[0][0].shape[1], 40, 360]
            q_opt, f_opt = pso(pso_objective_1, lb, ub, args=args,
                               swarmsize=100, omega=0.5, phip=0.5, phig=0.5,
                               maxiter=100, minstep=1e-8)
            if f_opt<np.log(0.5):
                img = addMudSplat(imgData[0], cv2.imread(mudImgPath, cv2.IMREAD_UNCHANGED),
                                  q_opt[0], q_opt[1], q_opt[2], q_opt[3])
                cv2.imwrite(outputFolder+'/'+str(i)+'.png', img)
                i+=1
                print(q_opt)
                print('Hooray!')
'''
numSplats = [1]
i = 0
for num in numSplats:
    for mudImgPath in ['AdversaryImages/mudSplat1.png', 'AdversaryImages/mudSplat2.png']:
        testTLimgs = np.zeros(np.append(len(testTurnLeftImgs), testTurnLeftImgs[0][0].shape))
        for j, img in enumerate(testTurnLeftImgs):
            testTLimgs[j, ...] = img[0]
        args = (np.float32(testTLimgs), mudImgPath)
        lb = [0.2*args[0][0].shape[0], 0.2*args[0][0].shape[1], 15, 0]
        ub = [0.6*args[0][0].shape[0], 0.6*args[0][0].shape[1], 40, 360]
        q_opt, f_opt = pso(pso_objective_1, lb, ub, args=args,
                           swarmsize=100, omega=0.5, phip=0.5, phig=0.5,
                           maxiter=100, minstep=1e-8, debug=True)
        print(q_opt)
        print(f_opt)
        print('Hooray!')