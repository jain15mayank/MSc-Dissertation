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

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

#from pso_pyswarm import pso
from pso_pyswarm_mudSplat import pso

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
        newImgList = np.zeros((numImgs*36, imgRows, imgCols, nChannels))
        obliquePercentages = [20, 25, 30, 35]
        farScales = [1, 0.65, 0.45, 0.25]
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
    newImages = deepcopy(originalImages).astype('float32')
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
    accuracy = 0
    for outputs in predOutput:
        if np.any(np.argmax(outputs) == originalClass):
            predScore += 2
            accuracy += 1
        elif np.any(np.argmax(outputs) == targetClass):
            predScore += 0
        else:
            predScore += 1
    return predScore, predOutput, accuracy*100/len(predOutput)

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

x_complete = np.concatenate((turnLeft, turnRight, goStraight))
y_complete = np.zeros((x_complete.shape[0], 3))
y_complete[0:turnLeft.shape[0], 0] = 1
y_complete[turnLeft.shape[0]:turnLeft.shape[0] + turnRight.shape[0], 1] = 1
y_complete[turnLeft.shape[0] + turnRight.shape[0]:, 2] = 1

x_turnLeft = turnLeft
y_turnLeft = np.zeros((x_turnLeft.shape[0], 3))
y_turnLeft[:,0] = 1

x_turnRight = turnRight
y_turnRight = np.zeros((x_turnRight.shape[0], 3))
y_turnRight[:,1] = 1

x_goStraight = goStraight
y_goStraight = np.zeros((x_goStraight.shape[0], 3))
y_goStraight[:,2] = 1

#modelPath = '../Models/ResNet20 Batch Max - Data Random - LR 1e-6/model.h5'
modelPath = '../Models/VGG16 - Data Random - LR 1e-7/model.h5'

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('model.h5')

outputFolder = "Misclassified Images"
def pso_objective(features, *args):
    xOffset, yOffset, scale, rotate = features
    imgData, oriClass, tarClass, mudImgPath, model = args
    mudSplatObj = mudSplat(mudImgPath, xOffset, yOffset, scale, rotate)
    return predictModelMudSplat(imgData, oriClass, tarClass, model,
                                [mudSplatObj])[0]

'''
For all images in same class - TurnLeft
'''
numSplats = [1]
for num in numSplats:
    for mudImgPath in ['AdversaryImages/mudSplat2.png', 'AdversaryImages/mudSplat1.png']:
        testTLimgs = np.zeros(np.append(len(x_turnLeft), x_turnLeft[0].shape))
        for j, img in enumerate(x_turnLeft):
            testTLimgs = makeObservations(img)
            oriClass = 0
            tarClass = 1
            args = (np.float32(testTLimgs), oriClass, tarClass, mudImgPath, model)
            lb = [0.2*args[0][0].shape[0], 0.2*args[0][0].shape[1], 15, 0]
            ub = [0.6*args[0][0].shape[0], 0.6*args[0][0].shape[1], 40, 360]
            #q_opt, f_opt = pso(pso_objective, lb, ub, args=args,
            #                   swarmsize=100, omega=0.8, phip=2.0, phig=2.0,
            #                   maxiter=3000, minstep=1e-8, debug=True, processes=1)
            q_opt, f_opt = pso(pso_objective, lb, ub, args=args,
                               swarmsize=100, omega=0.8, phip=2.0, phig=2.0,
                               maxiter=2, minstep=1e-8, debug=True)
            print(q_opt)
            print(f_opt)
            print('Hooray!')
            f = open("PSOoutput.txt", "a+")

            f.write("For Turn Left Class (mudImgPath: " + mudImgPath + ") - image #" + str(j) + ":\n")
            f.write("Original Accuracy: " + str(predictModelMudSplat(testTLimgs, oriClass, tarClass, model)[2]) + "\n")
            f.write("q_opt = " + str(q_opt) + "\n")
            f.write("f_opt = " + str(f_opt) + "\n")
            mudSplatObject = mudSplat(mudImgPath, q_opt[0], q_opt[1], q_opt[2], q_opt[3])
            f.write("New Accuracy: " + str(predictModelMudSplat(testTLimgs, oriClass, tarClass, model, [mudSplatObject])[2]) + "\n\n")
            f.close()

'''
For all images in same class - TurnRight
'''
numSplats = [1]
for num in numSplats:
    for mudImgPath in ['AdversaryImages/mudSplat2.png', 'AdversaryImages/mudSplat1.png']:
        testTRimgs = np.zeros(np.append(len(x_turnRight), x_turnRight[0].shape))
        for j, img in enumerate(x_turnRight):
            testTRimgs[j, ...] = img
        oriClass = 1
        tarClass = 0
        args = (np.float32(testTRimgs), oriClass, tarClass, mudImgPath)
        lb = [0.2*args[0][0].shape[0], 0.2*args[0][0].shape[1], 15, 0]
        ub = [0.6*args[0][0].shape[0], 0.6*args[0][0].shape[1], 40, 360]
        #q_opt, f_opt = pso(pso_objective, lb, ub, args=args,
        #                   swarmsize=100, omega=0.8, phip=2.0, phig=2.0,
        #                   maxiter=3000, minstep=1e-8, debug=True, processes=1)
        q_opt, f_opt = pso(pso_objective, lb, ub, args=args,
                           swarmsize=100, omega=0.8, phip=2.0, phig=2.0,
                           maxiter=3000, minstep=1e-8, debug=True)
        print(q_opt)
        print(f_opt)
        print('Hooray!')
        f = open("PSOoutput.txt", "a+")
        f.write("For Turn Right Class (mudImgPath: " + mudImgPath + "):\n")
        f.write("q_opt = " + str(q_opt) + "\n")
        f.write("f_opt = " + str(f_opt) + "\n")
        f.close()

'''
For all images in same class - Go Straight
'''
numSplats = [1]
for num in numSplats:
    for mudImgPath in ['AdversaryImages/mudSplat2.png', 'AdversaryImages/mudSplat1.png']:
        testGSimgs = np.zeros(np.append(len(x_goStraight), x_goStraight[0].shape))
        for j, img in enumerate(x_goStraight):
            testGSimgs[j, ...] = img
        oriClass = 2
        tarClass = 0
        args = (np.float32(testGSimgs), oriClass, tarClass, mudImgPath)
        lb = [0.2*args[0][0].shape[0], 0.2*args[0][0].shape[1], 15, 0]
        ub = [0.6*args[0][0].shape[0], 0.6*args[0][0].shape[1], 40, 360]
        #q_opt, f_opt = pso(pso_objective, lb, ub, args=args,
        #                   swarmsize=100, omega=0.8, phip=2.0, phig=2.0,
        #                   maxiter=3000, minstep=1e-8, debug=True, processes=1)
        q_opt, f_opt = pso(pso_objective, lb, ub, args=args,
                           swarmsize=100, omega=0.8, phip=2.0, phig=2.0,
                           maxiter=3000, minstep=1e-8, debug=True)
        print(q_opt)
        print(f_opt)
        print('Hooray!')
        f = open("PSOoutput.txt", "a+")
        f.write("For Go Straight Class (mudImgPath: " + mudImgPath + "):\n")
        f.write("q_opt = " + str(q_opt) + "\n")
        f.write("f_opt = " + str(f_opt) + "\n")
        f.close()
