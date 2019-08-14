# -*- coding: utf-8 -*-
"""
Created on Tue Jul  30 15:15:21 2019

@author: Mayank Jain
"""

import os
import cv2
from copy import deepcopy
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

from pso_pyswarm_parallel import pso

from utils_mudSlap import *
from utils_naturalPerturbations import addRain, addFog

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

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
        increaseFactor = ((2*len(obliquePercentages))+1)*len(farScales)
        newImgList = np.zeros((numImgs*increaseFactor, imgRows, imgCols, nChannels))
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
        increaseFactor = 1
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
    return newImgList, increaseFactor

'''
HELPER Functions
'''
def alterImages(imageList, alterFeatures = None, withRain = True):
    """
    Given a feature vector of alterations (as described below), alters a set of
    images accordingly

    Arguments:
    -----------
        imageList: np.ndarray (numImages, Width, Height, numChannels)
            List of images on which alterations are required to be added
        alterFeatures: List [<mudSplat1>, <mudSplat2>, <mudSplat3>, <rain>, <fog>]
            List of features to explain modifications in the order as defined above.
            Further explanations of each component as follows:
                <mudSplat1> : [mudSpaltObject]
                <mudSplat2> : [mudSpaltObject]
                <mudSplat3> : [mudSpaltObject]
                <rain>      : [randomSeed, rainy]
                <fog>       : [fogIntensity, randomSeed]
        withRain: boolean
            If False, will ignore the rain features
    Returns:
    -----------
        outImgs: np.ndarray (numImages, Width, Height, numChannels)
            The list of transformed output images with 'alterFeatures' effects
    """
    if len(imageList.shape)==3:
        imageList = np.expand_dims(imageList, axis=0)
    numImgs = imageList.shape[0]
    W = imageList.shape[1]
    H = imageList.shape[2]
    nCh = imageList.shape[3]
    if alterFeatures is not None:
        mudObj1  = alterFeatures[0]
        mudObj2  = alterFeatures[1]
        mudObj3  = alterFeatures[2]
        rainSeed = alterFeatures[3]
        rainExt  = alterFeatures[4]
        fogInten = alterFeatures[5]
        fogSeed  = alterFeatures[6]

        if mudObj1.scale>0 and mudObj2.scale>0 and mudObj3.scale>0:
            allSplatImg = combineSplats([mudObj1]+[mudObj2]+[mudObj3], W, H).astype('uint8')
            allSplatImg[:,:,:-1][allSplatImg[:,:,:-1]==0] = 255
            splatImgs = np.zeros(imageList.shape)
            for i, image in enumerate(imageList):
                splatImgs[i, ...] = addMudSplat(image, allSplatImg)
        elif mudObj1.scale>0 and mudObj2.scale>0:
            allSplatImg = combineSplats([mudObj1]+[mudObj2], W, H).astype('uint8')
            allSplatImg[:,:,:-1][allSplatImg[:,:,:-1]==0] = 255
            splatImgs = np.zeros(imageList.shape)
            for i, image in enumerate(imageList):
                splatImgs[i, ...] = addMudSplat(image, allSplatImg)
        elif mudObj1.scale>0:
            allSplatImg = combineSplats([mudObj1], W, H).astype('uint8')
            allSplatImg[:,:,:-1][allSplatImg[:,:,:-1]==0] = 255
            splatImgs = np.zeros(imageList.shape)
            for i, image in enumerate(imageList):
                splatImgs[i, ...] = addMudSplat(image, allSplatImg)
        else:
            splatImgs = imageList

        if np.ceil(rainExt)>0 and withRain and np.ceil(fogInten)>0:
            outImgs = addRain(addFog(splatImgs, fogInten, int(fogSeed)), int(rainSeed))
        elif np.ceil(rainExt)>0 and withRain:
            outImgs = addRain(splatImgs, int(rainSeed))
        elif np.ceil(fogInten)>0:
            outImgs = addFog(splatImgs, fogInten, int(fogSeed))
        else:
            outImgs = splatImgs
    else:
        outImgs = imageList
    return outImgs

def predictModel(originalImages, originalClass, targetClass,
                         model, alterFeatures = None, exportDir = None):
    """
    Adds a mud splat on the original image according to the specified feature
    vector and then predict its class using the trained model.

    Arguments:
    -----------
        originalImages: np.ndarray (numImages, Width, Height, numChannels)
            List of original images (without any perturbations)
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
        alterFeatures: List [[<mudSplat1>, <mudSplat2>, <mudSplat3>, <rain>, <fog>]]
            List of list of features to explain modifications in the order as defined above.
            **Must be either of length 1 or equal to the number of originalImages or None.
            Further explanations of each component as follows:
                <mudSplat1> : [mudSpaltObject]
                <mudSplat2> : [mudSpaltObject]
                <mudSplat3> : [mudSpaltObject]
                <rain>      : [randomSeed, rainy]
                <fog>       : [fogIntensity, randomSeed]
        exportDir: string
            Path of directory to store correctly and incorrectly classified images
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
    if len(originalImages.shape)==3:
        originalImages = np.expand_dims(originalImages, axis=0)
    numImgs = originalImages.shape[0]
    W = originalImages.shape[1]
    H = originalImages.shape[2]
    nCh = originalImages.shape[3]

    if alterFeatures is None:
        # Predict on originalImages directly
        finImages = originalImages
        predOutput = model.predict(finImages)
    elif len(alterFeatures) == 1:
        # Apply same modifications to all originalImages
        finImages = np.zeros(originalImages.shape).astype("uint8")
        for i, image in enumerate(originalImages):
            finImages[i, ...] = addMultiSplats(image, alterFeatures[:3])
        if alterFeatures[3]>0:
            finImages = addFog(finImages, alterFeatures[3], int(alterFeatures[4]))
        if alterFeatures[7]>0:
            finImages = addRain(finImages, int(alterFeatures[5]), int(alterFeatures[6]))
        predOutput = model.predict(finImages)
    else:
        raise Exception('More than 1 alterFeatures provided.')

    predScore = 0
    accuracy = 0
    for i, outputs in enumerate(predOutput):
        if np.any(np.argmax(outputs) == originalClass):
            predScore += 2
            accuracy += 1
            if exportDir is not None:
                if not os.path.exists(exportDir+'Correct Classification/'):
                    os.makedirs(exportDir+'Correct Classification/')
                cv2.imwrite(exportDir+'Correct Classification/'+str(i)+'.png', finImages[i,:,:,:])
        elif np.any(np.argmax(outputs) == targetClass):
            predScore += 0
            if exportDir is not None:
                if not os.path.exists(exportDir+'Target Classification/'):
                    os.makedirs(exportDir+'Target Classification/')
                cv2.imwrite(exportDir+'Target Classification/'+str(i)+'.png', finImages[i,:,:,:])
        else:
            predScore += 1
            if exportDir is not None:
                if not os.path.exists(exportDir+'Incorrect Classification/'):
                    os.makedirs(exportDir+'Incorrect Classification/')
                cv2.imwrite(exportDir+'Incorrect Classification/'+str(i)+'.png', finImages[i,:,:,:])
    return predScore, predOutput, accuracy*100/len(predOutput)

'''
CREATE Dataset
'''
folder = 'Image Data/Reshaped Signs/'
x_realExp = load_images_from_folder(folder).astype("uint8")
y_realExp = np.zeros((x_realExp.shape[0], 3))
y_realExp[:,0] = 1

#modelPath = '../../Models/ResNet20 Batch Max - Data Random - LR 1e-6/model.h5'
modelPath = '../../Models/VGG16 - Data Random - LR 1e-7/model.h5'

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model(modelPath)

def pso_objective(features, *args):
    xOffset, yOffset, scale, rotate = features
    imgData, oriClass, tarClass, mudImgPath, model = args
    mudSplatObj = mudSplat(mudImgPath, xOffset, yOffset, scale, rotate)
    return predictModel(imgData, oriClass, tarClass, model, [mudSplatObj])[0]

'''
For all images obtained in Real World Experiment
'''
folder = "PSOoutput"
mudSplat1_exist = True
mudSplat2_exist = True
mudSplat3_exist = True
addFog_exist = True
addRain_exist = True


for mudId, mudImgPath in enumerate(['../AdversaryImages/mudSplat2.png', '../AdversaryImages/mudSplat1.png']):
    oriClass = 0
    tarClass = 1
    args = (x_realExp, oriClass, tarClass, mudImgPath, model)
    # Pre-process features
    fog_lb = [0, 0]
    rain_lb = [0, 0, 0]
    if mudSplat1_exist:
        ms1_lb  = [0.2*args[0][0].shape[0], 0.2*args[0][0].shape[1], 15, 0]
        ms1_ub  = [0.6*args[0][0].shape[0], 0.6*args[0][0].shape[1], 40, 360]
    else:
        ms1_lb  = [0, 0, 0, 0]
        ms1_ub  = [0, 0, 0, 0]
    if mudSplat2_exist:
        ms2_lb  = [0.2*args[0][0].shape[0], 0.2*args[0][0].shape[1], 15, 0]
        ms2_ub  = [0.6*args[0][0].shape[0], 0.6*args[0][0].shape[1], 30, 360]
    else:
        ms2_lb  = [0, 0, 0, 0]
        ms2_ub  = [0, 0, 0, 0]
    if mudSplat3_exist:
        ms3_lb  = [0.2*args[0][0].shape[0], 0.2*args[0][0].shape[1], 15, 0]
        ms3_ub  = [0.6*args[0][0].shape[0], 0.6*args[0][0].shape[1], 20, 360]
    else:
        ms3_lb  = [0, 0, 0, 0]
        ms3_ub  = [0, 0, 0, 0]
    if addFog_exist:
        fog_ub  = [0.5, (2**16)-1]
    else:
        fog_ub  = [0, 0]
    if addRain_exist:
        rain_ub = [(2**16)-1, 3, 1]
    else:
        rain_ub = [0, 0, 0]
    
    lb = ms1_lb + ms2_lb + ms3_lb + rain_lb + fog_lb
    ub = ms1_ub + ms2_ub + ms3_ub + rain_ub + fog_ub
    
    q_opt, f_opt, f_hist = pso(pso_objective, lb, ub, args=args,
                                    swarmsize=100, omega=0.8, phip=2.0, phig=2.0,
                                    maxiter=2, minstep=1e-8, debug=True, abs_min=0)
    
    print(q_opt)
    print(f_opt)
    print('Hooray!')
    if not os.path.exists(folder+'/Mud'+str(mudId)+'/'):
        os.makedirs(folder+'/Mud'+str(mudId)+'/')
    fig = plt.figure()
    plt.plot(f_hist)
    plt.title('Model accuracy')
    plt.ylabel('Global Minima')
    plt.xlabel('Iteration')
    plt.ylim(0, 25)
    plt.legend(['Convergence Characteristics'], loc='upper right')
    fig.savefig(folder + '/Mud' + str(mudId) + '/pso_convergence.png')
    
    f = open("PSOoutput.txt", "a+")
    oriAccuracy = predictModel(x_realExp, oriClass, tarClass, model)[2]
    f.write("For mudImgPath: " + mudImgPath + " :\n")
    f.write("Original Accuracy: " + str(oriAccuracy) + "\n")
    f.write("q_opt = " + str(q_opt) + "\n")
    f.write("f_opt = " + str(f_opt) + "\n")
    
    mudSplatObject1 = mudSplat(mudImgPath, int(q_opt[0]), int(q_opt[1]), q_opt[2], q_opt[3])
    mudSplatObject2 = mudSplat(mudImgPath, int(q_opt[4]), int(q_opt[5]), q_opt[6], q_opt[7])
    mudSplatObject3 = mudSplat(mudImgPath, int(q_opt[8]), int(q_opt[9]), q_opt[10], q_opt[11])
    fogFeatures     = [q_opt[12], int(q_opt[13])]
    rainFeatures    = [int(q_opt[14]), int(q_opt[15]), np.ceil(q_opt[16])]
    allFeatures     = [[mudSplatObject1] + [mudSplatObject2] + [mudSplatObject3] + fogFeatures + rainFeatures]
    
    predAccuracy = predictModel(x_realExp, oriClass, tarClass, model, allFeatures, folder+'/Mud'+str(mudId)+'/')[2]
    f.write("New Accuracy: " + str(predAccuracy) + "\n\n")
    f.close()