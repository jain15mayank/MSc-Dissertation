# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:10:14 2019

@author: Mayank Jain
"""

import numpy as np
import cv2
import copy

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    #source[source==255] = np.average(source[source != 255])
    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def addMudSplat(originalImage, mudSplatRef, SplatOffsetX = 0,
                SplatOffsetY = 0, scaleParam = 100, rotateParam = None):
    """
    Adds a mud splat on the original image at specified location. If location
    is not specified, places it arbitrarily near the center of original image.

    Arguments:
    -----------
        originalImage: np.ndarray
            Image on which mud-splat is required to be added
        mudSplatRef: np.ndarray
            Reference image of mud splat; should have alpha channel; can have
            different dimensions to source
        SplatOffsetX: scalar (int)
            Offset value in X-Dimension where the splat should be placed
        SplatOffsetY: scalar (int)
            Offset value in Y-Dimension where the splat should be placed
        scaleParam: scalar (float) (0-100)
            Specifies how much big the mud splat should be in reference to the
            size of original image (e.g.: if scaleParam==100: the splat will be
            almost equal to the size of original image); values are by default
            clipped between 0 and 100
        rotateParam: scalar (float) (0-360)
            Specifies how much to rotate the original image by (in degrees)
    Returns:
    -----------
        muddyImage: np.ndarray
            The transformed output image with mud-splat on it
    """
    # Clip scaleParam value between 0 & 100
    if scaleParam<0:
        scaleParam = 0
    elif scaleParam>100:
        scaleParam = 100
    # Scale the mud splat image - store in the variable newSplat
    sizeSplat = int(scaleParam*min([originalImage.shape[0], originalImage.shape[1]])/100)
    if mudSplatRef.shape[0]<mudSplatRef.shape[1]:
        newSplat = cv2.resize(mudSplatRef,
                   (int(mudSplatRef.shape[0]*sizeSplat/mudSplatRef.shape[1]), sizeSplat))
    else:
        newSplat = cv2.resize(mudSplatRef,
                   (sizeSplat, int(mudSplatRef.shape[1]*sizeSplat/mudSplatRef.shape[0])))
    if rotateParam is not None:
        rows,cols,nChannels = newSplat.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), rotateParam, 1)
        newSplat = cv2.warpAffine(newSplat, M, (cols,rows))
    # Perform histogram matching of newSplat w.r.t. originalImage - only on V
    # channel in HSV format
    alpha_newSplat = newSplat[:,:,3]
    testSplat = copy.deepcopy(newSplat[:,:,:3])
    #print(np.average(testSplat[:,:,0][alpha_newSplat!=0].ravel()))
    #for ch in range(3):
    #    testSplat[:,:,ch][alpha_newSplat==0] = np.average(testSplat[:,:,ch][alpha_newSplat!=0].ravel())
    #print(np.average(testSplat[:,:,0][alpha_newSplat!=0].ravel()))
    testSplat = cv2.cvtColor(testSplat, cv2.COLOR_BGR2HSV)
    img2 = copy.deepcopy(originalImage)
    tempSplat = cv2.cvtColor(newSplat[:,:,:3], cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    testSplat[:,:,2] = np.uint8(hist_match(tempSplat[:,:,2], originalImage[:,:,2]))
    testSplat = cv2.cvtColor(testSplat, cv2.COLOR_HSV2BGR)
    finSplat = np.dstack((testSplat, alpha_newSplat))
    testSplat = copy.deepcopy(finSplat)
    # Calculate standard deviation for Gaussian Blur using Original Image - to
    # be applied to the mud-splat for overlaying
    ret = cv2.threshold(np.uint8(originalImage[25:75,25:75,1]),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    '''
    temp_wh_half = []
    for pixel in originalImage[25:75,25:75,1].flatten():
        if pixel >= ret:
            temp_wh_half.append(pixel)
    '''
    temp_wh_half = originalImage[25:75,25:75,1][originalImage[25:75,25:75,1]>=ret]
    sigma = np.std(temp_wh_half)
    # Apply Gaussian Blur to Mud Splat
    testSplat = cv2.GaussianBlur(testSplat, (5, 5), sigma)
    # Apply alpha-channel adjustments and overlay mudSplat on originalImage
    y1, y2 = int(SplatOffsetY), int(SplatOffsetY) + testSplat.shape[0]
    x1, x2 = int(SplatOffsetX), int(SplatOffsetX) + testSplat.shape[1]
    alpha_s = testSplat[:, :, 3]/255
    alpha_l = 1.0 - alpha_s
    muddyImage = copy.deepcopy(originalImage)
    muddyImage[y1:y2, x1:x2, :3] = np.moveaxis(np.add(np.multiply(alpha_s, np.moveaxis(testSplat[:,:,:3],-1,0)),
            np.multiply(alpha_l, np.moveaxis(originalImage[y1:y2, x1:x2, :3],-1,0))).astype(int),0,-1)
    '''
    for c in range(0, 3):
        muddyImage[y1:y2, x1:x2, c] = ((alpha_s * testSplat[:, :, c]).astype(int) +
                  (alpha_l * originalImage[y1:y2, x1:x2, c]).astype(int))
    '''
    return muddyImage

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

def addMultiSplats(originalImage, mudSplatObjects):
    """
    Adds all mudSplatObjects onto originalImage

    Arguments:
    -----------
        originalImage: np.ndarray
            Image on which mud-splat is required to be added
        mudSplatObjects: list
            List of mud splat objects to be combined

    Returns:
    -----------
        muddyImage: np.ndarray
            The transformed output image with all mud-splats on it
    """
    imgW = originalImage.shape[0]
    imgH = originalImage.shape[1]
    xOffset = []
    yOffset = []
    finalSplats = []
    for splatObj in mudSplatObjects:
        mudSplatRef = cv2.imread(splatObj.imgPath, cv2.IMREAD_UNCHANGED)
        mudSplatRef[:,:,3][np.sum(mudSplatRef[:,:,:3], axis=2)>600] = 0
        for ch in range(3):
            mudSplatRef[:,:,ch][mudSplatRef[:,:,3]==0] = np.average(mudSplatRef[:,:,ch][mudSplatRef[:,:,3]!=0].ravel())
        xOffset.append(splatObj.xOffset)
        yOffset.append(splatObj.yOffset)
        scaleParam = splatObj.scale
        rotateParam = splatObj.rotate
        # Clip scaleParam value between 0 & 100
        if scaleParam<0:
            scaleParam = 0
        elif scaleParam>100:
            scaleParam = 100
        # Scale the mud splat image - store in the variable newSplat
        sizeSplat = int(scaleParam*min([imgW, imgH])/100)
        if mudSplatRef.shape[0]<mudSplatRef.shape[1]:
            newSplat = cv2.resize(mudSplatRef,
                       (int(mudSplatRef.shape[0]*sizeSplat/mudSplatRef.shape[1]), sizeSplat))
        else:
            newSplat = cv2.resize(mudSplatRef,
                       (sizeSplat, int(mudSplatRef.shape[1]*sizeSplat/mudSplatRef.shape[0])))
        if rotateParam is not None:
            rows,cols,nChannels = newSplat.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2), rotateParam, 1)
            newSplat = cv2.warpAffine(newSplat, M, (cols,rows))
        # Perform histogram matching of newSplat w.r.t. originalImage - only on V
        # channel in HSV format
        alpha_newSplat = newSplat[:,:,3]
        testSplat = copy.deepcopy(newSplat[:,:,:3])
        testSplat = cv2.cvtColor(testSplat, cv2.COLOR_BGR2HSV)
        img2 = copy.deepcopy(originalImage)
        tempSplat = cv2.cvtColor(newSplat[:,:,:3], cv2.COLOR_BGR2HSV)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        testSplat[:,:,2] = np.uint8(hist_match(tempSplat[:,:,2], originalImage[:,:,2]))
        testSplat = cv2.cvtColor(testSplat, cv2.COLOR_HSV2BGR)
        finSplat = np.dstack((testSplat, alpha_newSplat))
        testSplat = copy.deepcopy(finSplat)
        # Calculate standard deviation for Gaussian Blur using Original Image - to
        # be applied to the mud-splat for overlaying
        ret = cv2.threshold(np.uint8(originalImage[25:75,25:75,1]),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
        temp_wh_half = originalImage[25:75,25:75,1][originalImage[25:75,25:75,1]>=ret]
        sigma = np.std(temp_wh_half)
        # Apply Gaussian Blur to Mud Splat
        testSplat = cv2.GaussianBlur(testSplat, (5, 5), sigma)
        finalSplats.append(testSplat.astype("uint8"))
    allMudSplats = np.zeros((imgW, imgH, 4))
    for i, splat in enumerate(finalSplats):
        allMudSplats[yOffset[i]:yOffset[i]+splat.shape[0],
                    xOffset[i]:xOffset[i]+splat.shape[1], :] = np.maximum(splat,
                        allMudSplats[yOffset[i]:yOffset[i]+splat.shape[0],
                                    xOffset[i]:xOffset[i]+splat.shape[1], :])
    alpha_s = allMudSplats[:, :, 3]/255
    alpha_l = 1.0 - alpha_s
    muddyImage = copy.deepcopy(originalImage)
    muddyImage[:, :, :3] = np.moveaxis(np.add(np.multiply(alpha_s, np.moveaxis(allMudSplats[:,:,:3],-1,0)),
            np.multiply(alpha_l, np.moveaxis(originalImage[:, :, :3],-1,0))).astype(int),0,-1)
    return muddyImage.astype("uint8")

def combineSplats(mudSplatObjects, finW, finH):
    """
    Combines the given mudSplatObjects into 1 image of dimensions 'finW X finH':

    Arguments:
    -----------
        mudSplatObjects: list
            List of mud splat objects to be combined
        finW: scalar (int)
            Width of the required resulting image. Also used to determine the
            placement and orientation of the mud splat.
        finH: scalar (int)
            Height of the required resulting image. Also used to determine the
            placement and orientation of the mud splat.

    Returns:
    -----------
        allMudSplats: np.ndarray
            An image of 'finW X finH' shape with all the mudSplats added on it
    """
    allMudSplats = np.zeros((finW, finH, 4))
    for splatObj in mudSplatObjects:
        mudSplatRef = cv2.imread(splatObj.imgPath, cv2.IMREAD_UNCHANGED)
        #mudSplatRef[:,:,:-1][mudSplatRef[:,:,:-1]==255] = 0
        xOffset = splatObj.xOffset
        yOffset = splatObj.yOffset
        scaleParam = splatObj.scale
        rotateParam = splatObj.rotate
        # Clip scaleParam value between 0 & 100
        if scaleParam<0:
            scaleParam = 0
        elif scaleParam>100:
            scaleParam = 100
        # Scale the mud splat image - store in the variable newSplat
        sizeSplat = int(scaleParam*min([finW, finH])/100)
        if mudSplatRef.shape[0]<mudSplatRef.shape[1]:
            newSplat = cv2.resize(mudSplatRef,
                       (int(mudSplatRef.shape[0]*sizeSplat/mudSplatRef.shape[1]), sizeSplat))
        else:
            newSplat = cv2.resize(mudSplatRef,
                       (sizeSplat, int(mudSplatRef.shape[1]*sizeSplat/mudSplatRef.shape[0])))
        if rotateParam is not None:
            rows,cols,nChannels = newSplat.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2), rotateParam, 1)
            newSplat = cv2.warpAffine(newSplat, M, (cols,rows))
        allMudSplats[yOffset:yOffset+newSplat.shape[0],
                    xOffset:xOffset+newSplat.shape[1], :] = np.maximum(newSplat,
                        allMudSplats[yOffset:yOffset+newSplat.shape[0],
                                    xOffset:xOffset+newSplat.shape[1], :])
    return allMudSplats