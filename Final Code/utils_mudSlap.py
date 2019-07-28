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

def addMudSplat(originalImage, mudSplatRef, SplatOffsetX = None,
                SplatOffsetY = None, scaleParam = 10, rotateParam = None):
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
    testSplat = cv2.GaussianBlur(testSplat, (3, 3), sigma)
    # Apply alpha-channel adjustments and overlay mudSplat on originalImage
    y1, y2 = int(SplatOffsetY), int(SplatOffsetY) + testSplat.shape[0]
    x1, x2 = int(SplatOffsetX), int(SplatOffsetX) + testSplat.shape[1]
    alpha_s = testSplat[:, :, 3]/255
    alpha_l = 1.0 - alpha_s
    muddyImage = copy.deepcopy(originalImage)
    muddyImage[y1:y2, x1:x2, :] = np.moveaxis(np.add(np.multiply(alpha_s, np.moveaxis(testSplat[:,:,:3],-1,0)),
            np.multiply(alpha_l, np.moveaxis(originalImage[y1:y2, x1:x2, :],-1,0))).astype(int),0,-1)
    '''
    for c in range(0, 3):
        muddyImage[y1:y2, x1:x2, c] = ((alpha_s * testSplat[:, :, c]).astype(int) +
                  (alpha_l * originalImage[y1:y2, x1:x2, c]).astype(int))
    '''
    return muddyImage
