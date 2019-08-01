# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:14:00 2019

@author: Mayank Jain
"""

import numpy as np
import cv2
import math

def precalc_gradtable(gradtable, W, H, randSeed=np.random.randint(0,(2**16)-1)):
    rnd = np.random.RandomState(seed = randSeed)
    for i in range(0,H):
        for j in range(0, W):
            x = float((rnd.randint(1,2*W))-W)/W
            y = float((rnd.randint(1,2*H))-H)/H
            s = math.sqrt(x * x + y * y)
            if s!=0:
                x = x / s
                y = y / s
            else:
                x = 0
                y = 0
            gradtable[i*H+j] = (x,y)
    return gradtable

#calculate dot product for v1 and v2
def dot(v1,v2):
    return ( (v1[0]*v2[0]) + (v1[1]*v2[1]) )

# get a pseudorandom gradient vector
def gradient(x,y, gradtable, W, H):
    # normalize!
    return gradtable[y*H+x]

def s_curve(x):
    return ( 3*x*x - 2*x*x*x )

def noise2d(x, y, gradtable, W, H):
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    
    i_x0 = int(x0)
    i_x1 = int(x1)
    i_y0 = int(y0)
    i_y1 = int(y1)
    
    s = dot(gradient(i_x0, i_y0, gradtable, W, H),(x-x0, y-y0))
    t = dot(gradient(i_x1, i_y0, gradtable, W, H),(x-x1, y-y0))
    u = dot(gradient(i_x0, i_y1, gradtable, W, H),(x-x0, y-y1))
    v = dot(gradient(i_x1, i_y1, gradtable, W, H),(x-x1, y-y1))
    
    s_x = s_curve( x - x0 )
    a = s + s_x*t - s_x*s
    b = u + s_x*v - s_x*u
    
    s_y = s_curve( y - y0 )
    z = a + s_y*b - s_y*a
    
    return z

def col( a ):
    return int(round((128-(128*a))))

def scaleTo8Bit(image, displayMin = None, displayMax=None ):
    if displayMin == None:
        displayMin = np.min(image)
    
    if displayMax == None:
        displayMax = np.max(image)
    
    np.clip(image, displayMin, displayMax, out=image)
    
    image = image - displayMin
    cf = 255. / (displayMax - displayMin)
    imageOut = ( cf * image).astype(np.uint8)
    
    return imageOut

def generatePerlin(W, H, randomSeed):
    gradtable = [ (0,0) for i in range(0,W*H) ]
    
    gradtable = precalc_gradtable(gradtable, W, H, randomSeed)
    
    perlin = np.zeros((W,H))
    
    zoom_x = 0.2
    zoom_y = 0.2
    x = 0.0
    y = 0.0
    for j in range(0,H):
        for i in range(0,W):
            a = col(noise2d(x,y,gradtable,W,H))
            perlin[i,j] = a
            x = x + zoom_x
        y = y + zoom_y
        x = 0.0
    perlin = cv2.GaussianBlur(perlin, (int(W/2)+1,int(H/2)+1), 0)
    return perlin

def addFog(imageList, fogIntensity=0.8, randomSeed=10):
    """
    Adds foggy effect to the image given as input.
    
    Arguments:
    -----------
        imageList: np.ndarray (numImages, Width, Height, numChannels)
            List of images on which fog is required to be added
        fogIntensity: float (0-1)
            Defines the intensity or thickness of the fog. Higher this number,
            higher the thickness.
        randomSeed: +ve int (uint16)
            Specifies the randomSeed for generating Perlin's Noise which will
            be used to add fog later. In other words, it alters the orientation
            of fog particles
    Returns:
    -----------
        fogImgs: np.ndarray (numImages, Width, Height, numChannels)
            The list of transformed output images with foggy effect
    """
    if len(imageList.shape)==3:
        imageList = np.expand_dims(imageList, axis=0)
    numImgs = imageList.shape[0]
    W = imageList.shape[1]
    H = imageList.shape[2]
    nCh = imageList.shape[3]
    
    perlin = scaleTo8Bit(generatePerlin(W, H, randomSeed))
    #cv2.imwrite('perlinFog.png', perlin)
    fogImgs = np.zeros((numImgs,W,H,nCh))
    for n in range(nCh):
        fogImgs[:,:,:,n] = np.add((fogIntensity*perlin), np.multiply(1-(fogIntensity*perlin/255), imageList[:,:,:,n]))
    return fogImgs.astype('uint8')

def addRain(imageList, randomSeed=10, mode='withMist'):
    """
    Adds rain effect on glass to the image given as input.
    
    Arguments:
    -----------
        imageList: np.ndarray (numImages, Width, Height, numChannels)
            List of images on which raindrop effect is required to be added
        randomSeed: +ve int (uint16)
            Specifies the randomSeed for generating Perlin's Noise which will
            be used to add rain drops later. In other words, it alters the
            location of rain drops.
        mode: string ('withMist' or 'noMist')
            If 'withMist': adds a gaussian blur to original image before adding rain drops
            If 'noMist':   no effect of blur/mist on original image
    Returns:
    -----------
        rainImgs: np.ndarray (numImages, Width, Height, numChannels)
            The list of transformed output images with raindrops effect
    """
    if len(imageList.shape)==3:
        imageList = np.expand_dims(imageList, axis=0)
    numImgs = imageList.shape[0]
    W = imageList.shape[1]
    H = imageList.shape[2]
    nCh = imageList.shape[3]
    rainImgs = np.zeros((numImgs,W,H,nCh))
    for j,image in enumerate(imageList):
        if (mode=='withMist'):
            image[:,:,:3] = cv2.GaussianBlur(image[:,:,:3], (5,5), 5)
        perlin = generatePerlin(W, H, randomSeed)
        #cv2.imwrite('perlinRain.png', perlin)
        perlin_thr = cv2.threshold(perlin,127,255,cv2.THRESH_BINARY_INV)[1]
        alpha = perlin_thr==255
        alpha = 255*alpha.astype(int)
        perlin_thr = cv2.merge((perlin_thr.astype(int), perlin_thr.astype(int), perlin_thr.astype(int), alpha))
        rainDrops = cv2.GaussianBlur(scaleTo8Bit(perlin_thr), (3,3), 0)
        imgIn = rainDrops[:,:,:-1].astype(np.float32)
        #Create the  filter
        kernel = np.zeros( (1,2), np.float32)
        kernel[0,0] = -3.0  #Try 2.0
        kernel[0,1] = 3.0   #Try 2.0
        
        #Do the actual kernel operation...
        output = cv2.filter2D(imgIn.astype(np.float32), -1, kernel)
        #Scaling back so that the zero point is at 128 (gray)...
        output8bit = scaleTo8Bit(output)
        output8bit = cv2.GaussianBlur(output8bit, (3,3), 0)
        alpha[alpha==255]=100
        alpha_s = alpha/255
        alpha_l = 1.0 - alpha_s
        for i in range(nCh):
            if(i<3):
                rainImgs[j,:,:,i] = np.add(np.multiply(image[:,:,i], alpha_l), np.multiply(output8bit[:,:,i], alpha_s))
            else:
                rainImgs[j,:,:,i] = image[:,:,i]
    return rainImgs

def addAlpha(imageList):
    newImageList = np.zeros((len(imageList), imageList[0].shape[0], imageList[0].shape[1], 4))
    for i, image in enumerate(imageList):
        if image.shape[2]==3:
            newImageList[i,...] = np.dstack((image, 255*np.ones((image.shape[0],image.shape[1]))))
        else:
            newImageList[i,...] = image
    return newImageList