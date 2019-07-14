# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:20:12 2019

@author: Mayank Jain
"""

import os
import numpy as np
import cv2
from utils_mudSlap import addMudSplat

folder = '../GTSRB Dataset/categorized&cropped_NA/'

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

images33 = load_images_from_folder(folder+'33')
#images34 = load_images_from_folder(folder+'34')
#images35 = load_images_from_folder(folder+'35')

mudSplat = cv2.imread('AdversaryImages/mudSplat2.png', cv2.IMREAD_UNCHANGED)
print(mudSplat.shape)

if not os.path.exists('splattedImagesFeatureV/33/'):
    os.makedirs('splattedImagesFeatureV/33/')
m = 0
for i in range(2, 62, 5):
    for j in range(2, 62, 5):
        for k in range(15, 42, 5):
            for l in range(0, 360, 50):
                cv2.imwrite('splattedImagesFeatureV/33/'+str(m)+'.png', addMudSplat(images33[0], mudSplat, int(i*images33[0].shape[0]/100), int(j*images33[0].shape[1]/100), k, l))
                m+=1