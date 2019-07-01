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
images34 = load_images_from_folder(folder+'34')
images35 = load_images_from_folder(folder+'35')

mudSplat = cv2.imread('AdversaryImages/mudSplat2.png', cv2.IMREAD_UNCHANGED)
print(mudSplat.shape)

for i in range(images33.shape[0]):
    cv2.imwrite('splattedImages/33/'+str(i)+'.png', addMudSplat(images33[i], mudSplat, 35, 35, 20))
print('Category 33 Complete')
for i in range(images34.shape[0]):
    cv2.imwrite('splattedImages/34/'+str(i)+'.png', addMudSplat(images34[i], mudSplat, 35, 35, 20))
print('Category 34 Complete')
for i in range(images35.shape[0]):
    cv2.imwrite('splattedImages/35/'+str(i)+'.png', addMudSplat(images35[i], mudSplat, 35, 35, 20))
print('Category 35 Complete')