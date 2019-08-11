# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:52:41 2019

@author: Mayank Jain
"""

# load and evaluate a saved model
import os
#from numpy import loadtxt
import cv2
import numpy as np

from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

# load model
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  model = load_model('../Models/VGG16 - Data Random - LR 1e-7/model.h5')

# summarize model.
model.summary()

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

folder = 'Image Data/Reshaped Signs/'

x_complete = load_images_from_folder(folder)
y_complete = np.zeros((x_complete.shape[0], 3))
y_complete[:,0] = 1


'''
EVALUATE THE MODEL
'''
for i, image in enumerate(x_complete):
    score = model.evaluate(np.expand_dims(image, axis = 0), np.expand_dims(y_complete[i], axis = 0), verbose=0)
    print("For image #%d, %s: %.2f%%" % (i, model.metrics_names[1], score[1]*100))
    if score[1]==0:
        cv2.imwrite('unsuccessfull'+str(i)+'.png', image)