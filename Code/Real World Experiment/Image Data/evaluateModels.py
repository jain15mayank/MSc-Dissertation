# load and evaluate a saved model
import os
#from numpy import loadtxt
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

#modelPath = '../../../Models/ResNet20 Batch Max - Data Random - LR 1e-6/model.h5'
modelPath = '../../../Models/VGG16 - Data Random - LR 1e-7/model.h5'

# load model
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  model = load_model(modelPath)

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

folder = 'Reshaped Signs/'

imgs = load_images_from_folder(folder)
x_imgs = imgs[:]
y_imgs = np.zeros((imgs.shape[0], 3))
y_imgs[:,0] = 1

'''
EVALUATE THE MODEL
'''
score = model.evaluate(x_imgs, y_imgs, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))