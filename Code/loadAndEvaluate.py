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
  model = load_model('model.h5')

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

#folder = '../../MLdatasets/GTSRB Dataset/'
folder = 'splattedImages/'
#folder = '../GTSRB Dataset/categorized&cropped_NA/'

turnRight = load_images_from_folder(folder + '33')
print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder(folder + '34')
print("shape of original turnLeft", turnLeft.shape)
goStraight = load_images_from_folder(folder + '35')
print("shape of original goStraight", goStraight.shape)

np.random.shuffle(turnRight)
turnRight1  = turnRight[:]
np.random.shuffle(turnLeft)
turnLeft1   = turnLeft[:]
np.random.shuffle(goStraight)
goStraight1 = goStraight[:]

'''
CREATE Training and Test Data
'''
# Get Complete Dataset
x_complete = np.concatenate((turnLeft1, turnRight1, goStraight1))
y_complete = np.zeros((x_complete.shape[0], 3))
y_complete[0:turnLeft1.shape[0], 0] = 1
y_complete[turnLeft1.shape[0]:turnRight1.shape[0], 1] = 1
y_complete[turnRight1.shape[0]:, 2] = 1

def shuffle_in_unison(a, b):
  rng_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(rng_state)
  np.random.shuffle(b)
shuffle_in_unison(x_complete, y_complete)

'''
EVALUATE THE MODEL
'''
score = model.evaluate(x_complete, y_complete, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))