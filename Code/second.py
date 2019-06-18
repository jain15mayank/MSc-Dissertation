import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D
from keras.utils import np_utils
from keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cleverhans_utils_keras import cnn_model, KerasModelWrapper

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

turnRight = load_images_from_folder('../../MLdatasets/GTSRBdataset/33', 1000)
print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder('../../MLdatasets/GTSRBdataset/34', 1000)
print("shape of original turnLeft", turnLeft.shape)
goStraight = load_images_from_folder('../../MLdatasets/GTSRBdataset/35', 1000)
print("shape of original goStraight", goStraight.shape)

np.random.shuffle(turnRight)
turnRight1  = turnRight[:1000]
np.random.shuffle(turnLeft)
turnLeft1   = turnLeft[:1000]
np.random.shuffle(goStraight)
goStraight1 = goStraight[:1000]

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

# Get Test and Train Dataset
len_train = int(round(0.85*y_complete.shape[0]))
x_train = x_complete[0:len_train]
y_train = y_complete[0:len_train]
x_test  = x_complete[len_train:]
y_test  = y_complete[len_train:]

# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

'''
CREATE Model
'''
FLAGS = flags.FLAGS

TRAIN_FRAC = 0.85
NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
TRAIN_DIR = 'train_dir'
FILENAME = 'turnDirTest.ckpt'
LOAD_MODEL = False

# Define TF model graph
model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                  channels=nchannels, nb_filters=64,
                  nb_classes=nb_classes)
#compile model using accuracy to measure model performance
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=NB_EPOCHS)

#predict first 4 images in the test set
print(model.predict(x_test[:4]))
#actual results for first 4 images in test set
print(y_test[:4])

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")