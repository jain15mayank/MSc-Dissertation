import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

# Assignment rather than import because direct import from within Keras
# doesn't work in tf 1.8
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
KerasModel = tf.keras.models.Model

#from keras.models import Sequential
#from keras.layers.core import Dense, Flatten, Dropout, Activation
#from keras.layers import Conv2D, MaxPooling2D
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

'''
CREATE Training and Test Data
'''
# Get Train Dataset
turnRight = load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/33', 8300)
print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/34', 8300)
print("shape of original turnLeft", turnLeft.shape)
goStraight = load_images_from_folder('../GTSRB Dataset/categorized&cropped - Training/35', 8300)
print("shape of original goStraight", goStraight.shape)

x_train = np.concatenate((turnLeft, turnRight, goStraight))
y_train = np.zeros((x_train.shape[0], 3))
y_train[0:turnLeft.shape[0], 0] = 1
y_train[turnLeft.shape[0]:turnRight.shape[0], 1] = 1
y_train[turnRight.shape[0]:, 2] = 1

def shuffle_in_unison(a, b):
  rng_state = np.random.get_state()
  np.random.shuffle(a)
  np.random.set_state(rng_state)
  np.random.shuffle(b)
shuffle_in_unison(x_train, y_train)

# Get Test Dataset
turnRight = load_images_from_folder('../GTSRB Dataset/categorized&cropped_NA - Test/33')
print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder('../GTSRB Dataset/categorized&cropped_NA - Test/34')
print("shape of original turnLeft", turnLeft.shape)
goStraight = load_images_from_folder('../GTSRB Dataset/categorized&cropped_NA - Test/35')
print("shape of original goStraight", goStraight.shape)

x_test = np.concatenate((turnLeft, turnRight, goStraight))
y_test = np.zeros((x_test.shape[0], 3))
y_test[0:turnLeft.shape[0], 0] = 1
y_test[turnLeft.shape[0]:turnRight.shape[0], 1] = 1
y_test[turnRight.shape[0]:, 2] = 1

'''
CREATE Model
'''
# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

FLAGS = flags.FLAGS

TRAIN_FRAC = 0.85
NB_EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = .0000001
TRAIN_DIR = 'train_dir'
FILENAME = 'turnDirTest.ckpt'
LOAD_MODEL = False

# Define TF model graph
'''
model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                  channels=nchannels, nb_filters=64,
                  nb_classes=nb_classes)
#compile model using accuracy to measure model performance
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
'''
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(img_rows, img_cols, nchannels)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
'''
'''
#train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=NB_EPOCHS)

#predict first 4 images in the test set
print(model.predict(x_test[:4]))
#actual results for first 4 images in test set
print(y_test[:4])

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s Test: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(x_train, y_train, verbose=0)
print("%s Train: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_temp = np.zeros((turnRight.shape[0], 3))
y_temp[:,1] = 1
scores = model.evaluate(turnRight, y_temp, verbose=0)
print("%s Test (Turn Right): %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_temp = np.zeros((turnLeft.shape[0], 3))
y_temp[:,0] = 1
scores = model.evaluate(turnLeft, y_temp, verbose=0)
print("%s Test (Turn Left): %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_temp = np.zeros((goStraight.shape[0], 3))
y_temp[:,0] = 1
scores = model.evaluate(goStraight, y_temp, verbose=0)
print("%s Test (Go Straight): %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")
