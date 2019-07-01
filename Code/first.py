import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras import backend
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import tf_model_load, model_loss
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval

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

turnRight = load_images_from_folder('../../MLdatasets/GTSRB Dataset/33', 1000)
print("shape of original turnRight", turnRight.shape)
turnLeft = load_images_from_folder('../../MLdatasets/GTSRB Dataset/34', 1000)
print("shape of original turnLeft", turnLeft.shape)
goStraight = load_images_from_folder('../../MLdatasets/GTSRB Dataset/35', 1000)
print("shape of original goStraight", goStraight.shape)

np.random.shuffle(turnRight)
turnRight1 	= turnRight[:1000]
np.random.shuffle(turnLeft)
turnLeft1 	= turnLeft[:1000]
np.random.shuffle(goStraight)
goStraight1	= goStraight[:1000]

FLAGS = flags.FLAGS

TRAIN_FRAC = 0.85
NB_EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE = .001
TRAIN_DIR = 'train_dir'
FILENAME = 'turnDirTest.ckpt'
LOAD_MODEL = False

def binaryCNNsetup(train_set_frac=TRAIN_FRAC, nb_epochs=NB_EPOCHS,
                   batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                   train_dir=TRAIN_DIR, filename=FILENAME,
                   load_model=LOAD_MODEL, testing=True, label_smoothing=0.1):
  """
  Binary CNN Basic Setup
  :param train_set_frac: fraction of whole dataset to be considered for training
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param train_dir: Directory storing the saved model
  :param filename: Filename to save model under
  :param load_model: True for load, False for not load
  :param testing: if true, test error is calculated
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """
  backend.set_learning_phase(0)
  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()
  
  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)
  
  # Create TF session and set as Keras backend session
  sess = tf.Session()
  backend.set_session(sess)
  
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
  len_train = int(round(train_set_frac*y_complete.shape[0]))
  x_train = x_complete[0:len_train]
  y_train = y_complete[0:len_train]
  x_test  = x_complete[len_train:]
  y_test  = y_complete[len_train:]
  
  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]
  
  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  
  # Define TF model graph
  model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels, nb_filters=64,
                    nb_classes=nb_classes)
  preds = model(x)
  print("Defined TensorFlow model graph.")
  
  def evaluate():
    # Evaluate the accuracy of the model on test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    report.clean_train_clean_eval = acc
    # assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate examples: %0.4f' % acc)
  
  # Train the model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'train_dir': train_dir,
      'filename': filename
  }
  
  rng = np.random.RandomState([2017, 8, 30])
  if not os.path.exists(train_dir):
    os.mkdir(train_dir)
  
  ckpt = tf.train.get_checkpoint_state(train_dir)
  print(train_dir, ckpt)
  ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
  wrap = KerasModelWrapper(model)
  
  if load_model and ckpt_path:
    saver = tf.train.Saver()
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    print("Model loaded from: {}".format(ckpt_path))
    evaluate()
  else:
    saver = tf.train.Saver()
    print("Model was not loaded, training from scratch.")
    loss = CrossEntropy(wrap, smoothing=label_smoothing)
    train(sess, loss, x_train, y_train, evaluate=evaluate,
          args=train_params, rng=rng)
    save_path = saver.save(sess, "/train_dir/trainedModel.ckpt")
    print("Model saved in path: %s" % save_path)
  
  # Calculate training error
  if testing:
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_train, y_train, args=eval_params)
    report.train_clean_train_clean_eval = acc
  
  return report

def main(argv=None):
  binaryCNNsetup(nb_epochs=FLAGS.nb_epochs,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 train_dir=FLAGS.train_dir,
                 filename=FLAGS.filename,
                 load_model=FLAGS.load_model)

def del_all_flags(FLAGS):
  flags_dict = FLAGS._flags()
  keys_list = [keys for keys in flags_dict]
  for keys in keys_list:
    FLAGS.__delattr__(keys)

if __name__ == '__main__':
  del_all_flags(flags.FLAGS)
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_string('train_dir', TRAIN_DIR,
                      'Directory where to save model.')
  flags.DEFINE_string('filename', FILENAME, 'Checkpoint filename.')
  flags.DEFINE_boolean('load_model', LOAD_MODEL,
                       'Load saved model or train.')
  tf.app.run()
