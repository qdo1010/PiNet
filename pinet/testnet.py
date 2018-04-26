from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D,MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import keras.utils.np_utils as kutils
from keras.utils import np_utils
import keras.callbacks as callbacks
#from keras.utils.visualize_util import plot, model_to_dot
import keras.backend as K
#iimport MNIST.DataClean as dc
from keras.datasets import mnist
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import model_from_json


classes = 10
(trainX,trainY), (testX, testY) = mnist.load_data()
testX = testX.reshape(testX.shape[0], 1, 28, 28)
testX = testX.astype(float)
testX /= 255.0
#testY = kutils.to_categorical(testY)
testY = np_utils.to_categorical(testY, classes) * 2 - 1

lr_start = 1e-3
# load json and create model, change here to get a different weight
json_file = open('/Users/qdo/SqueezeXNOR/xnornet/Models/Squeezenet/squeezenet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model

#####change here to get a different weight
loaded_model.load_weights("/Users/qdo/SqueezeXNOR/xnornet/Models/Squeezenet/weights-improvement-04-0.98.hdf5")
print("Loaded model from disk")

# evaluate loaded model on test data
opt = Adam(lr=lr_start)
loaded_model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("compiling")
#score = loaded_model.evaluate(testX, testY, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
loaded_model.save('squeeze.h5')
from scipy.misc import imread
import time
t0 = time.time()
x = imread('test3.png',mode='L')
x = np.invert(x)
x = x.reshape(1, 1, 28, 28)/225.
out = loaded_model.predict(x)
print(np.argmax(out))
t1 = time.time() - t0
print(t1)
