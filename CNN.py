##First we should input the data the datasets at the keras file to be impurted to the python solver
'''Trains a simple convnet on the MNIST dataset.

"epochs are "
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
##epoch:Each epoch can be partitioned into groups of input-output pattern pairs called batches##
##batch_size:This define the number of patterns that the network is exposed to before the weights are updated within an epoch##

from __future__ import print_function
##It can be used to use features which will appear in newer versions while having an older release of Python.

import numpy as np
##imports numpy

np.random.seed(1337)  # for reproducibility
## seeds a random number


## import the data that I intered from the keras web site
from keras.datasets import mnist

## The Sequential model is a linear stack of layers
from keras.models import Sequential
## Dense is a nurmal neural network layer that can be eather hidden or even just an input one...etc
## Dropout is Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. It is a very efficient way of performing model averaging with neural networks.[1] The term "dropout" refers to dropping out units (both hidden and visible) in a neural network.
## Activation is activation='relu': activation functions that transform a summed signal/softmax from each neuron in a layer can be extracted and added to the Sequential as a layer-like object called Activation##
## Flatten is A layer which takes a 2-d input coming from a convolution layer or pooling layer and flattens it into something a form a fully connected layer can take. It cannot flatten already flat data.
from keras.layers import Dense, Dropout, Activation, Flatten

## Convolution2D Convolution operator for filtering windows of two-dimensional inputs.
##MaxPooling2D is Let's say we have an 4x4 matrix representing our initial input.  Let's say as well that we have a 2x2 filter that we'll run over our input. We'll have a stride of 2 (meaning the (dx, dy) for stepping over our input will be (2, 2)) and won't overlap regions.
from keras.layers import Convolution2D, MaxPooling2D

## Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
from keras.utils import np_utils

## At this time, Keras has two backend implementations available: the TensorFlow backend and the Theano backend.
##If you want the Keras modules you write to be compatible with both Theano and TensorFlow, you have to write them via the abstract Keras backend API(Application Programming Interface). Here's an intro.
from keras import backend as K

##epoch:Each epoch can be partitioned into groups of input-output pattern pairs called batches##
##batch_size:This define the number of patterns that the network is exposed to before the weights are updated within an epoch##
# Fit the model
## nb_classes 
### Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
batch_size = 128
nb_classes = 10
nb_epoch = 12


##the size of the input image
# input image dimensions
img_rows, img_cols = 28, 28

## number of convolutional filters to use
# number of convolutional filters to use
nb_filters = 32

## this pooling will have a 2x2 filter 
# size of pooling area for max pooling
pool_size = (2, 2)

## the fillter(kernels) will look into 3X3 size in the overall image which have a 32X32 size
# convolution kernel size
kernel_size = (3, 3)


## the data, shuffled and split between train and test sets
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


##!!!!!!!!!!!!!!!! The data can not be viewed !11111111111111111111
#if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
##!!!!!!!!!!!!!!!! The data can not be viewed !11111111111111111111



## this will print on the screen the following
#X_train shape: (60000, 28, 28, 1)
print('X_train shape:', X_train.shape)

## this will print on the screen the following
# 60000 train samples
print(X_train.shape[0], 'train samples')

## this will print on the screen the following
# 10000 test samples
print(X_test.shape[0], 'test samples')



## The Output become 0 or 1
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


## The Sequential model is a linear stack of layers
model = Sequential()
## nb_filters in this case we have a 32 filter every one of them will look into on feature and will cover all the image
## this is the nb_row that is 3 in this case
## this is the nb_col which is 3 in this case
## border_mode='valid' This is just an initiation for teh layer
## input_shape this is the size of our image in our case it is 32X32 and we have just one chanel
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
##rectified linear unit (ReLU) to process the data in accurdance with the relu function
model.add(Activation('relu'))
## same as above
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
## same as above
model.add(Activation('relu'))
## this will perform a pooling of the size 2X2 to get the max value of every 2X2 and output it as a value thus reducing the size
model.add(MaxPooling2D(pool_size=pool_size))
## Applies Dropout to the input. Dropout consists in randomly setting a fraction p of input units to 0 at each update during training time, which helps prevent overfitting.
model.add(Dropout(0.5))
## Flatten is A layer which takes a 2-d input coming from a convolution layer or pooling layer and flattens it into something a form a fully connected layer can take. It cannot flatten already flat data.
#Flattens the input. Does not affect the batch size.
model.add(Flatten())
## make a fully connected layer by using dense
model.add(Dense(128))
##rectified linear unit (ReLU) to process the data in accurdance with the relu function
model.add(Activation('relu'))
## same as above
model.add(Dropout(0.5))
## fully connect the network and output just 10 classes to be selected by the softmax layer
model.add(Dense(nb_classes))
## apply this layer to have a probability out put
model.add(Activation('softmax'))

## Compile the model by using the 'adadelta' method other papameter are same as the previus ones
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

## Fit the model
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

### text formatting% (return, return)
score = model.evaluate(X_test, Y_test, verbose=0)
## this will print Test score: 0.0309250710365
print('Test score:', score[0])
## this will print Test accuracy: 0.9898
print('Test accuracy:', score[1])




