import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils

"""CNN libraries""" 
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#from keras.layers import Convolution2D
from keras import backend as K
K.set_image_dim_ordering('th')

import matplotlib.pyplot as plt


"""Run through example, explain line by line
by Patrapee Pongtana on tutorial:
https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/"""

"""Load the data from MNIST"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()

"""Plot a sameple 0: X_train[0]"""
plt.subplot(221)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[3])
plt.show()
"""Randomize the seed"""
seed = 7
numpy.random.seed(seed)
"""
X_train.shape[0] = total samples = 60000
X_train.shape[1] = width = 28
X_train.shape[2] = height = 28
"""
num_pixels = X_train.shape[1] * X_train.shape[2]
print('Number of picture pixel',num_pixels)

"""Turn the shape into the vector of a vector. Row of 60000 samples.
	and 784 columns"""
"""Last dataset should be X_train[59999]"""
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

#Normalize all inputs from 0-255 to 0-1
X_train = X_train /255
X_test = X_test /255
print(X_train[59999])

"""Class label: Convert 1 dimension array into 10 dimensions"""
print('Class label')
print(y_train)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('After categorical\n',y_train)
num_classes = y_test.shape[1]
print('Number of class: ', num_classes)

"""CNN Architecture, Initialize the Model"""
def CNN_model():
	"""3 Convolutional Layers along with MaxPooling2D,
		with ReLu activation functions
		Layer 1: 30 features map, 5x5 pixels, MaxPoolSize 2,2
		Layer 2: 15 features map, 3x3 pixels, MaxPoolSize 2,2
		Layer 3: 7 features map, 2x2 pixels, MaxPoolSize 2,2
		Dropout layer with 20% (to reduce overfitting)
		Then Flat the layer and connect layer with Dense.
			Connect 128 neurons and rectifier activation
			Connect 10 neurons and rectifier activation (final output)"""
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(7, (2, 2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

"""Assigning the architecture"""
model = CNN_model()

"""Fit all the data into a model"""
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

"""Evaluate the percentage of accuracy on test data"""
score = model.evaluate(X_test, y_test, verbose=0)

print('Percentage of accuracy: %.2f%%' % (score[1]*100))
