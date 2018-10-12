# the file Ir_utils.py and dataset is from www.coursera.org
# Andrew Ng, Co-founder, Coursera; Adjunct Professor, Stanford University;
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from lr_utils import load_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# train_set_x_orig.shape == (209, 64, 64, 3)
# train_set_y.shape == (1, 209)
# test_set_x_orig.shape == (50, 64, 64, 3)
# test_set_y.shape == (1, 50)

# flatten the matrix
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)

# standardize the data
train_set_x = train_set_x_flatten / 255.
train_set_y = np.transpose(train_set_y)
test_set_x = test_set_x_flatten / 255.
test_set_y = np.transpose(test_set_y)

# create the model
model = Sequential()
model.add(Dense(input_dim=12288, output_dim=29))
model.add(Activation('relu'))
model.add(Dense(input_dim=29, output_dim=1))
model.add(Activation('sigmoid'))
# define optimizer
optimizer = SGD(lr=0.01)
model.compile(loss='binary_crossentropy',
              metrics=['accuracy'], optimizer='rmsprop')

x = train_set_x
y = train_set_y

# Fit the model
reslut = model.fit(x=x, y=y, epochs=4000, batch_size=209, validation_data=(test_set_x, test_set_y))
print("Train-Accuracy:", np.mean(reslut.history["acc"]))
print("Test-Accuracy:", np.mean(reslut.history["val_acc"]))
print("OK")