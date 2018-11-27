#Neural_Network_Model_Tester

import tensorflow as tf
from keras.callbacks import TensorBoard
import pandas as pd
import keras
import itertools
import numpy as np

from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import time

np.random.seed(42)

# training data
print('loading training data...')
train_data = pd.read_csv('csv_files/fer2017-training.csv')
#print(train_data.head())
train_labels = train_data['emotion'].copy()
train_data.drop('emotion', axis=1, inplace=True)

train_n_samples, n_features = train_data.shape
n_classes = len(set(train_labels))

X_train = train_data.values
y_train = train_labels.values

print('done')

# test data
print('loading testing data...')
test_data = pd.read_csv('csv_files/fer2017-testing-happy-updated.csv')
test_labels = test_data['emotion'].copy()

test_data.drop('emotion', axis=1, inplace=True)
test_n_samples, n_features = test_data.shape

X_test = test_data.values
y_test = test_labels.values

# normalise
X_test = X_test / 255
X_train = X_train / 255

# encode classes
y_train_labels = keras.utils.to_categorical(y_train, num_classes=n_classes)
y_test_labels = keras.utils.to_categorical(y_test, num_classes=n_classes)

print('done')

# iterate through different model structures
layers = [5]
layer_sizes = [32, 64, 128] # update to 128,256,512

for layer in layers:
    nodes = list(itertools.product(layer_sizes, repeat=layer))
    for node in nodes:
        NAME = 'mlp-{}--hiddenlayers-{}-nodes--{}'.format(
            layer, node, int(time.time()))
        tensorboard = TensorBoard(log_dir='mlp/logs/{}'.format(NAME))

        print('Current Model:\t', NAME)

        model = Sequential()

        # input layer
        model.add(
            Dense(n_features, input_shape=X_train[0].shape, activation='relu'))

        #hidden layers
        for n in node:
            model.add(Dense(n, activation='relu'))

        # output layers
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        model.fit(
            X_train,
            y_train_labels,
            epochs=50,
            verbose=1,
            batch_size=128,
            validation_data=(X_test, y_test_labels),
            callbacks=[tensorboard])
