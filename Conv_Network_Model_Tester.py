import tensorflow as tf
from keras.callbacks import TensorBoard
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import time
from keras import backend as K



train_path = 'cw3-ims/fullfer/train'
test_path = 'cw3-ims/fullfer/test'
fullfer_n_classes = 7
batch_size = 128
epochs = 200

conv_layers = [1,2,3,4]
layer_sizes = [128, 256, 512, 1024]
dense_layers = [0,1]


data_gen = ImageDataGenerator(rescale=1./255,
                             featurewise_std_normalization=True,
                             featurewise_center=True,
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=1,
                             width_shift_range=0.2,
                             height_shift_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255,
                                        featurewise_std_normalization=True,
                                        featurewise_center=True)


train_batches = data_gen.flow_from_directory(train_path, 
                    target_size=(48,48),
                    classes=['angry','disgust','fear','happy','neutral','sad','surprise'],
                    color_mode="grayscale",
                    batch_size=batch_size,
                    shuffle=True) #does this matter?


test_batches = test_datagen.flow_from_directory(test_path, 
                    target_size=(48,48),
                    classes=['angry','disgust','fear','happy','neutral','sad','surprise'],
                    color_mode="grayscale",
                    batch_size=batch_size,
                    shuffle=True)

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = 'Conv-layers{}-num-filters{}-denselayers{}-largeepoch-{}'.format(
                conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME, '\n')
            tensorboard = TensorBoard(log_dir='conv/logs/{}'.format(NAME))
            
            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), padding='same', activation='relu',
                 input_shape=(48, 48,1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3), padding='same', activation='relu'))
                if l % 2 ==0:
                	model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))


            model.add(Dense(fullfer_n_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])

            model.fit_generator(train_batches,
                        steps_per_epoch = 18976 // batch_size,
                        epochs=epochs,
                        validation_data= test_batches,
                        validation_steps=5741 // batch_size,
                        shuffle=True,
                        callbacks=[tensorboard])