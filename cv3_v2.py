
# coding: utf-8

# # Convolutional neural network - Fer2018

# In[2]:


import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


# #### variables

# In[3]:


train_path = 'ims/train'
valid_path = 'ims/valid'
test_path = 'ims/test'

n_classes = 7
batch_size = 128
epochs = 10


# ## create data generator
# 
# using ImageDataGenerator to normalise the images by dividing the images by 255
# 
# using the custome image data gen:
# - create batches for train, test and validation
#   - denotes classes, image shape batch size
#   - train is for training
#   - validation is for evaluating each epoch
#   - test is for evaluating the whole model

# In[4]:


data_gen = ImageDataGenerator(rescale=1./255,
                             featurewise_std_normalization=True,
                             featurewise_center=True,
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=1,
                             width_shift_range=0.2,
                             height_shift_range=0.2)

validation_datagen = ImageDataGenerator(rescale=1./255,
                                        featurewise_std_normalization=True,
                                        featurewise_center=True)

# In[5]:


train_batches = data_gen.flow_from_directory(train_path, 
                    target_size=(48,48),
                    classes=['angry','disgust','fear','happy','neutral','sad','surprise'],
                    color_mode="grayscale",
                    batch_size=128,
                    shuffle=True) #does this matter?

test_batches = validation_datagen.flow_from_directory(test_path, 
                    target_size=(48,48),
                    classes=['angry','disgust','fear','happy','neutral','sad','surprise'],
                    color_mode="grayscale",
                    batch_size=128,
                    shuffle=True)

validation_batches = validation_datagen.flow_from_directory(valid_path,
                    target_size=(48,48),
                    classes=['angry','disgust','fear','happy','neutral','sad','surprise'],
                    color_mode="grayscale",
                    batch_size=128,
                    shuffle=False)


# # Model
# 
# The code block below is the definition of the model 

# In[12]:


model = Sequential()

model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                 input_shape=(48, 48,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


# In[13]:


model.fit_generator(train_batches,
                    steps_per_epoch = 18976 // batch_size,
                    epochs=50,
                    validation_data= validation_batches,
                    validation_steps=5741 // batch_size,
                    shuffle=True)

