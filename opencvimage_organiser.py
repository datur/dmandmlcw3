#!/usr/bin/env python
# coding: utf-8

# In[19]:

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# In[20]:

print(cv2.__version__)

# In[21]:

training_data = pd.read_csv('csv_files/fer2017-training-happy-updated.csv')
testing_data = pd.read_csv('csv_files/fer2017-testing-happy-updated.csv')

# In[22]:

# In[23]:

training_happy = training_data.loc[training_data['emotion'] == 1].copy()
training_not_happy = training_data.loc[training_data['emotion'] == 0].copy()
testing_happy = testing_data.loc[testing_data['emotion'] == 1].copy()
testing_not_happy = testing_data.loc[testing_data['emotion'] == 0].copy()

training_happy.drop('emotion', axis=1, inplace=True)
training_not_happy.drop('emotion', axis=1, inplace=True)
testing_happy.drop('emotion', axis=1, inplace=True)
testing_not_happy.drop('emotion', axis=1, inplace=True)

train_path_happy = 'happy_ims/train/happy'
train_path_not_happy = 'happy_ims/train/nothappy'
test_path_happy = 'happy_ims/test/happy'
test_path_not_happy = 'happy_ims/test/nothappy'


def image_writer(emotion, fpath):
    print('saving files at:', fpath)
    for i, img in emotion.iterrows():
        img = img.values
        img = img.reshape(48, 48)
        path = os.path.join(fpath, 'image%d.jpg' % i)
        cv2.imwrite(path, img)


# In[36]:

#image_writer(angry, angry_path)

# In[43]:


def test_train_split(df, percentage=0.8):
    mask = np.random.rand(len(df)) < percentage
    train = df[mask]
    valid_mask = np.random.rand(len(train)) < 0.2
    valid = train[valid_mask]
    train = train.drop(valid.index)
    test = df[~mask]
    return train, test, valid


image_writer(training_happy, train_path_happy)
image_writer(training_not_happy, train_path_not_happy)
image_writer(testing_happy, test_path_happy)
image_writer(testing_not_happy, test_path_not_happy)
