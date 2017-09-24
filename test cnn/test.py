#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 08:23:34 2017

@author: aakashwadhwa
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#initializing the CNN
classifier = Sequential()

#step-1 convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#step-2 pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step adding second convolutional layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3 Flattening
classifier.add(Flatten())

#step 4 full connection
classifier.add(Dense(activation='relu',output_dim=128))
classifier.add(Dense(activation='softmax',output_dim=3))

#compiling the CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                 'dataset/train',
                                                  target_size=(64, 64),
                                                  batch_size=4,
                                                  class_mode='categorical')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test',
                                            target_size=(64, 64),
                                            batch_size=1,
                                            class_mode='categorical')

classifier.fit_generator(
                        training_set,
                        steps_per_epoch=8,
                        epochs=50,
                        validation_data=test_set,
                        validation_steps=2)

import numpy as np
from keras.preprocessing import image
test_image=image.load_img('dataset/single/l3.jpeg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1 :
      prediction = 'tiger'
else                 :
      prediction = 'leopard'