# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:05:22 2021

@author: lyndo
"""

import tensorflow as tf

#from tensorflow import keras
from keras import layers
from keras.models import Sequential

#we are using Google Colab to import data from our Google Drive folder. 
from google.colab import drive
drive.mount('/content/gdrive')

import pathlib
path_to_data = '' #provide path to the training data here

data = pathlib.Path(path_to_data)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names   #the training data should be split into 18 directories, one for each category
#class_names = ['accommodation', 'animals', 'beach', 'building_hist', 'building_modern', 'desert', 'entertainment', 'gastronomy', 'landscape', 'monument', 'mountains', 'museum', 'plantsflowers', 'road_traffic', 'shops_markets', 'sport', 'trees', 'water']

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 18

#BASELINE MODELS (skip for the final model)

#the 'vanilla' model - a 3 layer CNN - can be used to get a baseline accuracy on the training data
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  layers.MaxPooling2D((2,2)),
  layers.Flatten(),
  layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              #loss='binary_crossentropy',  #trying to do multi-label classification
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.summary()

epochs=10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#you can now evaluate the model using the test data split or on other data such as the YFCC100M dataset

#refinements - augmentation, dropout, learning rate annealment, transfer learning

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Flatten(),
  layers.Dense(128),
  layers.Dense(num_classes)
])

from tensorflow.keras.callbacks import ReduceLROnPlateau

lrr = ReduceLROnPlateau(monitor='val_accuracy', factor=.01, patience=10, min_lr=1e-5)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.summary()

epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[lrr]
)

#you can now evaluate the model using the test data split or on other data such as the YFCC100M dataset

#FINAL MODEL

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras.models import Model

#Defining the Convolutional Neural Net for Transfer Learning
base_model = InceptionResNetV2(include_top = False, weights = 'imagenet', pooling='max', input_shape = (img_height, img_width,3), classes = class_names)
base_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
#base_model.trainable = False

model= Sequential()
model.add(data_augmentation)
model.add(layers.Rescaling(1./255))
model.add(base_model)  
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.1))
model.add(Flatten()) 
model.add(Dense(128,activation=('relu'), kernel_initializer='he_uniform'))
model.add(Dense(num_classes,activation=('softmax'))) 

#model.summary()

batch_size= 100
epochs=6
learn_rate=.001 

from tensorflow.keras.optimizers import SGD, Adam

sgd=SGD(learning_rate=learn_rate,momentum=.9,nesterov=False)
adam=Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=sgd, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#Optional: save weights after each epoch at checkpoints
path_to_checkpoint = ''
ckpt = pathlib.Path(path_to_checkpoint)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt,
                                                 save_weights_only=True,
                                                 verbose=1)

#Training the model
model.fit(train_ds, epochs = epochs, validation_data = val_ds, callbacks=[cp_callback]) 

#Optional: load weights from past training instead of retraining the model
model.load_weights(path_to_checkpoint)

#you can now evaluate the model using the test data split or on other data such as the YFCC100M dataset


