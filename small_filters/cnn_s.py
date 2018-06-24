# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf

import sys

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop, SGD
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, concatenate
from keras.utils import to_categorical
from keras.models import Model
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

import keras
from keras import backend as K
from sklearn.metrics import f1_score

threshold = 0.6

def accuracy_with_threshold(y_true, y_pred):
	y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
	return K.mean(K.equal(y_true, y_pred))

train_dir = sys.argv[1]
val_dir = sys.argv[2]

# data generator com augmentation - para o treino
datagen_aug = ImageDataGenerator(
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    rescale=1./255,
    rotation_range=2,
    horizontal_flip=False)

# data generator sem o augmentation - para a validação
datagen_no_aug = ImageDataGenerator(rescale=1./255)
#datagen_no_aug = ImageDataGenerator()

# Create the model
input_img = Input(shape=(224,224,3))

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), padding='valid', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(5, 5), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, kernel_size=(5, 5), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(5, 5), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  

model.add(Conv2D(256, kernel_size=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  

model.add(Flatten())

model.add(Dense(1024, activation='selu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='selu'))
#model.add(Dropout(0.5))
#model.add(Dense(128, activation='selu', kernel_initializer='lecun_uniform'))

model.add(Dense(1, activation='sigmoid'))

#opt = RMSprop(lr=0.001, decay=1e-9)
#opt = Adagrad(lr=0.001, decay=1e-6)
#opt = Adadelta(lr=0.075, decay=1e-6)
opt = Adam(lr=0.0001, decay=1e-6)
# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy', accuracy_with_threshold])

checkpoint = ModelCheckpoint('saved_models/model_{epoch:0003d}--{loss:.2f}--{val_loss:.2f}.hdf5',
              save_best_only=True,
              save_weights_only=False)

# treina e valida o modelo - com data augmentation
train_generator = datagen_aug.flow_from_directory(train_dir, target_size=(224,224),
																									batch_size=32,
																									color_mode='rgb',
																									class_mode='binary',
																									seed=7,
																									)
val_generator = datagen_no_aug.flow_from_directory(val_dir, target_size=(224,224),
																									batch_size=32,
																									color_mode='rgb',
																									class_mode='binary',
																									seed=7)

model.fit_generator(
									train_generator,workers=1,
									class_weight={0:1, 1:1}, # balance
									steps_per_epoch=152, # (partition size / batch size)+1
									epochs=500,
									shuffle=True,
                  max_queue_size=20,
									validation_data=val_generator,
									callbacks=[EarlyStopping(patience=20), CSVLogger('training.log', separator=',', append=False), checkpoint])
