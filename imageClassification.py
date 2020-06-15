import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np 
import matplotlib.pyplot as plt

import os
import cv2

Train_Dir = "drive/My Drive/Training"
Val_Dir = "drive/My Drive/Validation"
LabelsList = ["Apple", "Banana", "Lemon", "Lime", "Orange",
               "Strawberry", "Tomato"]

Train_Total = 0
Val_Total = 0 
for label in LabelsList:
  Train_Total = Train_Total + len(os.listdir(os.path.join(Train_Dir, label)))
  Val_Total = Val_Total + len(os.listdir(os.path.join(Val_Dir, label)))

#Variables
img_size = 100
batch_size = 64
epochs = 10
  
#Training images
train_image_generator = ImageDataGenerator(rescale=1./255, rotation_range=45,
                    width_shift_range=.15, height_shift_range=.15,
                    horizontal_flip=True, zoom_range=0.5) 
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=Train_Dir,
                                                           shuffle=True,
                                                           target_size=(img_size, img_size),
                                                           class_mode='sparse')

#Validation Images
validation_image_generator = ImageDataGenerator(rescale=1./255)
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=Val_Dir,
                                                              target_size=(img_size, img_size),
                                                              class_mode='sparse')

#Create Model
model = Sequential([
Conv2D(16, 3, padding='same', activation='relu', input_shape = (img_size, img_size, 3)),
MaxPooling2D(),
Conv2D(32, 3, padding='same', activation='relu'),
MaxPooling2D(),
Dropout(0.4),
Conv2D(64, 3, padding='same', activation='relu'),
MaxPooling2D(),
Conv2D(128, 3, padding='same', activation='relu'),
MaxPooling2D(),
Dropout(0.4),
Flatten(),
Dense(320, activation='relu'),
Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])   

#Early stop to reduce overfitting
callback = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                         patience=5, verbose=1, mode='auto', 
                         restore_best_weights=True)

#Train model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=Train_Total // batch_size,
    epochs=epochs, 
    callbacks=[callback],
    validation_data=val_data_gen,
    validation_steps=Val_Total // batch_size, 
)