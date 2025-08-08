import numpy as np
from build import build_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

basedir=os.path.join(os.path.dirname(os.path.dirname(__file__)),"Data")

train_img=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
    )

test_img=ImageDataGenerator(
    rescale=1./255
)
train_data=train_img.flow_from_directory(
    os.path.join(basedir,"train"),
    target_size=(224,224),
    class_mode="categorical",
    batch_size=32#It will split the images and get trained
    )
test_data=test_img.flow_from_directory(
    os.path.join(basedir,"test"),
    target_size=(224,224),
    class_mode="categorical",
    batch_size=32
    )

model=build_model()
model.fit(train_data,epochs=5)
model.save(os.path.join(basedir,"covid_pneu.h5"))