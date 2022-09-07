# import the necessary packages


# Normal libraries
import pandas as pd
import numpy as np
import os
import shutil

# pickle
import pickle as pk

#
from IPython.display import Image

# visualization libraries
import matplotlib.pyplot as plt

# sklearn libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# sklearn tensorflow.keras  libraries
from tensorflow.keras.models  import Sequential, load_model
from tensorflow.keras  import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import binary_crossentropy, categorical_crossentropy
from tensorflow.keras import optimizers


import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import to_categorical


with open('/Users/amateos88/Downloads/healthy.npy', 'rb') as f:
    healthy = np.load(f)
with open('/Users/amateos88/Downloads/sick.npy', 'rb') as f:
    sick = np.load(f)
sick.shape
n_samples = 500

X_train_healthy = healthy[0:n_samples]
X_train_sick = sick[0:n_samples]
X_train = tf.convert_to_tensor(np.concatenate((X_train_healthy, X_train_sick)))
len(X_train)
y_train = []

for i in range(0, n_samples):
    y_train.append(0)

for i in range(0, n_samples):
    y_train.append(1)

y_train_healthy = 0

y_train = tf.convert_to_tensor(y_train)
y_train
n_test = int(n_samples*0.2)

X_test_healthy = healthy[n_samples:(n_samples + n_test)]
X_test_sick = sick[n_samples:(n_samples + n_test)]
X_test = tf.convert_to_tensor(np.concatenate((X_test_healthy, X_test_sick)))
X_test.shape
y_test = []

for i in range(0, n_test):
    y_test.append(0)

for i in range(0, n_test):
    y_test.append(1)
y_test = tf.convert_to_tensor(y_test)
# ——— CNN Version 1: Minimal Network
model = Sequential()
model.add(layers.Conv2D(50, (5,5), input_shape=(224, 224, 3), padding='same', activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(32, (5,5), padding='same', activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, (5,5), padding='same', activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(20, activation='relu')) # intermediate layer
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20,  # Use early stopping in practice
          batch_size=8,
          validation_split= 0.2,
          shuffle= True,
          callbacks=[es],
          verbose=1)
plt.imshow(sick[4])

# let's create a function to access the data




# create a function to reoder the data into anaemia and healthy
# since the data is already resized it will not need to be resized.
# let's split the data into train, test and validation
