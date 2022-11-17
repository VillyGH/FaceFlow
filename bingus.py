import os

import numpy as np
import tensorboard as tensorboard
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pickle
import time

NAME = "Cats-vs-dogs-CNN"

pickle_in = open("x.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))



checkpoint_path = "Cats-vs-dogs-CNN/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*batch_size)


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

X= np.array(X)
y = np.array(y)
print(model.layers)
model.fit(X, y,
          batch_size=32,
          epochs=0,
          validation_split=0.3,
          callbacks=[cp_callback])
print(model.layers)

#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#predictions = probability_model.predict(test_images)
'''

import  os as os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



import pickle
from pathlib import Path

HERE = Path(__file__).parent

X = pickle.load(open(HERE / "x.pickle","rb"))
y = pickle.load(open(HERE / "y.pickle","rb"))
print("allo",X)

X = X/255.0



model = load_model("this-shit.h5")'''