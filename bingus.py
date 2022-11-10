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



model = load_model("this-shit.h5")