from PIL import Image as ImagePil
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import os
import cv2
import pickle
from pathlib import Path


DATADIR = r"C:\Users\Utilisateur\Desktop\dataset"
def getText(dir):
    img = ImagePil.open(dir)
    tags = str(img.text).replace("}", "").replace("'", "").split(";")
    tags.pop(0)
    if('barbe' not in tags):
        tags.append("pas de barbe")
    if('lunettes' not in tags):
        tags.append("pas de lunettes")
    return tags
IMG_SIZE = 50
training_data = []
files = Path(DATADIR).glob('*')
for file in files:
    #file = Path(file)
    filename = os.path.join(file)
    tags = getText(file)
    img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # convert to array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
    training_data.append([img_array, tags])  # add this to our training_data


x = []
y = []

for feature, label in training_data:
    x.append(feature)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



HERE = Path(__file__).parent


pickle_out = open(HERE / "x.pickle","wb")
serialized = pickle.dumps(x)
with open(HERE / "x.pickle","wb") as file_object:
    file_object.write(serialized)
pickle_out.close()

pickle_out = open(HERE / "y.pickle","wb")
serialized = pickle.dumps(y)
with open(HERE / "y.pickle","wb") as file_object:
    file_object.write(serialized)
pickle_out.close()