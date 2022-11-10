
from PIL import Image, ExifTags



import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
from pathlib import Path

DATADIR = r"C:\Users\Utilisateur\Desktop\faces\metadata"



def getExifdata(file):
    img = Image.open(file)
    exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
    keywords = str(exif["XPKeywords"]).replace(r"\x00", "").replace("b'", '').replace("'", '')
    keywords = keywords.split(';')
    return keywords


IMG_SIZE = 50

training_data = []
for img in os.listdir(DATADIR):  # iterate over each image per dogs and cats
    #img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
    currentImage = os.path.join(DATADIR, img)
    currentTags = getExifdata(os.path.join(DATADIR,img))
    img_array = cv2.imread(currentImage, cv2.IMREAD_GRAYSCALE)  # convert to array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
    training_data.append([new_array, currentTags])  # add this to our training_data
    #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #plt.imshow(new_array, cmap='gray')

    break  # we just want one for now so break
#break  #...and one more!





print(len(training_data))

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