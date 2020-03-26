import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.stats import itemfreq
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, UpSampling2D
# from tensorflow.keras.preprocessing.image import imgDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from matplotlib import pyplot
from matplotlib.image import imread
import os
import numpy as np
import ssl
import csv

ssl._create_default_https_context = ssl._create_unverified_context

def matrix_Bin(labels):
    labels_bin=np.array([])
    labels_name, labels0 = np.unique(labels, return_inverse=True) # find unique elements in array of image folders. There will be 120 (num of breeds)
    labels0 # indices of unique array

    for _, i in enumerate(itemfreq(labels0)[:,0].astype(int)):
        labels_bin0 = np.where(labels0 == itemfreq(labels0)[:,0][i], 1., 0.) # function returns the indices of elements in an input array where the given condition is satisfied.
        labels_bin0 = labels_bin0.reshape(1,labels_bin0.shape[0])
        if (labels_bin.shape[0] == 0):
            labels_bin = labels_bin0
        else:
            labels_bin = np.concatenate((labels_bin,labels_bin0 ),axis=0)
    print("Nber SubVariables {0}".format(itemfreq(labels0)[:,0].shape[0]))
    labels_bin = labels_bin.transpose()
    print("Shape : {0}".format(labels_bin.shape))

    return labels_name, labels_bin

# Setup Model
img_size = 128

# resnet = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
#
# for layer in resnet.layers:
#     if isinstance(layer, BatchNormalization):
#         layer.trainable = True
#     else:
#         layer.trainable = False
#
# model = Sequential()
# model.add(resnet)
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(120, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

os.chdir('sample-dataset')
d = '.'
training_names = [os.path.join(d, o) for o in os.listdir(d)
                  if os.path.isdir(os.path.join(d, o))]

imgs = []
names = []
path = os.getcwd()
print('Parsing Files for dataset')
for img_folder in training_names:
    for file_name in os.listdir(img_folder):
        if 'n02' in file_name :
            img = image.load_img(path + '/' + img_folder[2:] + '/' + file_name, target_size=(img_size, img_size))
            img = cv2.imread(path + '/' + img_folder[2:] + '/' + file_name)
            if img is None:
                print(file_name)
            #print(file_name)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            imgs.append(img[0])
            # imgs.append(cv2.resize(img, (img_size, img_size)))
            names.append(img_folder)

#for x in range(names)
   # y = names[x]

   # if y = np.unique(dog_names, return_inverse=True)
#dataFrame = pd.DataFrame(data = dog_names)


# x_train_raw = np.array(imgs, np.float32) / 255
x_train_raw = np.array(imgs)#,np.float32)

num_class = 120

labels_name, labels_bin = matrix_Bin(names)
print(labels_bin)
print(labels_name[1])
print('done')
