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

import sys
np.set_printoptions(threshold=sys.maxsize)

ssl._create_default_https_context = ssl._create_unverified_context

# Setup Model
img_size = 224

resnet = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))

for layer in resnet.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model = Sequential()
model.add(resnet)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

os.chdir('sample-dataset')
d = '.'
training_names = [os.path.join(d, o) for o in os.listdir(d)
                  if os.path.isdir(os.path.join(d, o))]

num_class = 5
imgs = []
names = []
bin_names = []
dog_bins = np.zeros((1,num_class))
path = os.getcwd()
cnt_file = 0
cnt_bin = 0
row_to_be_added = np.zeros((1,num_class))
print('Parsing Files for dataset')
for img_folder in training_names:
    for file_name in os.listdir(img_folder):
        if 'n02' in file_name :
            if cnt_file == 0:
                previous_folder = img_folder
                bin_names.append(img_folder)
            else:
                dog_bins = np.vstack((dog_bins, row_to_be_added))
            img = image.load_img(path + '/' + img_folder[2:] + '/' + file_name, target_size=(img_size, img_size))
            # img = cv2.imread(path + '/' + img_folder[2:] + '/' + file_name)
            if img is None:
                print(file_name)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            imgs.append(img[0])
            # imgs.append(cv2.resize(img, (img_size, img_size)))
            names.append(img_folder)
            if img_folder != previous_folder:
                cnt_bin = cnt_bin+1
                bin_names.append(img_folder)
            dog_bins[cnt_file, cnt_bin] = 1
            previous_folder = img_folder
            cnt_file = cnt_file + 1

print('dog bins')
print(dog_bins)
print('names')
print(names)
print('bin names')
print(bin_names)
#labels_name, labels0 = np.unique(names, return_inverse=True)
x_train_raw = np.array(imgs, np.float32)


train_x, test_x, train_y, test_y = train_test_split(x_train_raw, dog_bins, test_size=0.2)

batch_size = 20
epochs = 20
learning_rate = 0.0001

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

one_up = os.path.abspath(os.path.join(__file__ ,".."))
os.chdir(one_up)

history = model.fit(train_x, train_y, batch_size=batch_size, validation_data=(test_x, test_y), verbose=2, epochs=epochs, shuffle = True)

model.save('sample-test-model.h5')

results = model.evaluate(test_x, test_y, batch_size=batch_size)
print('test loss, test acc:', results)

print('\n# Generate predictions for 20 samples')
predictions = model.predict(test_x[:15])
print('predictions shape:', predictions.shape)


results = model.evaluate(test_x, test_y, batch_size=batch_size)
print('test loss, test acc:', results)

