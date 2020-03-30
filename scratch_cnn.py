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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from matplotlib import pyplot
from matplotlib.image import imread
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import ssl

import sys
np.set_printoptions(threshold=sys.maxsize)

# Setup Model
img_size = 224

model = Sequential()
model.add(BatchNormalization(input_shape=(img_size, img_size, 3)))
model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(29, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

os.chdir('sample-dataset')
d = '.'
training_names = [os.path.join(d, o) for o in os.listdir(d)
                  if os.path.isdir(os.path.join(d, o))]

num_class = 29
imgs = []
names = []
bin_names = []
dog_bins = np.zeros((1,num_class))
path = os.getcwd()
cnt_file = 0
cnt_bin = 0
row_to_be_added = np.zeros((1,num_class))
print('Parsing Files for dataset')
for img_folder in tqdm(training_names):
    for file_name in os.listdir(img_folder):
        if 'n02' in file_name :
            if cnt_file == 0:
                previous_folder = img_folder
                bin_names.append(img_folder)
            else:
                dog_bins = np.vstack((dog_bins, row_to_be_added))
            img = image.load_img(path + '/' + img_folder[2:] + '/' + file_name, target_size=(img_size, img_size))
            if img is None:
                print(file_name)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            imgs.append(img[0])
            names.append(img_folder)
            if img_folder != previous_folder:
                cnt_bin = cnt_bin+1
                bin_names.append(img_folder)
            dog_bins[cnt_file, cnt_bin] = 1
            previous_folder = img_folder
            cnt_file = cnt_file + 1

x_train_raw = np.array(imgs, np.float32)


train_x, test_x, train_y, test_y = train_test_split(x_train_raw, dog_bins, test_size=0.2)

batch_size = 20
epochs = 20
learning_rate = 0.0001

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

one_up = os.path.abspath(os.path.join(__file__ ,".."))
os.chdir(one_up)

history = model.fit(train_x, train_y, batch_size=batch_size, validation_data=(test_x, test_y), verbose=2, epochs=epochs, shuffle = True)

model.save('scratch_cnn_model.h5')

results = model.evaluate(test_x, test_y, batch_size=batch_size)
print('test loss, test acc:', results)

print('\n# Generate predictions for 20 samples')
predictions = model.predict(test_x[:15])
print('predictions shape:', predictions.shape)


results = model.evaluate(test_x, test_y, batch_size=batch_size)
print('test loss, test acc:', results)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()