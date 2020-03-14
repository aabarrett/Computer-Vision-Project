import os
import cv2
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.chdir('stanford-dogs-dataset/images/Images')
d = '.'
training_names = [os.path.join(d, o) for o in os.listdir(d)
                  if os.path.isdir(os.path.join(d, o))]

images = []
names = []
path = os.getcwd()
new_shape = (128,128,3)

for image_folder in training_names:
    for file_name in os.listdir(image_folder):
        images.append(resize(cv2.imread(path + '/' + image_folder[2:] + '/' + file_name), new_shape))
        names.append(image_folder)

train_x, test_x, train_y, test_y = train_test_split(images, names, test_size=0.2)

# plot dog photos from the dogs vs cats dataset


from matplotlib import pyplot
from matplotlib.image import imread
import os
import numpy as np
batch_size = 128
epochs = 20
learning_rate = 0.001

num_class = y_train_raw.shape[1]

model = Sequential()
model.add(BatchNormalization(input_shape=(128, 128, 3)))
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
model.add(Dense(num_class, activation='softmax'))
model.summary()

# opt = optimizers.Adam(lr = learning_rate)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size,verbose=2, epochs=epochs, shuffle = True)

# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
# https://medium.com/@claymason313/dog-breed-image-classification-1ef7dc1b1967
