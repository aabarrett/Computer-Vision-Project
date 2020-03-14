import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from matplotlib.image import imread
import os
import numpy as np
import pickle

os.chdir('stanford-dogs-dataset/images/Images')
d = '.'
training_names = [os.path.join(d, o) for o in os.listdir(d)
                  if os.path.isdir(os.path.join(d, o))]

images = []
names = []
path = os.getcwd()
image_size = 128

for image_folder in training_names:
    for file_name in os.listdir(image_folder):
        if 'n02' in file_name :
            image = cv2.imread(path + '/' + image_folder[2:] + '/' + file_name)
            if image is None:
                print(file_name)
            images.append(cv2.resize(image, (image_size, image_size)))
            names.append(image_folder)

y_train_raw = np.array(names, np.uint8)
x_train_raw = np.array(images, np.float32) / 255.
num_class = 120 #y_train_raw.shape[1]

train_x, test_x, train_y, test_y = train_test_split(x_train_raw, y_train_raw, test_size=0.2)

# plot dog photos from the dogs vs cats dataset

batch_size = 128
epochs = 20
learning_rate = 0.001


model = Sequential()
model.add(BatchNormalization(input_shape=(image_size, image_size, 3)))
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
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=batch_size,verbose=2, epochs=epochs, shuffle = True)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
