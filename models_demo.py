import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.stats import itemfreq
from tensorflow.keras.models import Sequential, Model, load_model
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
from tqdm import tqdm
import os
import numpy as np
import ssl
import matplotlib.pyplot as plt

def getTestResults(file_name, model):
    img_size = 224
    img = cv2.imread(file_name)
    # img = image.img_to_array(img)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    results = model.predict(img)

    return sorted(zip(results.tolist()[0], bin_names), reverse=True)[:3]


def displayResults(results_list_tuples):
    for index, result in enumerate(results_list_tuples):
        print('[' + str(index + 1) +'] ' + result[1].split('-')[1] + ': ' + str(result[0] * 100) + '%')
    print()

os.chdir('sample-dataset')
d = '.'
training_names = [os.path.join(d, o) for o in os.listdir(d)
                  if os.path.isdir(os.path.join(d, o))]

num_class = 29
imgs = []
names = []
bin_names = []
path = os.getcwd()
cnt_file = 0
cnt_bin = 0
for img_folder in tqdm(training_names):
    for file_name in os.listdir(img_folder):
        if 'n02' in file_name :
            if cnt_file == 0:
                previous_folder = img_folder
                bin_names.append(img_folder)
            if img_folder != previous_folder:
                cnt_bin = cnt_bin+1
                bin_names.append(img_folder)
            previous_folder = img_folder
            cnt_file = cnt_file + 1

os.chdir('../')

scratch_model = load_model('scratch_cnn_model.h5')

african_hunting_dog_test = getTestResults('./test_images/african_hunting_dog.jpg', scratch_model)

print('===== Printing Test Results For Scratch CNN =====')

print('Top Three predictions for african hunting dog')
displayResults(african_hunting_dog_test)

newfoundland = getTestResults('./test_images/newfoundland.jpg', scratch_model)

print('Top Three predictions for Newfoundland')
displayResults(newfoundland)

chow_test = getTestResults('./test_images/chow.jpg', scratch_model)

print('Top Three Predictions for Chow')
displayResults(chow_test)

leonburg_test = getTestResults('./test_images/leonberg.jpg', scratch_model)
print('Top Three Predictions for Leonberg')
displayResults(leonburg_test)

afhgan_test = getTestResults('./test_images/afhgan.jpg', scratch_model)
print('Top Three Predictions for Afhgan')
displayResults(afhgan_test)

print('===== Printing Test Results For Transfer Learning CNN =====')

transfer_model = load_model('resnet_cnn_model.h5')

african_hunting_dog_test = getTestResults('./test_images/african_hunting_dog.jpg', transfer_model)

print('Top Three predictions for african hunting dog')
displayResults(african_hunting_dog_test)

newfoundland = getTestResults('./test_images/newfoundland.jpg', transfer_model)

print('Top Three predictions for Newfoundland')
displayResults(newfoundland)

chow_test = getTestResults('./test_images/chow.jpg', transfer_model)

print('Top Three Predictions for Chow')
displayResults(chow_test)

leonburg_test = getTestResults('./test_images/leonberg.jpg', transfer_model)
print('Top Three Predictions for Leonberg')
displayResults(leonburg_test)

afhgan_test = getTestResults('./test_images/afhgan.jpg', transfer_model)
print('Top Three Predictions for Afhgan')
displayResults(afhgan_test)