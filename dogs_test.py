from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input



model = tf.keras.models.load_model('new_model.h5')
#model = tf.keras.models.load_model('sample-test-model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#img_size = 224
img_size = 128
img = cv2.imread('test_images/pug.jpg')
img = cv2.resize(img,(img_size,img_size))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

classes = model.predict_classes(img)
test = model.predict(img)

print(classes)
print(test)
