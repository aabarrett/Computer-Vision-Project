import os
import cv2
from sklearn.model_selection import train_test_split

os.chdir('stanford-dogs-dataset/images/Images')
d = '.'
training_names = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]

images = []
names = []
path = os.getcwd()

for image_folder in training_names:
    for file_name in os.listdir(image_folder):
        images.append(cv2.imread(path + '/' + image_folder[2:] + '/' + file_name))
        names.append(image_folder)
        
train_x, test_x, train_y, test_y = train_test_split(images, names, test_size=0.2)

print(train_x)
