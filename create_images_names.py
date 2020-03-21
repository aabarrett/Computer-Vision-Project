import json
import os 
import cv2

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
            
with open('names.json', 'w', encoding='utf-8') as f:
    json.dump(names, f, ensure_ascii=False, indent=4)
    
with open('iamges.json', 'w', encoding='utf-8') as f:
    json.dump(images, f, ensure_ascii=False, indent=4)