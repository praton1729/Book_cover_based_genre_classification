from PIL import Image
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing import image

img_width, img_height = 150, 150

top_model_weights_path = '/home/sherlock31/Desktop/winter/bottleneck_fc_model_3.h5'
train_data_dir = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/Train'
validation_data_dir = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/Validation'


train_data = np.load(('features_train.npy'))
top_model = Sequential()
top_model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))

top_model.load_weights('top_model.h5')

directory_in_str = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Demo_Data'
directory = os.fsencode(directory_in_str)
vgg_model = applications.VGG16(include_top=False, weights='imagenet')

for file in os.listdir(directory):
	filename = os.fsdecode(file)
	img_path = directory_in_str + '/' + filename
	img = image.load_img(img_path, target_size=(150,150))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = x*(1/255.0)
	features = vgg_model.predict(x)
	print(top_model.predict_classes(features))
