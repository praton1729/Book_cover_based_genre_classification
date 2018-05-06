from PIL import Image
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.preprocessing import image

#This code outputs predictions for all the images present in the folder "directory_in_str"


img_width, img_height = 150, 150

train_data_dir = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/Train'
validation_data_dir = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/Validation'

train_data = np.load(('features_train.npy'))
top_model = Sequential()
top_model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))

top_model.load_weights('top_model.h5')

directory_in_str = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Demo_Data'	#Test Folder 
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
	temp = top_model.predict_classes(features)

	if(temp == [0]):
		temp2 = "Food"
	if(temp == [1]):
		temp2 = "Romance"
	if(temp == [2]):
		temp2 = "Science"
	if(temp == [3]):
		temp2 = "Sports"
	if(temp == [4]):
		temp2 = "Travel"

	print(temp2)
