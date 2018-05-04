import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import os

img_width, img_height = 224,224

train_data_dir = '/Data/Training/'

validation_data_dir = 'Data/Validation/'

save_training_features_path ='features_train.npy'

save_validation_features_path = 'features_validation.npy'

nb_train_samples = 7015 + 3548 + 7390 + 4723 + 8599 
nb_validation_samples = 1753 + 880 + 1847 + 1180 + 2149
top_model_weights_path = 'top_model.npy'
subfolder_list = []
train_subfolder_num = []
validation_subfolder_num = []

#for temp in os.listdir(train_data_dir):
#	subfolder_list.append(temp)

#for subfolder_temp in subfolder_list:								#calculating the no of training samples
#	temp_path = train_data_dir + '/' + subfolder_temp
#	train_subfolder_num.append(len(os.listdir(temp_path)))
#	nb_train_samples = nb_train_samples + len(os.listdir(temp_path))
#	
#for subfolder_temp in subfolder_list:

#	temp_path = validation_data_dir + '/' + subfolder_temp
#	validation_subfolder_num.append(len(os.listdir(temp_path)))
#	nb_validation_samples = nb_train_samples + len(os.listdir(temp_path))	 
#	

epochs = 35 							#no of training iteration through top level neural network 
batch_size = 16

datagen = ImageDataGenerator(rescale=1. / 255)							#starting the imagedatagenerator and normalizing the pixel values

vgg_model = applications.VGG16(include_top=False, weights='imagenet')	#loading in the VGGnet without the top layer to extract features

#Extract features from training data and store them
generator = datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode=None,shuffle=False)#load images

bottleneck_features_train = vgg_model.predict_generator(generator, (nb_train_samples // batch_size) ) #extracting the features
np.save(open(save_training_features_path, 'wb'),bottleneck_features_train)							   #saving the features

#Extract features from validation and store them
generator = datagen.flow_from_directory(validation_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode=None,shuffle=False)

bottleneck_features_validation = vgg_model.predict_generator(generator, (nb_validation_samples // batch_size))
np.save(open(save_validation_features_path, 'wb'),bottleneck_features_validation)

#Now we will train the top_model CNN to classify

train_data = np.load(open(save_training_features_path,'rb'))
train_labels = np.array([0]*7015 + [1]*3548 + [2]*7390 + [3]*4723 + [4]*8599 ) #automate this, making the labels, last wala 2132 hona chaiye, abhi temporarily change kiya hai    

validation_data = np.load(open(save_validation_features_path,'rb'))
validation_labels = np.array([0] *1753 + [1]*880 +[2]*1847 + [3]*1180 + [4]*2149) #automate this changed last waala 533 to 530 which is wrong

#creating the top CNN for classification - architecture of CNN
top_model = Sequential()
top_model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))
top_model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training the top model 
top_model.fit(train_data[:nb_train_samples,:,:,:], train_labels,epochs=epochs,batch_size=batch_size,validation_data=(validation_data[:nb_validation_samples,:,:,:], validation_labels))

#saving the top model weights    
top_model.save_weights(top_model_weights_path)
