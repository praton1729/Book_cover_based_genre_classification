import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import os

img_width, img_height = 150,150
train_data_dir = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/Train'
validation_data_dir = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/Validation/'
save_training_features_path ='features_train.npy'
save_validation_features_path = 'features_validation.npy'
nb_train_samples = 20000    
nb_validation_samples = 5000
top_model_weights_path = 'top_model.h5'
epochs = 30 							#no of training iteration through top level neural network 
batch_size = 8

datagen = ImageDataGenerator(rescale=1. / 255)							#starting the imagedatagenerator and normalizing the pixel values

vgg_model = applications.VGG16(include_top=False, weights='imagenet')	#loading in the VGGnet without the top layer to extract features

#Extract features from training data and store them
generator = datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode=None,shuffle=False)#load images

bottleneck_features_train = vgg_model.predict_generator(generator, ((nb_train_samples // batch_size) )) #extracting the features
np.save(open(save_training_features_path, 'wb'),bottleneck_features_train)							   #saving the features

#Extract features from validation and store them
generator = datagen.flow_from_directory(validation_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode=None,shuffle=False)

bottleneck_features_validation = vgg_model.predict_generator(generator, (nb_validation_samples // batch_size))
np.save(open(save_validation_features_path, 'wb'),bottleneck_features_validation)

#Now we will train the top_model CNN to classify

train_data = np.load(open(save_training_features_path,'rb'))
train_labels = np.array([0]*4000 + [1]*4000 + [2]*4000 + [3]*4000 + [4]*4000 )   

validation_data = np.load(open(save_validation_features_path,'rb'))
validation_labels = np.array([0]*1000 + [1]*1000 +[2]*1000 + [3]*1000 + [4]*1000)

#creating the top NN for classification - architecture of NN
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
