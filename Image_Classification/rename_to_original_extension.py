import glob 
import os 
import re 
import logging 
import traceback
from os import listdir
from os.path import isfile, join
import random
import imghdr

mypath = ("/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Original_Data/Travel")


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file_obj in onlyfiles:
    file_type = imghdr.what(mypath + '/' + file_obj)

    if(file_type != 'jpg'):
       # print("Winter")        
        temp = file_obj.replace('jpg', str(file_type))
        os.rename(mypath + '/' + file_obj, mypath + '/' + temp )
