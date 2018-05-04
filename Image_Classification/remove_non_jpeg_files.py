import glob 
import os 
import re 
import logging 
import traceback
from os import listdir
from os.path import isfile, join
import random
import imghdr

mypath = ("/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/preliminary_test/Training/Comics_&_Graphic_Novels")


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file_obj in onlyfiles:
    file_type = imghdr.what(mypath + '/' + file_obj)

    if(file_type != 'jpeg'):
       # print("Winter")        
        os.remove(mypath + '/' + file_obj)

