from os import listdir
from os.path import isfile, join
import random
import os
mypath = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/preliminary_test/Training/Comics_&_Graphic_Novels'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for i in onlyfiles:
    if(os.stat(mypath + '/' + i).st_size == 0):
        os.remove(mypath + '/' +i)

