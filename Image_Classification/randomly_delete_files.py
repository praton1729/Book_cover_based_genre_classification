from os import listdir
from os.path import isfile, join
import random
import os
mypath = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Original_Data/Travel'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

del_list = random.sample(onlyfiles, 8000)

for  i in del_list:
      
    os.remove(mypath + '/' + i)

