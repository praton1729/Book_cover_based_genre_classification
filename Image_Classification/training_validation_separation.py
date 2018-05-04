from os import listdir
from os.path import isfile, join
import random
import os

mypath = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/preliminary_test/Training/Test_Preparation'

dest_path = '/media/balraj/6A2CEF0A2CEECFDD/Acads/SEMESTER_8/EE769_Machine_Learning/Project/Data/preliminary_test/Validation/Test_Preparation'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mov_list = random.sample(onlyfiles, 534)

for  i in mov_list:
    os.rename(mypath + '/' + i, dest_path + '/' + i)
