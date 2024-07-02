import os
import random
import shutil

path = "./IluvVNPT"
train_path = "./data_split/train"
validate_path = "./data_split/val"
test_path = "./data_split/test"

for i in os.listdir(path):
    subfolder = path+'/'+i
    img_list = os.listdir(subfolder)
    random.shuffle(img_list)
    l_list = len(img_list)
    train, val, test = img_list[:int(l_list*0.75)], img_list[int(l_list*0.75): int(l_list*0.9)], img_list[int(l_list*0.9):]
    for j in [train_path, validate_path, test_path]:
        os.makedirs(j+'/'+i)
    for j in train:
        img_raw = subfolder+'/'+j
        img_dest = train_path+'/'+i+'/'+j
        shutil.copy(img_raw, img_dest)
    for j in val:
        img_raw = subfolder+'/'+j
        img_dest = validate_path+'/'+i+'/'+j
        shutil.copy(img_raw, img_dest)
    for j in test:
        img_raw = subfolder+'/'+j
        img_dest = test_path+'/'+i+'/'+j
        shutil.copy(img_raw, img_dest)