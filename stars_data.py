#this is to incorperate the single transient data in with the multiple transient data

import glob
import random
import os

#assuming working dir is 'stars_data/' # ##for collab
#assert os.getcwd() == '/content/drive/My Drive/Darknet Training Sets/darknet/stars_data/', f"you're in the wrong dir (current dir: {os.getcwd()})"

types=glob.glob(f'/mnt/annex/YOLO_data/darknet/Multi_trans/stars_data/*')
paths = []
abs_dir = '/mnt/annex/YOLO_data/darknet/Multi_trans/'

for t in types:
    dif = glob.glob(f'{t}/*')
    for d in dif:
        pairs=glob.glob(f'{d}/*')
        for p in pairs:
             paths.append(f'{abs_dir}{p}/1.fits.png')

# scamble paths so we can randomly assign training and val sets
random.shuffle(paths)

# 70% training 15% val 15% test
cut_train = int(len(paths)*0.7)
cut_val= int(len(paths)*0.85)

train = paths[:cut_train] #takes the first 70%
val = paths[cut_train:cut_val] #takes the next 15%
test = paths[cut_val:] #takes the last 15%

# check len of each set
assert len(val)+len(train)+len(test)==len(paths), 'val and training sets are of incorrrect lengths'



with open(os.path.join(abs_dir, "sets/trainMultiTrans.txt"), "a") as textfile:
    for element in train:
        textfile.write(element + "\n")

with open(os.path.join(abs_dir, "sets/valMultiTrans.txt"), "a") as textfile:
    for element in val:
        textfile.write(element + "\n")

with open(os.path.join(abs_dir, "sets/testMultiTrans.txt"), "a") as textfile:
    for element in test:
        textfile.write(element + "\n")

textfile.close()
