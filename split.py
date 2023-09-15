"""**Creating the label directory list**"""

import random
import glob
import os

#assuming working dir is 'stars_data/' # ##for collab
#assert os.getcwd() == '/content/drive/My Drive/Darknet Training Sets/darknet/stars_data/', f"you're in the wrong dir (current dir: {os.getcwd()})"

paths = []
abs_dir = '/mnt/annex/YOLO_data/darknet/Multi_trans/'

pairs = glob.glob(f'{abs_dir}residual_pngs/*')
for p in pairs:
    paths.append(f'{p}/1.fits.png')

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


# write txts
#define the path for the "sets" directory
sets_dir=os.path.join(abs_dir, 'sets')

#Check if the "sets" directory exists, and if not, create it
if not os.path.exists(sets_dir):
    os.makedirs(sets_dir)

with open(os.path.join(sets_dir, "trainMultiTrans.txt"), "w") as textfile:
    for element in train:
        textfile.write(element + "\n")

with open(os.path.join(sets_dir, "valMultiTrans.txt"), "w") as textfile:
    for element in val:
        textfile.write(element + "\n")

with open(os.path.join(sets_dir, "testMultiTrans.txt"), "w") as textfile:
    for element in test:
        textfile.write(element + "\n")

textfile.close()
