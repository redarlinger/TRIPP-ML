import cv2
import glob
import pandas as pd

#this is set up for the yolov4 model in luckystar

net = cv2.dnn.readNetFromDarknet(
    'cfg/yolov4_stars.cfg', 
    'backup/yolov4_stars_multi.weights')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

test_file = "Multi_trans/sets/testMultiTransnew.txt"
test_paths = []
with open(test_file) as f:
  test_paths = [l.replace("\n","").replace("/1.png","") for l in f.readlines()] #taking off the 1.png so it can see both png and text


acc = []
for path in test_paths:
    img_path = path + '/1.png'
    img = cv2.imread(img_path)
    classIds, scores, boxes = model.detect(img, confThreshold=0.1, nmsThreshold=0.1)  # model detecting transient

    f = open(f"{path}/1.txt", "r")
    lines = f.readlines()
    f.close()
    difference= len(scores)-len(lines)
    if difference>0: #see if the model found more than the correct transients (checking for false positives)
	for i in range(difference):
	    acc.append([
             img_path,
             0,  # using the score for the entry
             True,  # if deviates more than 20 pixels
             0,  # true x-center
             0,  # true y-center
             "false positive",  # predicted x-center
             "false positive",  # predicted y-center
            ])


    for line in lines:
        ground_truth = [400 * float(line.split()[1]), 400 * float(line.split()[2])]  # getting the true locations
        false_positive = True
        center1=[0,0]
        score1=[]
      
        for score, box in zip(scores, boxes):
            center = box[0] + box[2] / 2, box[1] + box[3] / 2  # box = x-topLeft, y-topLeft, width, height

            if abs(ground_truth[0] - center[0]) <= 20 and abs(ground_truth[1] - center[1]) <= 20:
                false_positive = False #checking to see if any of the predictions match with this ground truth (checking for false negatives)
                center1=center
                score1=score

        acc.append([
             img_path,
             score1,  # using the score for the entry
             false_positive,  # if deviates more than 20 pixels
             ground_truth[0],  # true x-center
             ground_truth[1],  # true y-center
             center1[0],  # predicted x-center
             center1[1],  # predicted y-center
            ])

pd.DataFrame(
    acc,
    columns=['img', 'confidence', 'false pos', 'true x', 'true y', 'pred x', 'pred y']
).to_csv(f'/mnt/annex/rachel/YOLO_data/darknet/performance.csv')

