import cv2
import glob
import pandas as pd

#this is set up for the yolov4 model in luckystar

net = cv2.dnn.readNetFromDarknet(
    'cfg/yolov4_stars.cfg', 
    'backup/yolov4_stars_multi.weights')


model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

test_file = "Multi_trans/sets/testMultiTrans.txt"
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
      
    for score, box in zip(scores, boxes):
        center = box[0] + box[2] / 2, box[1] + box[3] / 2  # box = x-topLeft, y-topLeft, width, height
        for line in lines:
            ground_truth = [400 * float(line.split()[1]), 400 * float(line.split()[2])]  # getting the true locations
            if abs(ground_truth[0] - center[0]) <= 20 and abs(ground_truth[1] - center[1]) <= 20:
                false_positive = False #checking to see if any of the predictions match with this ground truth (checking for false negatives)
                acc.append([
                 img_path,
                 score,  # using the score for the entry
                 false_positive,  # if deviates more than 20 pixels
                 ground_truth[0],  # true x-center
                 ground_truth[1],  # true y-center
                 center[0],  # predicted x-center
                 center[1],  # predicted y-center
                 ])
                break
        if center[0] not in [entry[5] for entry in acc]:
            false_positive=True
            acc.append([
                 img_path,
                 score,  # using the score for the entry
                 false_positive,  # if deviates more than 20 pixels
                 "falsepos",  # true x-center
                 "falsepos",  # true y-center
                 center[0],  # predicted x-center
                 center[1],  # predicted y-center
                 ])
    for line in lines:
        ground_truth = [400 * float(line.split()[1]), 400 * float(line.split()[2])]
        # Check if either x-coordinate or y-coordinate is in the accumulator (no match with any detection)
        if ground_truth[0] not in [entry[3] for entry in acc] or img_path not in [entry[0] for entry in acc]:
            false_positive=True
            acc.append([
                 img_path,
                 "not_found",  # using the score for the entry
                 false_positive,  # if deviates more than 20 pixels
                 ground_truth[0],  # true x-center
                 ground_truth[1],  # true y-center
                 "falseneg",  # predicted x-center
                 "falseneg",  # predicted y-center
                 ])
            

        

pd.DataFrame(
    acc,
    columns=['img', 'confidence', 'false pos', 'true x', 'true y', 'pred x', 'pred y']
).to_csv(f'/mnt/annex/rachel/YOLO_data/darknet/performance.csv')




