import cv2
import glob
import pandas as pd

#this is set up for the yolov4 model in luckystar

net = cv2.dnn.readNetFromDarknet(
    '/mnt/annex/rachel/YOLO_data/darknet/cfg/yolov4_stars.cfg', 
    '/mnt/annex/rachel/YOLO_data/darknet/backup/yolov4_stars_last.weights')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

test_file = "/mnt/annex/rachel/YOLO_data/Multi_trans/sets/testMultiTrans.txt"
test_paths = []
with open(test_file) as f:
  test_paths = [l.replace("\n","").replace("/1.png","") for l in f.readlines()]

acc = []
for path in test_paths:
    img_path = path+'/1.png'
    img = cv2.imread(img_path)
    classIds, scores, boxes = model.detect(img, confThreshold=0.1, nmsThreshold=0.1)

    f = open(f"{path}/1.txt","r")
    line = f.read().split()
    f.close()
    ground_truth = 400*float(line[1]), 400*float(line[2])

    if not len(scores):
      acc.append([img_path,img_path.split('/')[9],0,True,ground_truth[0], ground_truth[1],0,0])

    for score,box in zip(scores,boxes):
      center = box[0]+box[2]/2, box[1]+box[3]/2 # box = x-topLeft, y-topLeft, width, height

      false_positive = abs(ground_truth[0] - center[0]) > 20 or abs(ground_truth[1] - center[1]) > 20
      acc.append([
          img_path,
          img_path.split('/')[9],
          score, # confidence
          false_positive, # if deviates more than 20 pixels
          ground_truth[0], # true x-center
          ground_truth[1], # true y-center
          center[0], # predicted x-center
          center[1], # predicted y-center
      ])

pd.DataFrame(
    acc,
    columns=['img','mag_pair','confidence','false pos','true x','true y','pred x','pred y']
    ).to_csv(f'/mnt/annex/rachel/YOLO_data/darknet/performance.csv')
