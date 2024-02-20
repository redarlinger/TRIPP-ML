import cv2
import glob
import pandas as pd
import numpy as np
import sep
import fitsio
import math
#just needs to compare the number of found stars to the number of stars actually in the photo, and then see if the ones it found are even correct in the first place

#this is set up for the yolov4 model in luckystar

#net = cv2.dnn.readNetFromDarknet(
    #'/mnt/annex/rachel/YOLO_data/darknet/cfg/yolov4_stars.cfg', 
    #'/mnt/annex/rachel/YOLO_data/darknet/backup/yolov4_stars_last.weights')

#model = cv2.dnn_DetectionModel(net)
#model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

test_file = "/mnt/annex/ryan/testMultiTrans.txt"
test_paths = []
with open(test_file) as f: #append each path in the txt file to a list
  for l in f.readlines():
    l2 = "/mnt/annex/YOLO_data/darknet/" + l
    test_paths.append(l2.replace('\n',''))
f.close()
count = 1
num_right = 0 #number correctly found
num_found = 0 #total number found
num_fp = 0 #number of false positives
num_fn = 0 #number of false negatives
acc = [] #this list will store the accuracies that will eventually be put into the csv file
for path in test_paths:
    acc1 = [] #store accuracies here first so that I can ad the percentage to the first one, then to acc
    sep_centers = []
    #edit the paths to their correct locations
    if "original" in path or "stars_data" in path:
      fits_path = path.replace('_pngs','_fits').replace('/original','').replace('1.png','1.fits')
      if "stars_data" in path:
        if "one_mag" in path:
          fits_path = fits_path.replace('YOLO_data/darknet/Multi_trans/stars_data/one_mag','natalie/NewMLSimulations/residual/1mag/1sec').replace('pair_','residual_fits/Bramich/pair_')
        if "half_mags" in path:
          fits_path = fits_path.replace('YOLO_data/darknet/Multi_trans/stars_data/half_mags','natalie/NewMLSimulations/residual/half_mag/1sec').replace('pair_','residual_fits/Bramich/pair_')
        if "quarter_mags" in path:
          fits_path = fits_path.replace('YOLO_data/darknet/Multi_trans/stars_data/quarter_mags','nicole/NewMLSimulations/residual/quart_mag/1sec').replace('pair_','residual_fits/Bramich/pair_')

      image1 = abs(fitsio.read(fits_path)) 
      bkg = sep.Background(image1) #this does not perform background subtraction, just returns a representation of spatially variable image background and noise
      objects = sep.extract(image1, 1.5, err=bkg.globalrms*5) #sep.extract will find the objects in the fits file given the array image1 as an input. bkg.globalrms is a number, not an array
#reference https://sep.readthedocs.io/en/stable/api/sep.extract.html#sep.extract for information on what the different indices of the list that sep.extract returns refer to
#format is indices 3,4,5,6 for xmin, xmax, ymin, ymax
      for i in objects:
          xcen_sep = (i[4]+i[3])/2
          ycen_sep = (i[6]+i[5])/2
          sep_centers.append((xcen_sep,ycen_sep)) #this makes sep_centers a list of the centers of all the stars found by sep.extract
      num_found = num_found + len(objects)

#reword the png paths to the locations of the labels
      txt_path = path.replace('1.png','1.txt')
      

#now we have the path to each label that corresponds to the fits file of each line in testMultiTrans.txt. since some of these have multiple objects, we will have to run a for loop through all of them and compare to the results found in the objects variable
      lines = []
      with open(txt_path) as f: #this will take each line of the 1.txt file, split them into a list, then append said list into a list called lines
        for l in f.readlines():
          lines.append(l.split())
      ground_truths = [] #the lists in the lines variable will then be used to calculate the pixel locations of the objects in the image using the size of the image
      for i in lines:
        ground_truths.append((400*float(i[1]), 400*(1-float(i[2])))) #getting the true locations assuming a 400x400 pixel image



#check to see if sep.extract found any objects. if not, it should report the position of the objects it was supposed to find, and that they were not found
      if len(sep_centers) == 0: 
        for ground_truth in ground_truths:
          false_positive = True #object not found
          num_fn = num_fn + 1
          acc1.append([
              path,
              false_positive,
              ground_truth[0], # true x-center
              ground_truth[1], # true y-center
              0, # x-center not found
              0, # y-center not found
          ])
      elif len(sep_centers) == len(ground_truths): #check which objects were found. start by comparing the lengths of sep_centers and ground_truths. if they are equal, then we just compare and match them
        for center in sep_centers:
          possible_matches = [] #possible matches will be stored as [ground_truth index, center index]
          distances = [] #stores the corresponding distance of the possible pairs
          for ground_truth in ground_truths:
            distances.append(math.sqrt((float(ground_truth[0]) - float(center[0]))**2 + (float(ground_truth[1]) - float(center[1]))**2))
            possible_matches.append((ground_truths.index(ground_truth), sep_centers.index(center)))
            best_match = distances.index(min(distances)) #using the distances between the possible pairs, find the closest one
            match = possible_matches[best_match]
            false_positive = abs(ground_truths[match[0]][0] - sep_centers[match[1]][0]) > 10 or abs(ground_truths[match[0]][1] - sep_centers[match[1]][1]) > 10 #check for a false positive, if the object is too far away from the known center, then it is labeled a false positive
            distance = min(distances)
            false_negative = abs(min(distances)) > 25 #check for a false negative, if the object is a distance above a certain threshold from the known center, then it is a false negative
        #this series of if statements is just to make sure that false negatives are not double coutned as both a false negative and a false positive
          if false_positive == False:
            num_right = num_right + 1
          elif false_positive == True:
            num_fp = num_fp + 1
          if false_negative == True: 
            num_fn = num_fn + 1
          acc1.append([
              path,
              false_positive,
              ground_truths[match[0]][0], # true x-center
              ground_truths[match[0]][1], # true y-center
              sep_centers[match[1]][0], # x-center found
              sep_centers[match[1]][1], # y-center found
          ])
          ground_truths.pop(match[0]) #remove the ground truth that matched the center so that it won't get called on again for the other centers
      elif len(sep_centers) > len(ground_truths): #if sep.extract finds more objects than in the image, then we first find all the pairs, then write the rest as false positives
        while len(ground_truths) > 0:
          for ground_truth in ground_truths:
            possible_matches = [] #possible matches will be stored as [ground_truth index, center index]
            distances = [] #stores the corresponding distance of the possible pairs
            for center in sep_centers:
              distances.append(math.sqrt((float(ground_truth[0]) - float(center[0]))**2 + (float(ground_truth[1]) - float(center[1]))**2))
              possible_matches.append((ground_truths.index(ground_truth), sep_centers.index(center)))
            best_match = distances.index(min(distances)) #using the distances between the possible pairs, find the closest one
            match = possible_matches[best_match]
            false_positive = abs(ground_truths[match[0]][0] - sep_centers[match[1]][0]) > 10 or abs(ground_truths[match[0]][1] - sep_centers[match[1]][1]) > 10
            distance = min(distances)
            false_negative = abs(min(distances)) > 25
            if false_positive == False:
              num_right = num_right + 1
            elif false_positive == True:
              num_fp = num_fp + 1
            if false_negative == True: 
              num_fn = num_fn + 1
            acc1.append([
                path,
                false_positive,
                ground_truths[match[0]][0], # true x-center
                ground_truths[match[0]][1], # true y-center
                sep_centers[match[1]][0], # x-center found
                sep_centers[match[1]][1], # y-center found
            ])
            ground_truths.pop(match[0]) #remove the ground truth of this pair so that it won't get called on again for the other centers
            sep_centers.pop(match[1]) #remove the center of this pair so that the remaining values in sep_center will be marked as extra
        for center in sep_centers: #for the remaining objects in sep_centers, append to acc as false positives
          false_positive = True #no such object
          num_fp = num_fp + 1
          acc1.append([
              path,
              false_positive,
              0, # no x-center
              0, # no y-center
              center[0], # x-center found
              center[1], # y-center found
          ])
      elif len(sep_centers) < len(ground_truths): #if sep.extract finds more objects than in the image, then we first find all the pairs, then write the rest as false positives
        while len(sep_centers) > 0:
          for center in sep_centers:
            possible_matches = [] #possible matches will be stored as [ground_truth index, center index]
            distances = [] #stores the corresponding distance of the possible pairs
            for ground_truth in ground_truths:
              distances.append(math.sqrt((float(ground_truth[0]) - float(center[0]))**2 + (float(ground_truth[1]) - float(center[1]))**2))
              possible_matches.append((ground_truths.index(ground_truth), sep_centers.index(center)))
            best_match = distances.index(min(distances)) #using the distances between the possible pairs, find the closest one
            match = possible_matches[best_match]
            false_positive = abs(ground_truths[match[0]][0] - sep_centers[match[1]][0]) > 10 or abs(ground_truths[match[0]][1] - sep_centers[match[1]][1]) > 10
            distance = min(distances)
            false_negative = abs(min(distances)) > 25
            if false_positive == False:
              num_right = num_right + 1
            elif false_positive == True:
              num_fp = num_fp + 1
            if false_negative == True: 
              num_fn = num_fn + 1
            acc1.append([
                path,
                false_positive,
                ground_truths[match[0]][0], # true x-center
                ground_truths[match[0]][1], # true y-center
                sep_centers[match[1]][0], # x-center found
                sep_centers[match[1]][1], # y-center found
            ])
            ground_truths.pop(match[0]) #remove the ground truth of this pair so that it won't get called on again for the other centers
            sep_centers.pop(match[1]) #remove the center of this pair so that the remaining values in sep_center will be marked as extra
        for ground_truth in ground_truths: #for the remaining objects in ground_truths not found by sep.extract, append to acc as false positives
          false_positive = True #object not found
          num_fn = num_fn + 1
          acc1.append([
              path,
              false_positive,
              ground_truth[0], # true x-center
              ground_truth[1], # true y-center
              0, # x-center not found
              0, # y-center not found
          ])

    

      for a in acc1:
        acc.append(a)
      print('file ' + str(count) + ' completed')
      count = count +1

num_fp = num_fp - num_fn #correct for double counting

accuracy = num_right/num_found
precision = num_right / (num_right + num_fp)
recall = num_right / (num_right + num_fn)
F1 = 2*precision*recall / (precision + recall)

for a in acc:
  if acc.index(a) == 0:
    a.append(accuracy)
    a.append(num_fp)
    a.append(num_fn)
    a.append(precision)
    a.append(recall)
    a.append(F1)
  else:
    a.append('')   
    a.append('')
    a.append('')
    a.append('')
    a.append('')
    a.append('')

pd.DataFrame(
    acc,
    columns=['img','false pos','true x','true y','pred x','pred y', 'overall accuracy', 'total false positives', 'total false negatives', 'precision', 'recall', 'F1']
    ).to_csv(f'/mnt/annex/ryan/performance.csv')

print('total found:' + str(num_found))
print('total fp:' + str(num_fp))
print('total fn:' + str(num_fn))
print('total correct:' + str(num_right))
