from astropy.io import fits
from glob import glob
import os
import itertools
from astropy.modeling import models, fitting
import math
import numpy as np
import warnings
from PIL import Image
from math import nan
import itertools
from statistics import mean
import shutil
from PIL import Image
from os.path import exists


###this code requires that you run starlist_analysis.py on the set that you're planning to make labels for before it will work. this code needs the differences.txt file that comes from starlist_analysis.py to function properly
###another note: please make sure that there are an equal number of pairs of pngs, fits files, and lists

def transient_positions(filepath): #filepath input is a string
	path1 = '/mnt/annex/redarlinger/Multi_trans/residual_pngs/' #this is where the pngs are located
	path2 = '/mnt/annex/ryan/outbox/Multi_trans_analysis/set_4/lists/' #this is where the differences.txt files are located
	ans = [] #this function will eventually return a list of lists, with an x and y value corresponding to each of the transients in the image [[x1,y1],[x2,y2]...]	
	temp = [] #a temporary variable used to store variables until they are manipulated and put into ans
	width, height = Image.open(path1 + filepath + '/0.fits.png').size #get the size of the image in pixels
	
	#now we need to open the corresponding differences.txt file to the png	
	f = open(path2 + filepath + '/differences.txt')
	for i in f: #this block of code isolates the x and y positions of the transients in each png
		x0 = list(i.split())	
		x1 = x0[2].replace("'","")
		x1 = x1.replace(',','')	
		x1 = float(x1)	
		x2 = x0[3].replace("'","")	
		x2 = x2.replace(',','')	
		x2 = float(x2)	
		temp.append([x1,x2])
	
	for i in temp: #now we take the x and y positions of each of the transients and position them relative to the size of the image
		j1 = i[0]
		j2 = i[1]
		x = j1/width
		y = 1 - j2/height
		ans.append([x,y])
	f.close()
	return(ans)

def avg_flux(image, x, y):
	# return the average absolute flux of a nxn square about x,y - assuming n is odd
	# return nan if coordinate is out of bounds
	n = 7
	step = int(n/2)
	# check if x,y are in bounds
	if x >= 400 or y >= 400 or x < 0 or y < 0:
		return nan
	x_max = x+step
	x_min = x-step
	y_max = y+step
	y_min = y-step
	# check that box is in bounds, if not crop
	while True:
		if x_max >= 400:
			x_max -= 1
		elif y_max >= 400:
			y_max -= 1
		elif x_min < 0:
			x_min += 1
		elif y_min < 0:
			y_min += 1
		else:
			break

	# get avg flux
	x_vals = list(range(x_min, x_max+1))
	y_vals = list(range(y_min, y_max+1))
	fluxes = []
	for coord in list(itertools.product(x_vals,y_vals)):
		fluxes.append(abs(image[0].data[coord[1]][coord[0]]))
	return mean(fluxes)





path1 = '/mnt/annex/redarlinger/Multi_trans/sim_fits/' #where the fits files are located
path2 = '/mnt/annex/ryan/outbox/Multi_trans_analysis/set_4/lists/' #where the lists that correspond to the fits files are located
number_of_files = 1001 #edit here to change total number of files that need labels 

for i in range(1, number_of_files):
	labels = []	
	filepath = 'pair_' + str(i)
	f = open(path2 + filepath + '/differences.txt')
	
	###get the <x> and <y> values for the label
	xypositions = transient_positions(filepath)

	###choose the fits file to be opened
	image = fits.open(path1 + filepath + '/starfits_1.fits')

	### get center coord of transient ###
	whvalues = []	
	for j in f: ##this part of the code is based on the formatting of the difference.txt file, if you end up using a different format you will have to change this section accordingly to get the coordinates properly
		x0 = list(j.split())	
		x = x0[2].replace("'","")
		x = float(x.replace(',',''))	
		xc = int(round(x,0))-1	
		y = x0[3].replace("'","")	
		y = float(y.replace(',',''))	
		yc = int(round(y,0))-1
	###xc and yc correspond to the center coordinates of the transient
	
	### find vertices of the bounding box ###
	###the vertices of the box will always start at xc and yc. the average flux around said centerpoint is then calculated. if it is above a certain flux threshold, then that coordinate will expand outward by 1 pixel and check the flux of that pixel. this will repeat until the flux is found to be below said threshold, and will be considered the size of the transient
		x_min = xc; x_max = xc; y_min = yc; y_max = yc 
		f_xmax = avg_flux(image,xc+1,yc)
		f_ymax = avg_flux(image,xc,yc+1)
		f_xmin = avg_flux(image,xc-1,yc)
		f_ymin = avg_flux(image,xc,yc-1)
		flux_thresh = 1000
		while True:
			if not math.isnan(f_xmax) and f_xmax > flux_thresh:
				x_max += 1
				f_xmax = avg_flux(image,x_max+1,yc)
			elif not math.isnan(f_ymax) and f_ymax > flux_thresh:
				y_max += 1
				f_ymax = avg_flux(image,xc,y_max+1)
			elif not math.isnan(f_xmin) and f_xmin > flux_thresh:
				x_min -= 1
				f_xmin = avg_flux(image,x_min-1,yc)
			elif not math.isnan(f_ymin) and f_ymin > flux_thresh:
				y_min -= 1
				f_ymin = avg_flux(image,xc,y_min-1)
			else:
				break

	### get width, height, and center ###
	# SOURCE: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/	#:~:text=In%20yolo%20%2C%20a%20bounding%20box,x%2D%20and%20y%2Daxis.
				
	# add a lot more padding if bounding box is less than 5 pixels
		x_diff = x_max-x_min
		if x_diff < 5:
			x_max = x_max+(5-x_diff)/2 if x_max+(5-x_diff)/2 <= 399 else 399
			x_min = x_min-(5-x_diff)/2 if x_min-(5-x_diff)/2 >= 0 else 0
	
		y_diff = y_max-y_min			
		if y_diff < 5:
			y_max = y_max+(5-y_diff)/2 if y_max+(5-y_diff)/2 <= 399 else 399
			y_min = y_min-(5-y_diff)/2 if y_min-(5-y_diff)/2 >= 0 else 0
	
		w = (x_max - x_min)
		h = (y_max - y_min)
		##we need the ratio of the width and height of the transient to the overall size of the image
		w_norm = w/400
		h_norm = h/400
		whvalues.append([w_norm, h_norm])
		image.close()

	##here we create the labels that will eventually be written to a txt file
	if len(xypositions) == len(whvalues):
		for k in range(0,len(xypositions)):
			xy = xypositions[k]
			wh = whvalues[k]
			label = '0 ' + str(xy[0]) + ' ' + str(xy[1]) + ' ' + str(wh[0]) + ' ' + str(wh[1])
			labels.append(label)
	else:
		print(filepath + ' has an error in with its data')
		
	###write the labels to a txt file in the same directory where the differences txt file is located

	os.chdir('/mnt/annex/ryan/outbox/Multi_trans_analysis/set_4/lists/' + filepath)##this needs to put the label in the same file as its png
	file_exists = exists('/mnt/annex/ryan/outbox/Multi_trans_analysis/set_4/lists/' + filepath + '/1.txt')
	if file_exists == False:
		with open('1.txt', 'w') as a:
			for y in labels:
				a.write(str(y) + '\n')
	elif file_exists == True:
		os.remove(path2 + filepath + '/1.txt')
		with open('1.txt', 'w') as a:
			for y in labels:
				a.write(y + '\n')
