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
#credit Benjamin Fogiel


def avg_flux(image, x, y):
	# return the average absolute flux of a nxn square about x,y - assuming n is odd
	# return nan if coordinate is out of bounds
	n = 7
	step = int(n/2)
	# check if x,y are in bounds
	if x >= 3000 or y >= 2000 or x < 0 or y < 0:
		return nan
	x_max = x+step
	x_min = x-step
	y_max = y+step
	y_min = y-step
	# check that box is in bounds, if not crop
	while True:
		if x_max >= 3000:
			x_max -= 1
		elif y_max >= 2000:
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

filepath = 'pair_1'
path1 = '/mnt/annex/ryan/set_8/fits/'
path2 = '/mnt/annex/ryan/set_8/lists/'
f = open(path2 + filepath + '/differences.txt')

###choose the fits file to be opened
image = fits.open(path1 + filepath + '/starfits_1.fits')

### get center coord of transient ###
for i in f:
	x0 = list(i.split())	
	x = x0[2].replace("'","")
	x = float(x.replace(',',''))	
	xc = int(round(x,0))-1	
	y = x0[3].replace("'","")	
	y = float(y.replace(',',''))	
	yc = int(round(y,0))-1
### find vertices of the bounding box ###
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
# SOURCE: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#:~:text=In%20yolo%20%2C%20a%20bounding%20box,x%2D%20and%20y%2Daxis.
				
# add a lot more padding if bounding box is less than 3 pixels
	x_diff = x_max-x_min
	if x_diff < 10:
		x_max = x_max+(10-x_diff)/2 if x_max+(10-x_diff)/2 <= 2999 else 2999
		x_min = x_min-(10-x_diff)/2 if x_min-(10-x_diff)/2 >= 0 else 0
	
	y_diff = y_max-y_min			
	if y_diff < 10:
		y_max = y_max+(10-y_diff)/2 if y_max+(10-y_diff)/2 <= 1999 else 1999
		y_min = y_min-(10-y_diff)/2 if y_min-(10-y_diff)/2 >= 0 else 0
	
	w = (x_max - x_min)
	h = (y_max - y_min)
	w_norm = w/3000
	h_norm = h/2000
	print(w_norm,h_norm)
