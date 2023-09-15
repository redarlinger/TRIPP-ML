from astropy.io import fits
from glob import glob
import os
import itertools
from astropy.modeling import models, fitting
import math
import numpy as np
import warnings
import img_scale
from PIL import Image
from math import nan
import itertools
from statistics import mean
import cv2
import shutil
#credit Benjamin Fogiel

nat_dir = "/mnt/annex/nicole/NewMLSimulations/residual/quart_mag/"
my_dir = "/mnt/annex/bfogiel/ML/quarter_mag/"

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
		fluxes.append(abs(image[1].data[coord[1]][coord[0]]))
	return mean(fluxes)

exposures = glob(f"{nat_dir}/*")
for e in exposures:
	# skip exposure times thsat we don't want
	if '10sec' not in e:
		continue
	## create directory, replace if it exists
	#exp_dir = e.replace(nat_dir,my_dir)
	#if os.path.exists(exp_dir):
    	#	shutil.rmtree(exp_dir)
	#os.mkdir(exp_dir)

	mags = glob(f"{e}/*")
	for mag in mags:
		#if "190_195" in mag or "195_200" in mag or "200_205" in mag:
		#	continue
		#if not "1325_1350" in mag:
		#	continue
		# create directory, replace if it exists
		mag_dir = mag.replace(nat_dir,my_dir)
		if os.path.exists(mag_dir):
    			shutil.rmtree(mag_dir)
		os.mkdir(mag_dir)

		pairs = glob(f"{mag}/residual_fits/Bramich/*")
		for i,pair in enumerate(pairs):
			# check if fits exist in the pair
			try:
				image = fits.open(f'{pair}/0.fits')
			except:
				print(f'no fits in {pair}')
				continue
			# create directory, replace if it exists
			my_pair_path = pair.replace(nat_dir,my_dir).replace('/residual_fits/Bramich','')
			if os.path.exists(my_pair_path):
    				shutil.rmtree(my_pair_path)
			os.mkdir(my_pair_path)

			### get center coord of transient ###
			x,y = map(float,image[1].header['TRANSLOC'].strip('()').split(','))
			xc, yc = (int(round(x,0))-1, int(round(y,0))-1)

			### find vertices of the bounding box ###
			flux_c = abs(image[1].data[yc][xc])
			x_min = xc; x_max = xc; y_min = yc; y_max = yc
			f_xmax = avg_flux(image,xc+1,yc)
			f_ymax = avg_flux(image,xc,yc+1)
			f_xmin = avg_flux(image,xc-1,yc)
			f_ymin = avg_flux(image,xc,yc-1)
			#flux_thresh = flux_c*0.025 if flux_c*0.025 > 28 else 28 # 2 standard deviations (assuming normally distributed), below 28 is considered background noise
			flux_thresh = 50
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
				
			# add a lot more padding if bounding box is less than 10 pixels
			x_diff = x_max-x_min
			if x_diff < 10:
				x_max = x_max+(10-x_diff)/2 if x_max+(10-x_diff)/2 <= 399 else 399
				x_min = x_min-(10-x_diff)/2 if x_min-(10-x_diff)/2 >= 0 else 0
			y_diff = y_max-y_min			
			if y_diff < 10:
				y_max = y_max+(10-y_diff)/2 if y_max+(10-y_diff)/2 <= 399 else 399
				y_min = y_min-(10-y_diff)/2 if y_min-(10-y_diff)/2 >= 0 else 0

			w = (x_max - x_min)
			h = (y_max - y_min)
			w_norm = w/400
			h_norm = h/400	
			x_norm = (x_max+x_min)/2/400
			y_norm = 1-(y_max+y_min)/2/400 # origin (0,0) is top left of image

			### write to txt ###
			label = open(f'{my_pair_path}/1.txt', 'w')
			label.write(f'0 {x_norm} {y_norm} {w_norm} {h_norm}')
			label.close()
			

						
			### insert png ###
			#fit_paths = [f'{pair}/0.fits', f'{pair}/1.fits'] !!replace naming below to use this!!
			fit_paths = [f'{pair}/1.fits'] # just get 1.fits
			for i,filename in enumerate(fit_paths):
				if filename.endswith(".fits"):
					image_data = fits.getdata(filename)
					if len(image_data.shape) == 2:
						sum_image = image_data
					else:
						sum_image = image_data[0] - image_data[0]
						for single_image_data in image_data:
							sum_image += single_image_data  
					sum_image = img_scale.log(sum_image, scale_min=0, scale_max=np.amax(image_data))
					sum_image = sum_image * 200
					im = Image.fromarray(sum_image)
					if im.mode != 'RGB':
						im = im.convert('RGB')
					#im.save(f'{my_pair_path}/{i}.png')
					im.save(f'{my_pair_path}/1.png')
					im.close()
					# flip image about the horizontal axis bc it's inverted
					image= cv2.imread(f'{my_pair_path}/1.png')
					flippedimage= cv2.flip(image, 0)
					cv2.imwrite(f'{my_pair_path}/1.png',flippedimage)
			
					
			

		



	
	
	



