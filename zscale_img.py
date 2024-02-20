import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize
import fitsio
import os
from PIL import Image
import cv2

#/mnt/annex/YOLO_data/darknet/Multi_trans/residual_fits/

file_path = "/mnt/annex/ryan/LCO/LCO_set8_resfits/"
pair_num = 28925

while pair_num < 30001:
	fits_path = file_path + 'pair_' + str(pair_num) + '/1.fits' # This is where the original fits files are
	
	# Create a sample astronomical image
	image_data = abs(fitsio.read(fits_path))

	# Use ZScaleInterval to determine intensity scaling
	interval = ZScaleInterval()
	vmin, vmax = interval.get_limits(image_data)

	# Create an ImageNormalize object using the determined scaling
	norm = ImageNormalize(vmin=vmin, vmax=vmax)
	 
	# Display the image using matplotlib
	plt.figure(figsize=(30,20)) #figsize should be the desired amount of pixels in one dimension divided by 100
	plt.imshow(image_data, cmap='grey', origin='lower', norm=norm)
	plt.axis('off')

	# Save the image to its new locations
	png_path = "/mnt/annex/ryan/LCO/LCO_set8_pngs/"
	destination = png_path + 'pair_' + str(pair_num) + '/1.png'
	plt.savefig(destination, dpi=10000/77, bbox_inches='tight', pad_inches=0) 
	#NOTE: the size of the image will be the dimensions of figsize multiplied by the dpi, but the numbers are not exact. a dimension of 30 multiplied by a dpi of 10000/77 will be 3000 pixels
	plt.close()
	
	print(str(pair_num) + " done")
	pair_num = pair_num + 1


