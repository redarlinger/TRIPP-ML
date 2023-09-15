import os
import numpy as np
from PIL import Image
from astropy.io import fits
import img_scale
#from astropy.utils.data import download_file
#image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True )


def fitstoimg(dir_path):
    os.chdir(dir_path)
    os.makedirs(dir_path+'/'+'sqrt'+'/')
    for filename in os.listdir(dir_path):
        if filename.endswith(".fits"):
            image_data = fits.getdata(filename)
            if len(image_data.shape) == 2:
                sum_image = image_data
            else:
                sum_image = image_data[0] - image_data[0]
                for single_image_data in image_data:
                    sum_image += single_image_data  

            sum_image = img_scale.sqrt(sum_image, scale_max=np.amax(image_data))
            sum_image = sum_image * 200
            im = Image.fromarray(sum_image)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(dir_path+'/'+'sqrt'+'/'+filename+".png")
            im.close()

