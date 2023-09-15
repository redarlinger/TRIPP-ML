import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

parent_dir = '/mnt/annex/redarlinger/'

imgdata = fits.getdata('/mnt/annex/redarlinger/Multi_trans/residual_fits/pair_4/0.fits')
plt.imshow(imgdata, cmap= 'gray') #this is how you logscale matplotlib, vmin and vmax behave like the upper and lower histogram bounds in DS9
plt.show()
