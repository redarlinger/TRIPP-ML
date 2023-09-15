import glob
import numpy as np
from astropy.io import fits

def read(inpath):
    """
    Takes a directory containing fits files and returns them as a list
    """
    try:
        paths = glob.glob("{}/*.fits*".format(inpath))
        paths = sorted(paths, key=lambda item: int(item[len(inpath):].split('_')[1][:-5]))
        # Create a list to store all HDUs from all files
        hduls = []
        for p in paths:
            hdulist = fits.open(p)
            hduls.extend(hdulist)  # Append all HDUs from this file to the hduls list
        #print(hduls)
        return hduls
    except Exception as e:
        print("Error occurred:", e)
        return []

