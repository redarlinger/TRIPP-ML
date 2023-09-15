import glob
import numpy as np
from astropy.io import fits

def write(hdul, outpath):
    """
    Writes all given hduls into the directory specified
    """
    try:
        import os
        # Check if directory exists or create it if it doesn't
        if not os.path.exists(outpath):
            os.makedirs(outpath)
            print('Directory was not present, now created at ' + outpath)
        for i, hdu in enumerate(hdul):
            data_hdu = hdu
            path = os.path.join(outpath, "{number}.fits".format(number=i))
            data_hdu.writeto(path)
        print("Files written successfully.")
        return True
    except Exception as e:
        print("Error occurred:", e)
        return False
