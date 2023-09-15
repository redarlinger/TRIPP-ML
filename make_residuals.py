import os 
import os.path
import fitstoimg
from read import read  # Import the read function from read.py
from write import write  # Import the write function from write.py
from subtract import subtract #Import the subtract funtion from subtract.py
from fitstoimg import fitstoimg
import subprocess
#credit to Natalie LeBaron for writing original code (many edits)

number_of_files = 1001 #one more than actual


# Define the parent directory for residual_fits
parent_dir = '/mnt/annex/redarlinger/Multi_trans/'

for i in range(1,number_of_files): #Make residuals
    inpath = '/mnt/annex/redarlinger/Multi_trans/sim_fits/pair_'+str(i)+'/'
    outpath = '/mnt/annex/redarlinger/Multi_trans/residual_fits/pair_'+str(i)+''
    hduls = read(inpath)
    hdul=subtract(hduls)    #calling the function in order read, subtract, write, no need to align as the images are simulated
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    write(hdul, outpath)  
    print('Finished processing pair_' + str(i))
    
    


for i in range(1,number_of_files): #Make PNGS
    path = '/mnt/annex/redarlinger/Multi_trans/residual_fits/pair_'+str(i)+''
    fitstoimg(path)
    print('pngmade'+str(i))


for i in range(1,number_of_files): #Move PNGs to residuals folder
    path = '/mnt/annex/redarlinger/Multi_trans/residual_fits/pair_'+str(i)+'/sqrt'
    os.chdir(path)
    respath = '/mnt/annex/redarlinger/Multi_trans/residual_pngs/pair_'+str(i)+''
    os.makedirs(respath)
    os.rename('0.fits.png', respath+'/0.fits.png')
    os.rename('1.fits.png', respath+'/1.fits.png')



