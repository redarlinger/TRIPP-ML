import ois #Bramich subtraction
from astropy.io import fits
from astropy.io.fits import CompImageHDU


outputs=[]
def subtract(images):
    image1=images[0]
    image2=images[1]
    result_images = []  # Create a new list to store the modified HDUs
    for i, hdu in enumerate(images):
        try:
            diff = ois.optimal_system(image=image2.data, refimage=image1.data, method='Bramich')[0]
        except ValueError:
            diff = ois.optimal_system(image=image2.data.byteswap().newbyteorder(), refimage=image1.data.byteswap().newbyteorder(), method='Bramich')[0]
        new_hdu = fits.CompImageHDU(data=diff, header=images[i].header, name="SUB")
        # Append the new HDU to the result_images list
        result_images.append(new_hdu)
    return result_images
