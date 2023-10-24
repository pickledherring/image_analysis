from PIL import Image, ImageFile
import numpy as np
from colorscale import grayscale
ImageFile.LOAD_TRUNCATED_IMAGES = True

# open_in_gray is the only function to open an image as all other functions
# are for grayscale images. run open_in gray on a file, then a function or
# functions on the returned image, then to_image on that return to get a viewable image
def open_in_gray(file):
    img = Image.open(file)
    img_t = np.asarray(img, dtype=np.float32)
    weights = np.array([.2126, .7152, .0722], dtype=np.float32)
    # all functions are on grayscale, so this step is required
    pix_t = np.dot(img_t, weights)
    return pix_t

def to_image(pix_list, save_loc=None):
    size = [len(pix_list[0]), len(pix_list)]
    flat_list = []
    for row in pix_list:
        for i in range(len(row)):
            value = row[i]
            flat_list.append(value)
    output = Image.new("L", size)
    output.putdata(flat_list)
    
    if save_loc:
        output.save(f"{save_loc}")
    return output