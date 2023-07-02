from PIL import Image
import random
from colorscale import grayscale


# open_in_gray is the only function to open an image as all other functions
# are for grayscale images. run open_in gray on a file, then a function or
# functions on the returned image, then to_image on that return to get a viewable image
def open_in_gray(file):
    img = Image.open(file)
    width, height = img.size
    pix = img.getdata()
    pix_list = []
    for y in range(height):
        row = []
        for x in range(width):
            value = pix[y * width + x]
            row.append(value)
        pix_list.append(row)

    # all functions are on grayscale, so this step is required
    pix_list = grayscale(pix_list)
    return pix_list

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