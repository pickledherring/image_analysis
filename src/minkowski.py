from quantization import hist_thresh
import math

# Uses Minkowski addition to add pixels in a binarized image
# Weights should be 0 for no addition in this pixel relative to the
# center pixel or 1 for addition, matrix should be square
def dilate(img, weights):
    # use a square filter with 0 for negative (white) and 1 for positive (black)
    length = len(img)
    width = len(img[0])
    mask_radius = math.floor(len(weights) / 2)
    binary = hist_thresh(img)
    new_img = []
    for i in range(length):
        new_img.append([])
        for j in range(width):
            new_img[i].append(255)

    for i in range(length):
        for j in range(width):
            if binary[i][j] == 0:
                for m in range(-mask_radius, mask_radius + 1):
                    for n in range(-mask_radius, mask_radius + 1):
                        if weights[m+mask_radius][n+mask_radius] == 1\
                            and i + m >= 0 and i + m < length\
                            and j + n > 0 and j + n < width:
                            new_img[i+m][j+n] = 0

    return new_img

# The opposite of dilate(). If all of the 1s in the weight matrix are hits,
# keep the pixel at that location, otherwise, remove it.
def erode(img, weights):
    # use a square filter with 0 for negative (white) and 1 for positive (black)
    length = len(img)
    width = len(img[0])
    mask_radius = math.floor(len(weights) / 2)
    binary = hist_thresh(img)
    new_img = []

    for i in range(length):
        new_img.append([])
        for j in range(width):
            new_img[i].append(255)

    for i in range(length):
        for j in range(width):
            present = True
            for m in range(-mask_radius, mask_radius + 1):
                for n in range(-mask_radius, mask_radius + 1):
                    if weights[m+mask_radius][n+mask_radius] == 1\
                        and i + m >= 0 and i + m < length\
                        and j + n > 0 and j + n < width:
                        if binary[i+m][j+n] != 0:
                            present = False
                            break
                if not present:
                    break
            if present:    
                new_img[i][j] = 0

    return new_img