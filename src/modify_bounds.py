def crop(img, amount):
    # amount is width in pixels to crop from all edges
    length = len(img)
    width = len(img[0])
    new_img = []
    for i in range(length - 2 * amount):
        new_img.append([])
        for j in range(width - 2 * amount):
            value = img[amount+i][amount+j]
            new_img[i].append(value)

    return new_img

def pad(img, amount):
    # amount is width in pixels to pad on all edges
    length = len(img)
    width = len(img[0])
    new_img = []
    for i in range(length + 2 * amount):
        new_img.append([])
        for j in range(width + 2 * amount):
            if i >= length + amount or j >= width + amount\
                or i < amount or j < amount:
                value = 255
            else:
                value = img[i-amount][j-amount]
            new_img[i].append(value)

    return new_img

# Filtering operations:
# *ext_bounds helps the linear filters by extending the image past
#  its edges using a wraparound.
# *Linear filter with user-specified mask size and pixel weights
# *Median filter with user-specified mask size and pixel weights
# Gaussian 3x3 and 5x5 filters already created for easy use!
def ext_bounds(img, length, width, mask_size):
    # works well when there is no border
    new_img = []
    for i in range(mask_size):
        new_img.append([])
        # top left corner
        for j in range(mask_size):
            new_img[i].append(img[length-mask_size+i][width-mask_size+j])
        # top center
        for j in range(width):
            new_img[i].append(img[length-mask_size+i][j])
        # top right corner
        for j in range(mask_size):
            new_img[i].append(img[length-mask_size+i][j])
    for i in range(length):
        new_img.append([])
        # middle left
        for j in range(mask_size):
            new_img[i+mask_size].append(img[i][width-mask_size+j])
        # middle center
        for j in range(width):
            new_img[i+mask_size].append(img[i][j])
        # middle right
        for j in range(mask_size):
            new_img[i+mask_size].append(img[i][j])
    for i in range(mask_size):
        new_img.append([])
        # bottom left corner
        for j in range(mask_size):
            new_img[i+length+mask_size].append(img[i][width-mask_size+j])
        # bottom center
        for j in range(width):
            new_img[i+length+mask_size].append(img[i][j])
        # bottom right corner
        for j in range(mask_size):
            new_img[i+length+mask_size].append(img[i][j])

    return new_img
