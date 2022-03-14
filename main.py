import glob
import matplotlib.pyplot as plt
import math
import statistics as st
import random
import time
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Noise addition functions: 
# * Salt and pepper noise of user-specified strength 
# * Gaussian noise of user-specified parameters
def salt_n_pepper(img, prob):
    # prob should be int [0 - 1], recommend <.1
    new_img = []
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            if random.random() <= prob:
                if random.choice(["black", "white"]) == "black":
                    new_img[i].append(0)
                else:
                    new_img[i].append(255)
            else:
                new_img[i].append(img[i][j])
    return new_img
    
def gaussian_noise(img, mu = 0, sigma = 1):
    new_img = []
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            new_val = img[i][j] + random.gauss(mu, sigma)
            if new_val < 0:
                new_img[i].append(0)
            elif new_val > 255:
                new_img[i].append(1)
            else:
                new_img[i].append(new_val)

    return new_img

# Converting color images to selected single color spectrum. 
def grayscale(img):
    new_img = []
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            # CIE recommended constants
            grey = .2126 * img[i][j][0] + .7152 * img[i][j][1]+ .0722 * img[i][j][2]
            new_img[i].append(grey)
    return new_img

# open_in_gray is the only function to open an image as all other functions
# are for grayscale images. run open_in gray on a file, then a function or
# functions on the returned image, then to_image on that return to get a viewable image
def open_in_gray(file):
    img = Image.open(file)
    width, height = img.size
    pix = img.getdata()
    pix_list = []
    for y in range(height):
        pix_list.append([pix[y * width + x] for x in range(width)])

    # all functions are on grayscale, so this step is required
    pix_list = grayscale(pix_list)
    return pix_list

def to_image(pix_list, save_loc=None):
    size = [len(pix_list[0]), len(pix_list)]
    flat_list = []
    for row in pix_list:
        for pixel in row:
            flat_list.append(pixel)
    output = Image.new("L", size)
    output.putdata(flat_list)
    
    if save_loc:
        output.save(f"{save_loc}")
    return output

# Histogram calculation for each individual image
def hist(img, bins = 255):
    bin_divs = []
    counts = []
    for i in range(1, bins + 1):
        bin_divs.append(i * 255/bins)
        counts.append(0)

    for i in range(len(img)):
        for j in range(len(img[i])):
            # better as a binary search
            for k in range(bins):
                if img[i][j] <= bin_divs[k]:
                    counts[k] += 1
                    break
    
    return bin_divs, counts

def plot_hist(bin_divs, counts, title=None):
    half_width = bin_divs[0]/2
    plt.bar([x - half_width for x in bin_divs], counts, width = half_width*2)
    plt.xlabel("intensity")
    plt.ylabel("count")
    if title:
            titles = {"cyl": "columnar epithelial",
                    "para": "parabasal squamous epithelial",
                    "inter": "intermediate squamous epithelial",
                    "super": "superficial squamous epithelial",
                    "let": "mild nonkeratinizing dysplastic",
                    "mod": "moderate nonkeratinizing dysplastic",
                    "svar": "severe nonkeratinizing dysplastic"}
            plt.title(titles[title])
    plt.show()

# Averaged histograms of pixel values for each class of images
# hist_avg_class can be run independently of open_in_gray as it runs it internally
def hist_avg_class(folder, abbr, bins = 255):
    # abbr can be:
    # "cyl": columnar epithelial?
    # "para": parabasal squamous epithelial
    # "inter": intermediate squamous epithelial
    # "super": superficial squamous epithelial
    # "let": mild nonkeratinizing dysplastic?
    # "mod": moderate nonkeratinizing dysplastic
    # "svar": severe nonkeratinizing dysplastic?
    files = glob.glob(f"{folder}/{abbr}*")

    bin_divs = [i * 255/bins for i in range(1, bins + 1)]
    counts = [0 for _ in range(bins)]
    # open and grayscale each file
    for file in files:
        pix_list = open_in_gray(file)
        # get and add counts
        _, ind_counts = hist(pix_list, bins = bins)
        counts = [a + b for a, b in zip(counts, ind_counts)]

    # average counts
    counts = [x / len(files) for x in counts]

    return bin_divs, counts

# Histogram equalization for each image
def hist_eq(img):
    new_img = []
    # get cdf
    num_pix = len(img) * len(img[0])
    _, counts = hist(img)
    cdf = []
    sum_int = 0
    for i in range(len(counts)):
        sum_int += counts[i]
        cdf.append(math.floor(255 * sum_int / num_pix))
    # apply cdf
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            new_img[i].append(cdf[int(img[i][j])])
    return new_img

# Selected image quantization technique for user-specified levels
def quantizer(img, num_levels):
    new_img = []
    bins = [i * 255/num_levels for i in range(1, num_levels + 1)]
    dQ = 255 / num_levels
    msqe = 0
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            for k in range(num_levels):
                if img[i][j] <= bins[k]:
                    new_img[i].append((k + .5) * dQ)
                    msqe += (img[i][j] - new_img[i][j])**2
                    break
    msqe /= len(img) * len(img[0])

    return new_img, msqe

# Filtering operations: 
# *Linear filter with user-specified mask size and pixel weights
# *Median filter with user-specified mask size and pixel weights
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

def avg_linear_filter(img, weights):
    length = len(img)
    width = len(img[0])
    mask_size = math.floor(len(weights) / 2)
    ext_img = ext_bounds(img, length, width, mask_size)

    sum_weights = 0
    for i in range(len(weights)):
        sum_weights += sum(weights[i])
    # apply filter
    new_img = []
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            summed = 0
            for k in range(len(weights)):
                for m in range(len(weights[k])):
                    summed += ext_img[i-mask_size+k][j-mask_size+m] *\
                    weights[k][m]
            new_img[i].append(summed / sum_weights)

    return new_img

def med_linear_filter(img, weights):
    length = len(img)
    width = len(img[0])
    mask_size = math.floor(len(weights) / 2)
    ext_img = ext_bounds(img, length, width, mask_size)

    # apply filter
    new_img = []
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            products = []
            for k in range(len(weights)):
                for m in range(len(weights[k])):
                    products.append(ext_img[i-mask_size+k][j-mask_size+m] *\
                    weights[k][m])
            new_img[i].append(st.median(products))

    return new_img

def batch_process(folder, funcs, abbr=None, save_loc=None, verbose=False):
    # funcs should be a tuple or list like:
    # [[func1, {param1:arg1, param2:arg2}], [func2, {param1:arg1}]]
    # where paramaters are anything besides "img" or "folder"
    
    # abbr can be:
    # "cyl": columnar epithelial?
    # "para": parabasal squamous epithelial
    # "inter": intermediate squamous epithelial
    # "super": superficial squamous epithelial
    # "let": mild nonkeratinizing dysplastic?
    # "mod": moderate nonkeratinizing dysplastic
    # "svar": severe nonkeratinizing dysplastic?
    # if we only want to apply provided functions to one type of cell
    
    if abbr:
        files = glob.glob(f"{folder}/{abbr}*")
    else:
        files = glob.glob(f"{folder}/*")

    start = time.perf_counter()
    num = 0
    sum_msqe = 0
    bin_list = []
    count_list = []
    # open and grayscale each file
    for file in files:
        num += 1
        pix_list = open_in_gray(file)
        for func in funcs:
            if func[0] == quantizer:
                pix_list, msqe = func[0](pix_list, **(func[1]))
                sum_msqe += msqe
            elif func[0] == hist:
                bins, counts = func[0](pix_list, **(func[1]))
                bin_list.append(bins)
                count_list.append(counts)
            else:
                pix_list = func[0](pix_list, **(func[1]))
        if save_loc:
            to_image(pix_list, save_loc=f"{save_loc}/out{num}.bmp")
        else:
            to_image(pix_list)

    if verbose:
        # statistics
        end = time.perf_counter()
        batch = end - start
        ind = batch / len(files)
        out_string = f"batch time: {batch}\naverage individual time: {ind}"
        if quantizer in [func[0] for func in funcs]:
            out_string += f"\n mean of msqe: {sum_msqe/len(files)}"
    
        print(out_string)
    
    if save_loc:
        if hist in [func[0] for func in funcs]:
            with open(f"{save_loc}/hist_bins_and_counts.txt", "a") as file:
                file.write(f"bins: {bin_list}\n")
                file.write(f"counts: {count_list}\n")