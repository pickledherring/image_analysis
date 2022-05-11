import glob
import matplotlib.pyplot as plt
import math
import statistics as st
import random
import copy
import csv
import time
from re import search
from numpy import arange, ravel, array
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

gauss_3x3 = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
gauss_5x5 = [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]

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
    
def gaussian_noise(img, mu=0, sigma=1):
    # mu can be [0, 255], sigma can be >= 0
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

# Histogram calculation for each individual image
def hist(img, bins=255):
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
def hist_avg_class(folder, abbr, bins=255):
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

def make_bgnd_white_arrayify(img):
    length = len(img)
    width = len(img[0])
    # evaluate img, if mostly black, flip every pixel
    # I assume the background will occupy most of the image
    sum_white = sum([sum(img[i]) for i in range(len(img))])
    if sum_white / 255 < length * width / 2:
        less = ravel(img) - 255
        inverted = abs(less)
        inverted = inverted.reshape((length, width))
    else:
        inverted = array(img)
    return inverted

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

def avg_linear_filter(img, weights=gauss_5x5, sum_weights=0):
    length = len(img)
    width = len(img[0])
    mask_radius = math.floor(len(weights) / 2)
    ext_img = ext_bounds(img, length, width, mask_radius)

    # sum_weights can be changed from 0 for an edge detector or other
    # filter where the sum would equal 0 and cause zero division error.
    # Otherwise, the function will figure it out, below.
    if sum_weights == 0:
        for i in range(len(weights)):
            sum_weights += sum(weights[i])
        
        if sum_weights == 0:
            print("please supply a non-zero sum_weights for this filter,\
            maybe the sum of their absolute values.")
    
    # apply filter
    new_img = []
    for i in range(mask_radius, length + mask_radius):
        new_img.append([])
        for j in range(mask_radius, width + mask_radius):
            summed = 0
            for k in range(len(weights)):
                for m in range(len(weights[k])):
                    summed += ext_img[i-mask_radius+k][j-mask_radius+m] *\
                    weights[k][m]
            new_img[i-mask_radius].append(summed / sum_weights)

    return new_img

def med_linear_filter(img, weights=gauss_5x5):
    length = len(img)
    width = len(img[0])
    mask_radius = math.floor(len(weights) / 2)
    ext_img = ext_bounds(img, length, width, mask_radius)

    # apply filter
    new_img = []
    for i in range(mask_radius, length + mask_radius):
        new_img.append([])
        for j in range(mask_radius, width + mask_radius):
            products = []
            for k in range(len(weights)):
                for m in range(len(weights[k])):
                    for _ in range(weights[k][m]):
                        products.append(ext_img[i-mask_radius+k][j-mask_radius+m])
            new_img[i-mask_radius].append(st.median(products))

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

# Finds the edges in an image. Default can be kinda dark, so set
# sharpen_thresh=<numeric> to return a binary image with everything
# above sharpen_thresh = 255 and everything else = 0.
def edge_operator(img, type="Jahne", sharpen_thresh=0):
    new_img_x = []
    new_img_y = []
    if type == "Prewitt":
        x_mask = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        y_mask = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
        new_img_x = avg_linear_filter(img, weights=x_mask, sum_weights=6)
        new_img_y = avg_linear_filter(img, weights=y_mask, sum_weights=6)
    elif type == "Sobel":
        x_mask = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        y_mask = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        new_img_x = avg_linear_filter(img, weights=x_mask, sum_weights=8)
        new_img_y = avg_linear_filter(img, weights=y_mask, sum_weights=8)
    elif type == "Jahne":
        x_mask = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        y_mask = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]
        new_img_x = avg_linear_filter(img, weights=x_mask, sum_weights=32)
        new_img_y = avg_linear_filter(img, weights=y_mask, sum_weights=32)

    length = len(new_img_y)
    width = len(new_img_y[0])
    new_img = []
    for i in range(length):
        new_img.append([])
        for j in range(width):
            grad = math.sqrt(new_img_x[i][j]**2 + new_img_y[i][j]**2)
            if sharpen_thresh > 0:
                if grad > sharpen_thresh:
                    new_img[i].append(255)
                else:
                    new_img[i].append(0)
            else:
                new_img[i].append(grad)
    return new_img

# Also returns a binary image, but finds the threshold that minimizes
# within-group variance of the two sides.
def hist_thresh(img):
    bins, counts = hist(img)
    num_pix = sum(counts)
    within_group_vars = []

    for thresh in range(2, 255):
        obj_prior = sum(counts[:thresh]) / num_pix
        back_prior = sum(counts[thresh:]) / num_pix

        if obj_prior != 0:
            obj_mean = (sum([i * counts[i] for i in range(thresh)]) / num_pix)\
                / obj_prior
            obj_var = sum([(i - obj_mean)**2 * counts[i] for i in range(thresh)])\
                / (num_pix * obj_prior)
        else:
            obj_var = 0
        
        if back_prior != 0:
            back_mean = (sum([i * counts[i] for i in range(thresh, 255)]) / num_pix)\
                / back_prior
            back_var = sum([(i - back_mean)**2 * counts[i] for i in range(thresh, 255)])\
                / (num_pix * back_prior)
        else:
            back_var = 0

        within_group_vars.append(obj_var * obj_prior + back_var * back_prior)
    
    index = within_group_vars.index(min(within_group_vars))
    threshold = bins[index+1]

    length = len(img)
    width = len(img[0])

    new_img = []
    for i in range(length):
        new_img.append([])
        for j in range(width):
            if img[i][j] <= threshold:
                new_img[i].append(0)
            else:
                new_img[i].append(255)
    return new_img

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

# Essentially quantization with the bins decided by K-means clusters
def k_means_quantize(img, k, iter=10):
    length = len(img)
    width = len(img[0])

    intensity = random.sample(range(256), k)
    centroids = [i for i in intensity]

    new_img = [[0 for _ in range(width)] for _ in range(length)]

    for _ in range(iter):
        # sums keeps track of the sum for [x, y, intensity] for each centroid
        # counts is number of points assigned to each centroid
        sums = [0 for _ in range(k)]
        counts = [0 for _ in range(k)]

        for i in range(length):
            for j in range(width):
                cent_dist = 9999999999999
                closest = 0

                for m in range(k):
                    int_dist = centroids[m] - img[i][j]
                    dist = math.sqrt(int_dist**2)

                    if dist < cent_dist:
                        cent_dist = dist
                        closest = m
                
                # assign pixels to centroids and update stats for cluster centroids
                new_img[i][j] = closest
                sums[closest] += img[i][j]
                counts[closest] += 1
            
        # move centroids to avg of assigned pixels
        for m in range(k):
            if counts[m] != 0:
                centroids[m] = sums[m] / counts[m]

    colors = list(range(0, 255, round(255/k)))
    for i in range(length):
        for j in range(width):
            new_img[i][j] = colors[new_img[i][j]]

    return new_img

# K-means clustering with distance to centroid as a feature
# Distance weight is best left <= .5, but you can make it 1
# to consider it equally.
def k_means_dist(img, k, iter=10, dist_weight=.25):
    length = len(img)
    width = len(img[0])
    avg_dim = (width + length) / 2

    ii = random.sample(range(length), k)
    jj = random.sample(range(width), k)
    intensity = random.sample(range(256), k)
    centroids = [[i, j, inten] for i, j, inten in zip(ii, jj, intensity)]

    new_img = [[0 for _ in range(width)] for _ in range(length)]

    for _ in range(iter):
        # sums keeps track of the sum for [x, y, intensity] for each centroid
        # counts is number of points assigned to each centroid
        sums = [[0, 0, 0] for _ in range(k)]
        counts = [0 for _ in range(k)]

        for i in range(length):
            for j in range(width):
                cent_dist = 9999999999999
                closest = 0

                for m in range(k):
                    i_dist = centroids[m][0] - i
                    j_dist = centroids[m][1] - j
                    int_dist = centroids[m][2] - img[i][j]

                    # intensity is scaled by the avg of the dimensions, distances can be
                    # scaled by dist_weight
                    dist = math.sqrt((i_dist * dist_weight)**2 + (j_dist * dist_weight)**2\
                        + (int_dist * avg_dim / 255)**2)

                    if dist < cent_dist:
                        cent_dist = dist
                        closest = m
                
                # assign pixels to centroids and update stats for cluster centroids
                new_img[i][j] = closest
                sums[closest][0] += i
                sums[closest][1] += j
                sums[closest][2] += img[i][j]
                counts[closest] += 1
            
        # move centroids to avg of assigned pixels
        for m in range(k):
            if counts[m] != 0:
                centroids[m][0] = sums[m][0] / counts[m]
                centroids[m][1] = sums[m][1] / counts[m]
                centroids[m][2] = sums[m][2] / counts[m]

    colors = list(range(0, 255, round(255/k)))
    for i in range(length):
        for j in range(width):
            new_img[i][j] = colors[new_img[i][j]]

    return new_img

# DBSCAN clustering of an image using intensity and distance. Very slow!
# I have found the optimal relation of min_obj to radius is
# min_obj = .2 * radius**2 * 3.1
# Decrease radius (and min_obj) to expedite
def dbscan(img, radius=10, min_obj=60):
    length = len(img)
    width = len(img[0])
    avg_dim = (length + width) / 2

    new_img = [[0 for _ in range(width)] for _ in range(length)]

    cores = []
    for i in range(length):
        for j in range(width):
            n_nearby = 0
            nearby = []
            # searching within a circle within a square - there's probably a better way to do this
            for m in range(-radius, radius):
                for n in range(-radius, radius):
                    if i + m >= 0 and j + n >= 0 and i + m < length and j + n < width:
                        # scale the intensity distance to image dimensions
                        int_dist = (img[i][j] - img[i+m][j+n]) * avg_dim / 255
                        if math.sqrt(m**2 + n**2 + int_dist**2) <= radius:
                            n_nearby += 1
                            nearby.append([i + m, j + n])
                            new_img[i+m][j+n] = -1

            if n_nearby >= min_obj:
                cores.append({"loc": [i, j], "neighbors": nearby})
                new_img[i][j] = -1

    clusters = {}
    cluster_serial = 0
    for core in cores:
        # 0 for the pixel value will stand in for "background", 1, 2, 3, etc. will be cluster #s
        # if we find a background core, make it the start of a new cluster
        i = core["loc"][0]
        j = core["loc"][1]
        if new_img[i][j] == -1:
            new_img[i][j] = cluster_serial
            clusters[str(cluster_serial)] = [[i, j]]
            cluster_serial += 1
        # go through the neighbors of the core and add them to the core's cluster if they are background
        for point in core["neighbors"]:
            m = point[0]
            n = point[1]
            if new_img[m][n] == -1:
                new_img[m][n] = new_img[i][j]
                clusters[str(new_img[i][j])].append([m, n])

            # if we find a point belonging to a different cluster, add this cluster to that one
            elif new_img[m][n] != new_img[i][j]:
                temp = str(new_img[i][j])
                for nb in clusters[str(new_img[i][j])]:
                    new_img[nb[0]][nb[1]] = new_img[m][n]

                clusters[str(new_img[m][n])].extend(clusters[temp])
                clusters.pop(temp)
    if cluster_serial > 0:
        step = 255/len(clusters)
        colors = list(arange(step, 255 + step, step))
        cluster_keys = list(clusters.keys())
        for i in range(length):
            for j in range(width):
                if new_img[i][j] == -1:
                    new_img[i][j] = 0
                else:
                    cluster_index = cluster_keys.index(str(new_img[i][j]))
                    new_img[i][j] = colors[cluster_index]

    return new_img

dilate_erode_weights = [[1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]]

def dbscan_seg(img, radius=10, min_obj=60):
    length = len(img)
    width = len(img[0])
    avg_dim = (length + width) / 2

    new_img = [[0 for _ in range(width)] for _ in range(length)]

    cores = []
    for i in range(length):
        for j in range(width):
            n_nearby = 0
            nearby = []
            # searching within a circle within a square - there's probably a better way to do this
            for m in range(-radius, radius):
                for n in range(-radius, radius):
                    if i + m >= 0 and j + n >= 0 and i + m < length and j + n < width:
                        # scale the intensity distance to image dimensions
                        int_dist = (img[i][j] - img[i+m][j+n]) * avg_dim / 255
                        if math.sqrt(m**2 + n**2 + int_dist**2) <= radius:
                            n_nearby += 1
                            nearby.append([i + m, j + n])
                            new_img[i+m][j+n] = -1

            if n_nearby >= min_obj:
                cores.append({"loc": [i, j], "neighbors": nearby})
                new_img[i][j] = -1

    clusters = {}
    cluster_serial = 0
    for core in cores:
        # 0 for the pixel value will stand in for "background", 1, 2, 3, etc. will be cluster #s
        # if we find a background core, make it the start of a new cluster
        i = core["loc"][0]
        j = core["loc"][1]
        if new_img[i][j] == -1:
            new_img[i][j] = cluster_serial
            clusters[str(cluster_serial)] = [[i, j]]
            cluster_serial += 1
        # go through the neighbors of the core and add them to the core's cluster if they are background
        for point in core["neighbors"]:
            m = point[0]
            n = point[1]
            if new_img[m][n] == -1:
                new_img[m][n] = new_img[i][j]
                clusters[str(new_img[i][j])].append([m, n])

            # if we find a point belonging to a different cluster, add this cluster to that one
            elif new_img[m][n] != new_img[i][j]:
                temp = str(new_img[i][j])
                for nb in clusters[str(new_img[i][j])]:
                    new_img[nb[0]][nb[1]] = new_img[m][n]

                clusters[str(new_img[m][n])].extend(clusters[temp])
                clusters.pop(temp)

    return clusters

def get_perimeter(seg_img):
    cross_weights = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    dilated = dilate(seg_img, weights=cross_weights)
    dilated_1d = ravel(dilated) + 255
    seg_1d = ravel(seg_img)
    outlines = dilated_1d - seg_1d
    negative = outlines - 255
    return -sum(negative) / 255

def get_area_bbox_com(clusters):
    areas = []
    bboxs = []
    coms = []
    for cluster_k in clusters.keys():
        num_points = len(clusters[cluster_k])
        areas.append(num_points)

        j_sum = k_sum = j_max = k_max = 0
        j_min = k_min = 100000
        for i in range(num_points):
            j = clusters[cluster_k][i][0]
            k = clusters[cluster_k][i][1]
            j_sum += j
            k_sum += k
            if j < j_min:
                j_min = j
            if k < k_min:
                k_min = k
            if j > j_max:
                j_max = j
            if k > k_max:
                k_max = k

        # bounding boxes have shape [height, width]
        bbox_top_left = [j_min, k_min]
        bbox_bottom_right = [j_max, k_max]

        center_of_mass = [round(j_sum / num_points), round(k_sum / num_points)]

        bboxs.append([bbox_top_left, bbox_bottom_right])
        coms.append(center_of_mass)
    
    return areas, bboxs, coms

def draw_bboxs(img, bboxs):
    img_draw = copy.deepcopy(img)
    for i in range(len(bboxs)):
        br_y = bboxs[i][1][0]
        tl_y = bboxs[i][0][0]
        br_x = bboxs[i][1][1]
        tl_x = bboxs[i][0][1]
        for j in range(br_y - tl_y):
            img_draw[tl_y+j][tl_x] = 125
            img_draw[tl_y+j][br_x] = 125
        for k in range(br_x - tl_x):
            img_draw[tl_y][tl_x+k] = 125
            img_draw[br_y][tl_x+k] = 125

    img_1d = ravel(img_draw)
    output = Image.new("L", [len(img[0]), len(img)])
    output.putdata(img_1d)
    return output

def cut_clusters(img, bboxs):
    # returns list of image patches from bounding boxes
    new_images = []
    for i in range(len(bboxs)):
        new_images.append([])
        br_i = bboxs[i][1][0]
        tl_i = bboxs[i][0][0]
        br_j = bboxs[i][1][1]
        tl_j = bboxs[i][0][1]
        for j in range(tl_i, br_i + 1):
            new_images[i].append([])
            for k in range(tl_j, br_j + 1):
                try:
                    new_images[i][j-tl_i].append(img[j][k])
                except:
                    print(i, j, k)

    
    return new_images

def get_centroid(img):
    height = len(img)
    width = len(img[0])

    flip_sum_ix = flip_sum_jx = flip_area = 0
    for i in range(height):
        for j in range(width):
            flip_sum_ix += (img[i][j] - 255) * i
            flip_sum_jx += (img[i][j] - 255) * j
            flip_area += img[i][j] - 255
    area = flip_area
    return [flip_sum_ix / area, flip_sum_jx / area]

def cent_moment(img, com, p, q):
    height = len(img)
    width = len(img[0])
    summed = 0
    for i in range(height):
        for j in range(width):
            value = abs(img[i][j] - 255) / 255
            summed += (i - com[0])**p * (j - com[1])**q * value
    return summed

def get_cms_orient(img, com):
    cm_00 = cent_moment(img, com, 0, 0)
    cm_11 = cent_moment(img, com, 1, 1)
    cm_02 = cent_moment(img, com, 0, 2)
    cm_20 = cent_moment(img, com, 2, 0)

    cm_20_prime = cm_20 / cm_00
    cm_02_prime = cm_02 / cm_00
    cm_11_prime = cm_11 / cm_00

    theta = 1/2 * math.atan(2 * cm_11_prime / (cm_20_prime - cm_02_prime))

    return cm_11, cm_02, cm_20, theta

def get_feats(img):
    # all the feature extraction and processing for that, takes a grayscale array image
    # get rid of annoying line artifacts on the edges
    crop_img = copy.deepcopy([x[3:-3] for x in img[3:-3]])
    # binarize image
    seg = k_means_quantize(crop_img, 2)
    dilate_erode_weights = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # clean it up a bit
    eroded = erode(seg, weights=dilate_erode_weights)
    clean = dilate(eroded, weights=dilate_erode_weights)

    # make the largest cluster, probably the background, white so dbscan_seg will ignore it
    clean = make_bgnd_white_arrayify(clean)
    # print("image cleaned, clustering (will take a while)")

    # get object clusters (pixel locations) from image
    clusters = dbscan_seg(clean, radius=7, min_obj=30)
    # print("done clustering! extracting cells and features")

    # somewhat subjectively cut off small clusters
    for cluster_k in clusters.keys():
        num_points = len(clusters[cluster_k])
        if num_points < 200:
            clusters.pop(cluster_k)

    # get the background cluster out of there
    avg_intensities = []
    keys = []
    for cluster_k in clusters.keys():
        num_points = len(clusters[cluster_k])
        
        sum_intensity = 0
        for i in range(num_points):
            j = clusters[cluster_k][i][0]
            k = clusters[cluster_k][i][1]
            sum_intensity += clean[j][k]
        avg_intensities.append(sum_intensity / num_points)
        keys.append((cluster_k))

    max_index = avg_intensities.index(max(avg_intensities))
    clusters.pop(keys[max_index])

    # start retrieving features
    areas, bboxs, coms = get_area_bbox_com(clusters)
    height = len(img)
    width = len(img[0])

    # get square patches from segmented image
    patches = cut_clusters(clean, bboxs)
    perimeters = []
    centroids = []
    cms_11 = []
    cms_20 = []
    cms_02 = []
    orientations = []
    for i, patch in enumerate(patches):
        # pad image so dilation to get perimeter works
        padded = pad(patch, 3)
        perimeters.append(get_perimeter(padded))

        # get centroid and center of mass
        centroid = get_centroid(patch)
        centroids.append([centroid[0] / len(patch), centroid[1] / len(patch[0])])

        coms[i] = [coms[i][0] / height, coms[i][1] / width]

        # 1st and 2nd central moments and orientation
        cm_11, cm_02, cm_20, theta = get_cms_orient(patch, coms[i])
        cms_11.append(cm_11)
        cms_20.append(cm_20)
        cms_02.append(cm_02)
        orientations.append(theta)

    return (areas, bboxs, coms, perimeters, centroids, cms_11,
    cms_20, cms_02, orientations)

# abbrs = ["cyl", "para", "inter", "super", "let", "mod", "svar"]
# for abbr in abbrs:
#     to_image(edge_operator(open_in_gray(f"Cancerous cell smears/{abbr}01.BMP"), sharpen_thresh=5),
        # save_loc=f"outputs/{abbr}_sharp.png")

# make a csv, takes several hours
paths = glob.glob(f"Cancerous cell smears/*")
data = []
# sum_n_feats = 0
for index, path in enumerate(paths[:250:5]):
    start = time.time()
    img = open_in_gray(path)
    feats = get_feats(img)
    sum_n_feats = len(feats[1])
    diff = time.time() - start
    # if index % 1 == 0:
    print(f"processed {index} images, avg n feats: {sum_n_feats}, time: {diff}")
    for i in range(len(feats[1])):
        values = []
        # returning multiple objects per feature, need to invert order
        for j in range(len(feats)):
            values.append(feats[j][i])
        # add class based on file name
        values.append(search('/\D*', path).group(0)[1:])
    data.append(values)

with open("cells.csv", 'w') as f:
    cells_writer = csv.writer(f)
    cells_writer.writerow(['area', 'bbox', 'com', 'perimeter', 'centroid', 'cm_11',
                            'cm_20', 'cm_02', 'orientation', 'class'])
    
    for row in data:
        cells_writer.writerow(data)





