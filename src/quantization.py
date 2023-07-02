import math
from .hist import hist

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

# Selected image quantization technique for user-specified levels
def quantizer(img, num_levels):
    new_img = []
    bins = [i * 255/num_levels for i in range(1, num_levels + 1)]
    d_q = 255 / num_levels
    msqe = 0
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            for k in range(num_levels):
                if img[i][j] <= bins[k]:
                    new_img[i].append((k + .5) * d_q)
                    msqe += (img[i][j] - new_img[i][j])**2
                    break
    msqe /= len(img) * len(img[0])

    return new_img, msqe