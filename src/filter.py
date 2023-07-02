import random
import math
from .modify_bounds import ext_bounds


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


def min_max_normalize(data, bounds = [0, 1]):
    num_feats = len(data[0])
    maxes = [data[0][i] for i in range(num_feats)]
    mins = [data[0][i] for i in range(num_feats)]
    for row in data:
        for col in range(num_feats):
            if row[col] > maxes[col]:
                maxes[col] = row[col]
            
            if row[col] < mins[col]:
                mins[col] = row[col]

    new_data = []
    for i, row in enumerate(data):
        new_data.append([])
        for j in range(num_feats):
            curr_range = maxes[j] - mins[j]
            new_range = bounds[1] - bounds[0]
            value = ((row[j] - mins[j]) / curr_range) * new_range + bounds[0]
            new_data[i].append(value)

    return new_data