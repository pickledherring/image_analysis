from numpy import ravel, array
import math
from .minkowski import dilate, erode
from .cluster import k_means_quantize, dbscan_seg
from .modify_bounds import pad

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

    # get object clusters (pixel locations) from image
    clusters = dbscan_seg(clean, radius = 7, min_obj=30)
    
    if len(clusters.keys()) == 0:
        return None
        
    # somewhat subjectively cut off small and large clusters
    to_pop = []
    for cluster_k in clusters.keys():
        num_points = len(clusters[cluster_k])
        if num_points < 200 or num_points > 300000:
            to_pop.append(cluster_k)
    
    # print("to pop", *zip(clusters.keys(), [len(clusters[key]) for key in to_pop]))
    for key in to_pop:
        clusters.pop(key)

    # get the background cluster out of there
    if len(clusters.keys()) > 1:
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
    else:
        print("single cluster for image, may be background")

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