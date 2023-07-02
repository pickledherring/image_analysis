from numpy import arange
import random
import math


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