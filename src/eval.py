# Display the following performance measures:
# * Processing time for the entire batch per each procedure 
# * Averaged processing time per image per each procedure 
# * MSQE for image quantization levels 

import time
from src.main import *

dilate_erode_weights = [[1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]]

funcs_list = [
    [[[edge_operator, {}]], "edge operator"],
    [[[hist_thresh, {}]], "histogram thresholding"],
    [[[dilate, {"weights": dilate_erode_weights}]], "dilation"],
    [[[erode, {"weights": dilate_erode_weights}]], "erosion"],
    [[[k_means_quantize, {"k": 2}]], "K-means on intensity, k = 2"],
    [[[k_means_quantize, {"k": 3}]], "K-means on intensity, k = 3"],
    [[[k_means_quantize, {"k": 4}]], "K-means on intensity, k = 4"],
    [[[k_means_dist, {"k": 2}]], "K-means with distance, k = 2"],
    [[[k_means_dist, {"k": 3}]], "K-means with distance, k = 3"],
    [[[k_means_dist, {"k": 4}]], "K-means with distance, k = 4"],
    [[[dbscan, {"radius": 7, "min_obj": 30}]], "DBSCAN"]
    ]

# [[[salt_n_pepper, {"prob": .05}]], "salt and pepper noise addition"],
# [[[gaussian_noise, {"mu": 0, "sigma": 20}]], "gaussian noise addition"],
# [[], "grayscale alone"],
# [[[hist, {}]], "histogram"],
# [[[hist_eq, {}]], "histogram equalization"],
# [[[quantizer, {"num_levels": 20}]], "quantizing"],
# [[[avg_linear_filter, {"weights": gauss_5x5}]], "average linear filter"],
# [[[med_linear_filter, {"weights": gauss_5x5}]], "median linear filter"]

for func in funcs_list:
    print(func[1], ":")
    batch_process("Cancerous cell smears", func[0], verbose=True)

# types = ["cyl", "para", "inter", "super", "let", "mod", "svar"]
# start = time.perf_counter()
# for type in types:
#     hist_avg_class("Cancerous cell smears", abbr=type)
# end = time.perf_counter()
# print("histogram average by class, for all classes: ", end - start)