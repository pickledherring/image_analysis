from main import *
from quantization import hist_eq

##### EDIT BELOW #####
file_location = "Cancerous cell smears"
output_folder = "output"
functions = [[hist_eq, {}], [quantizer, {"num_levels": 20}]]
verbose = True
abbr = "inter"
##### EDIT ABOVE #####


batch_process(file_location, functions, abbr=abbr,
                save_loc=output_folder, verbose=verbose)

# You can edit the file location, functions (and associated arguments)
# and output location, verbose, and abbr if desired.

# Use "None" for output location if you don't want the outputs saved.
# You must create the output folder before saving to it.

# verbose can be "True" or "False" if you want statistics.

# abbr can be:
#     "cyl": columnar epithelial?
#     "para": parabasal squamous epithelial
#     "inter": intermediate squamous epithelial
#     "super": superficial squamous epithelial
#     "let": mild nonkeratinizing dysplastic?
#     "mod": moderate nonkeratinizing dysplastic
#     "svar": severe nonkeratinizing dysplastic?
#     if you only want to apply provided functions to one type of cell,
#     or "None" if you want to apply them to all images.

# Functions should look like:
# [[func1, {"param1":arg1, "param2":arg2}], [func2, {"param1":arg1}]]

# If you have one function to call, you still need the outer "[]".

# If you only want to grayscale the image, just use "[]".

# available functions for batch processing are:
# salt and pepper noise addition:
#     name: salt_n_pepper
#     parameters:
#         "prob" => float [0, 1] (<.1 recommended)
#     example use: [salt_n_pepper, {"prob": .05}]
# gaussian noise addition:
#     name: gaussian_noise
#     parameters:
#         "mu" => [-255, 255] ([-30, 30] recommended)
#         "sigma" => [0, inf] (<40 recommended)
#     example use: [gaussian_noise, {"mu": 0, "sigma": 20}]
# grayscale alone:
#     name: none
#     parameters: none
#     example use: []
# histogram:
#     name: "hist"
#     parameters: "bins" => [0, 255] (default 255)
#     example use: [hist, {}]
# histogram equalization:
#     name: "hist_eq"
#     parameters: none
#     example use: [hist_eq, {}]
# quantizing:
#     name: "quantizer"
#     parameters: "num_levels" => [0, 255] (>9 recommended)
#     example use: [quantizer, {"num_levels": 20}]
# average linear filter:
#     name: "avg_linear_filter"
#     parameters: "weights" => 2D matrix like below
#     example use: [avg_linear_filter, {"weights": gaussian_3x3}]
# median linear filter:
#     name: "med_linear_filter"
#     parameters: "weights" => 2D matrix like below
#     example use: [med_linear_filter, {"weights": gaussian_3x3}]
# edge detector:
#     name: "edge_operator"
#     parameters: "type" => "Prewitt", "Sobel", "Jahne"
#     example use: [edge_operator, {"type": "Prewitt"}]
# histogram thresholding:
#     name: "hist_thresh"
#     parameters: none
#     example use: [hist_thresh, {}]
# dilation:
#     name: "dilate"
#     parameters: "weights" => 2D matrix of 0s and 1s
#     example use: [dilate, {"weights": dilate_erode_weights}]
# erosion:
#     name: "erode"
#     parameters: "weights" => 2D matrix of 0s and 1s
#     example use: [erode, {"weights": dilate_erode_weights}]
# K-means quantization:
#     name: "k_means_quantize"
#     parameters: "k" => [2, 255] (<10 recommended)
#     example use: [k_means_quantize, {"k": 2}]
# K-means with distance:
#     name: "k_means_dist"
#     parameters: "k" => [2, 255] (<10 recommended)
#     example use: [k_means_dist, {"k": 2}]
# DBSCAN:
#     name: "dbscan"
#     parameters:
#         "radius" => [2, inf] (<10 recommended)
#         "min_obj" => [0, radius**2 * 3.14] (.2 * upper bound recommended)
#     example use: [dbscan, {"radius": 7, "min_obj": 30}]