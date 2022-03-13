from main import *

file_location = "Cancerous cell smears"
output_folder = None
functions = [[hist_eq, {}], [quantizer, {"num_levels": 20}]]
verbose = True
abbr = "inter"

batch_process(file_location, functions, abbr=abbr,
                save_loc=output_folder, verbose=verbose)

# gaussian blurring matrix for convenience
blur = [[1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]]

# You can edit the file location, functions (and associated arguments)
# and output location, verbose, and abbr if desired.

# Use "None" for output location if you don't want the outputs saved.

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
#     example use: [avg_linear_filter, {"weights": blur}]
# median linear filter:
#     name: "med_linear_filter"
#     parameters: "weights" => 2D matrix like below
#     example use: [med_linear_filter, {"weights": blur}]
