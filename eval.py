# Display the following performance measures:
# * Processing time for the entire batch per each procedure 
# * Averaged processing time per image per each procedure 
# * MSQE for image quantization levels 

import time
from main import *

blurring = [[1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]]

funcs_list = [[[[salt_n_pepper, {"prob": .05}]], "salt and pepper noise addition"],
            [[[gaussian_noise, {"mu": 0, "sigma": 20}]], "gaussian noise addition"],
            [[], "grayscale alone"],
            [[[hist, {}]], "histogram"],
            [[[hist_eq, {}]], "histogram equalization"],
            [[[quantizer, {"num_levels": 20}]], "quantizing"],
            [[[avg_linear_filter, {"weights": blurring}]], "average linear filter"],
            [[[med_linear_filter, {"weights": blurring}]], "median linear filter"]]

for func in funcs_list:
    print(func[1], ":")
    batch_process("Cancerous cell smears", func[0], verbose=True)

types = ["cyl", "para", "inter", "super", "let", "mod", "svar"]
start = time.perf_counter()
for type in types:
    hist_avg_class("Cancerous cell smears", abbr=type)
end = time.perf_counter()
print("histogram average by class, for all classes: ", end - start)