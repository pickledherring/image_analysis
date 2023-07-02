import glob
import csv
import time
from re import search
from PIL import ImageFile
# local
from quantization import quantizer
from open_save import open_in_gray, to_image
from hist import hist
from features import get_feats


ImageFile.LOAD_TRUNCATED_IMAGES = True

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

# make a csv, takes several hours
with open("cells.csv", 'w') as f:
    cells_writer = csv.writer(f)
    # cells_writer.writerow(['area', 'bbox', 'com', 'perimeter', 'centroid', 'cm_11',
                            # 'cm_20', 'cm_02', 'orientation', 'class'])
    
    paths = glob.glob(f"Cancerous cell smears/*")
    for index, path in enumerate(paths):
        start = time.time()
        img = open_in_gray(path)
        feats = get_feats(img)
                
        if feats:
            # check raggedness
            if any([len(feat) == 0 or len(feat) != len(feats[0]) for feat in feats]):
                print(f"ragged or empty feature array from image {path}, skipping")
                continue

            # print("feats\n", feats)
            for i in range(len(feats[1])):
                values = []
                # returning multiple objects per feature, need to invert order
                for j in range(len(feats)):
                    values.append(feats[j][i])
                    # print(f"\tfeats[{j}][{i}]", feats[j][i])
                # add class based on file name
                values.append(search('/\D*', path).group(0)[1:])
                print("row to write", values, "\n")
                cells_writer.writerow(values)
        else:
            print(f"no clusters from image {path}")
            continue
        diff = time.time() - start
        print(f"processed {index + 1} images, last ({path}) in {diff} seconds")

# X = []
# y = []
# with open("cells.csv", "r") as f:
#     cells_reader = csv.reader(f)
#     headers = next(cells_reader)
#     for i, row in enumerate(cells_reader):
#         X.append([])
#         for j, col in enumerate(row):
#             try:
#                 if headers[j] == "com" or headers[j] == "centroid":
#                     col = literal_eval(col)
#                     X[i].append(col[0])
#                     X[i].append(col[1])
#                 elif headers[j] == "bbox":
#                     continue
#                 elif headers[j] == "class":
#                     y.append(col)
#                 else:
#                     col = literal_eval(col)
#                     X[i].append(col)
#             except:
#                 print(col)

# X_scaled = min_max_normalize(X)

# for config in range(5):
#     num_objs = len(X_scaled)
#     shuffled = list(range(num_objs))
#     random.shuffle(shuffled)
#     k_arg = 2 * config + 1
#     knn = KNN_Classifier(k_arg)

#     accs = []
#     for i in range(10):
#         x_train = []
#         y_train = []

#         for x in range(10):
#             left = math.floor(num_objs * x / 10)
#             right = math.ceil(num_objs * (x + 1) / 10)
#             if i == x:
#                 test_indices = shuffled[left:right]
#                 x_test = [X_scaled[i] for i in test_indices]
#                 y_test = [y[i] for i in test_indices]

#             else:
#                 train_indices = shuffled[left:right]
#                 x_train = x_train + [X_scaled[i] for i in train_indices]
#                 y_train = y_train + [y[i] for i in train_indices]


#         width = math.floor(num_objs / 10)
#         knn.fit(x_train, y_train)
#         preds = knn.predict(x_test)
#         accs.append(sum([preds[j] == y_test[j] for j in range(width)]) / width)
#     mean_acc = sum(accs) / len(accs)
#     acc_var = sum([(acc - mean_acc)**2 for acc in accs]) / len(accs)
#     print(f"k: {k_arg}, mean accuracy: {round(mean_acc, 3)}, variance: {round(acc_var, 5)}")