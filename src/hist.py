import glob
from .open_save import open_in_gray

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