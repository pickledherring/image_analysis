import matplotlib as plt
import copy
from numpy import ravel
from PIL import Image

def plot_hist(bin_divs, counts, title=None):
    half_width = bin_divs[0]/2
    plt.bar([x - half_width for x in bin_divs], counts, width = half_width*2)
    plt.xlabel("intensity")
    plt.ylabel("count")
    if title:
            titles = {"cyl": "columnar epithelial",
                    "para": "parabasal squamous epithelial",
                    "inter": "intermediate squamous epithelial",
                    "super": "superficial squamous epithelial",
                    "let": "mild nonkeratinizing dysplastic",
                    "mod": "moderate nonkeratinizing dysplastic",
                    "svar": "severe nonkeratinizing dysplastic"}
            plt.title(titles[title])
    plt.show()
    
def draw_bboxs(img, bboxs):
    img_draw = copy.deepcopy(img)
    for i in range(len(bboxs)):
        br_y = bboxs[i][1][0]
        tl_y = bboxs[i][0][0]
        br_x = bboxs[i][1][1]
        tl_x = bboxs[i][0][1]
        for j in range(br_y - tl_y):
            img_draw[tl_y+j][tl_x] = 125
            img_draw[tl_y+j][br_x] = 125
        for k in range(br_x - tl_x):
            img_draw[tl_y][tl_x+k] = 125
            img_draw[br_y][tl_x+k] = 125

    img_1d = ravel(img_draw)
    output = Image.new("L", [len(img[0]), len(img)])
    output.putdata(img_1d)
    return output