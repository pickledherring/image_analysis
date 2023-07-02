# Converting color images to selected single color spectrum. 
def grayscale(img):
    new_img = []
    for i in range(len(img)):
        new_img.append([])
        for j in range(len(img[i])):
            # CIE recommended constants
            grey = .2126 * img[i][j][0] + .7152 * img[i][j][1]+ .0722 * img[i][j][2]
            new_img[i].append(grey)
    return new_img