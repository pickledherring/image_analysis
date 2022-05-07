from numpy import random
import main
from PIL import Image

black_img = Image.new("RGB", (20, 20))
white_img = Image.new("RGB", (20, 20), 'white')
black_img.save("test_imgs/black.png")
white_img.save("test_imgs/white.png")