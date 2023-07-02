# for CSMC 525 Software, Analysis, Testing, and Verification

from ..src.main import *
import pytest
from numpy import random
from PIL import Image

@pytest.fixture
def fuzz_img_color():
    # creates a 20x20 randomized color image, returns the path to it
    # actually saves an image, overwriting every time
    img = random.rand(20, 20, 3) * 255
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img.save("test_imgs/fuzz_color.png")
    return "test_imgs/fuzz_color.png"

@pytest.mark.parametrize('execute_5', range(5))
def test_open_in_gray(fuzz_img_color, execute_5):
    img_gray = open_in_gray(fuzz_img_color)
    asserts = []
    for i in range(len(img_gray)):
        for j in range(len(img_gray[i])):
            asserts.append(img_gray[i][j] >= 0 and img_gray[i][j] <= 255)
    assert all(asserts)

@pytest.mark.parametrize('file', ["test_imgs/black.png",
    "test_imgs/white.png", "test_imgs/fuzz_color.png"])
@pytest.mark.parametrize("prob", [0, .1, .5, .9, 1])
def test_snp(file, prob):
    img_list = open_in_gray(file)
    img_noise = salt_n_pepper(img_list, prob)
    asserts = []
    for i in range(len(img_noise)):
        for j in range(len(img_noise[i])):
            asserts.append(img_noise[i][j] >= 0 and img_noise[i][j] <= 255)
    assert all(asserts)

@pytest.mark.parametrize('file', ["test_imgs/black.png",
    "test_imgs/white.png", "test_imgs/fuzz_color.png"])
@pytest.mark.parametrize("mu", [0, 122.5, 255])
@pytest.mark.parametrize("sigma", [0, 30, 122.5, 255])
def test_gauss(file, mu, sigma):
    img_list = open_in_gray(file)
    img_noise = gaussian_noise(img_list, mu=mu, sigma=sigma)
    asserts = []
    for i in range(len(img_noise)):
        for j in range(len(img_noise[i])):
            asserts.append(img_noise[i][j] >= 0 and img_noise[i][j] <= 255)
    assert all(asserts)

@pytest.mark.parametrize("bins", [0, 1, 2, 254, 255])
@pytest.mark.parametrize('file', ["test_imgs/black.png",
    "test_imgs/white.png", "test_imgs/fuzz_color.png"])
def test_hist(file, bins):
    img_list = open_in_gray(file)
    bin_divs, counts = hist(img_list, bins=bins)
    img_size = len(img_list) * len(img_list[0])

    assert sum([counts[i] for i in range(len(counts))]) <= img_size
    assert len(bin_divs) == bins

@pytest.mark.parametrize('execute_5', range(5))
def test_hist_eq(fuzz_img_color, execute_5):
    img_gray = open_in_gray(fuzz_img_color)
    img_hist_eq = hist_eq(img_gray)
    asserts = []
    for i in range(len(img_hist_eq)):
        for j in range(len(img_hist_eq[i])):
            asserts.append(img_hist_eq[i][j] >= 0 and img_hist_eq[i][j] <= 255)
    assert all(asserts)

