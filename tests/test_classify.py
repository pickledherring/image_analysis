import pytest
from numpy import random
from PIL import Image
from ..src.open_save import open_in_gray
from ..src.classify import Cancer_Dataset, split_data, CNN, run_cnn
from ..src.filter import salt_n_pepper, gaussian_noise
from ..src.hist import hist
from ..src.quantization import hist_eq
from os import path
from torchvision import transforms
import torch

@pytest.fixture
def dataset():
    file_path = path.join("..", "Cancerous cell smears")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.406], std=[0.225])]
    )
    target_transform = transforms.Lambda(lambda y: torch.zeros(7, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    data_loader = Cancer_Dataset(file_path, transform=transform, target_transform=target_transform)
    return split_data(data_loader, .75)
    
def test_cnn(dataset):
    train_data, test_data = dataset
    losses, accs = run_cnn(train_data, test_data)
    assert len(losses) > 0
    # X_train, X_test, y_train, y_test = data_loader.get_data()
    # assert len(X_train) == 375
    # img_gray = open_in_gray(file_path)
    # asserts = []
    # for i in range(len(img_gray)):
    #     for j in range(len(img_gray[i])):
    #         asserts.append(img_gray[i][j] >= 0 and img_gray[i][j] <= 255)
    # assert_len = len(asserts)
    # asserts.append(assert_len > 0)
    # assert all(asserts)
    

# @pytest.mark.parametrize('file', ["test_imgs/black.png",
#     "test_imgs/white.png", "test_imgs/fuzz_color.png"])
# @pytest.mark.parametrize("prob", [0, .1, .5, .9, 1])
# def test_snp(file, prob):
#     img_list = open_in_gray(file)
#     img_noise = salt_n_pepper(img_list, prob)
#     asserts = []
#     for i in range(len(img_noise)):
#         for j in range(len(img_noise[i])):
#             asserts.append(img_noise[i][j] >= 0 and img_noise[i][j] <= 255)
#     assert all(asserts)

# @pytest.mark.parametrize('file', ["test_imgs/black.png",
#     "test_imgs/white.png", "test_imgs/fuzz_color.png"])
# @pytest.mark.parametrize("mu", [0, 122.5, 255])
# @pytest.mark.parametrize("sigma", [0, 30, 122.5, 255])
# def test_gauss(file, mu, sigma):
#     img_list = open_in_gray(file)
#     img_noise = gaussian_noise(img_list, mu=mu, sigma=sigma)
#     asserts = []
#     for i in range(len(img_noise)):
#         for j in range(len(img_noise[i])):
#             asserts.append(img_noise[i][j] >= 0 and img_noise[i][j] <= 255)
#     assert all(asserts)

# @pytest.mark.parametrize("bins", [0, 1, 2, 254, 255])
# @pytest.mark.parametrize('file', ["test_imgs/black.png",
#     "test_imgs/white.png", "test_imgs/fuzz_color.png"])
# def test_hist(file, bins):
#     img_list = open_in_gray(file)
#     bin_divs, counts = hist(img_list, bins=bins)
#     img_size = len(img_list) * len(img_list[0])

#     assert sum([counts[i] for i in range(len(counts))]) <= img_size
#     assert len(bin_divs) == bins

# @pytest.mark.parametrize('execute_5', range(5))
# def test_hist_eq(fuzz_img_color, execute_5):
#     img_gray = open_in_gray(fuzz_img_color)
#     img_hist_eq = hist_eq(img_gray)
#     asserts = []
#     for i in range(len(img_hist_eq)):
#         for j in range(len(img_hist_eq[i])):
#             asserts.append(img_hist_eq[i][j] >= 0 and img_hist_eq[i][j] <= 255)
#     assert all(asserts)

