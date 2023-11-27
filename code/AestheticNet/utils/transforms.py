import random
import cv2 as cv
import numpy as np
from PIL import Image
import math

def detail_enhance(image, sigma_s, sigma_r):
    degraded_img = cv.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)

    return degraded_img

def stylization(image, sigma_s, sigma_r):
    degraded_img = cv.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)

    return degraded_img


# Add gaussian noise
def add_gaussian_noise(image, sigma):
    image = image / 255
    noise = np.random.normal(0, sigma, image.shape)
    degraded_img = image + noise
    degraded_img = np.clip(degraded_img, 0, 1)

    degraded_img = degraded_img * 255
    degraded_img = np.uint8(degraded_img)

    return degraded_img

# color quantization (according to https://github.com/makelove/OpenCV-Python-Tutorial/blob/master/ch48-K%E5%80%BC%E8%81%9A%E7%B1%BB/48.2.3_%E9%A2%9C%E8%89%B2%E9%87%8F%E5%8C%96.py)
def quantization(image, K):
    tmp = image.reshape((-1, 3))
    tmp = np.float32(tmp)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv.kmeans(
        tmp, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    degraded_img = center[label.flatten()]
    degraded_img = degraded_img.reshape((image.shape))

    return degraded_img

# Gaussian blur
def gaussian_blur(image, std):
    degraded_img = cv.GaussianBlur(image, (5, 5), std)

    return degraded_img

# crop and resize
def crop_resize(image, factor):
    h, w, c = image.shape
    degraded_img = image[0:h - h // factor, 0:w - w // factor, :]
    degraded_img = cv.resize(degraded_img, (w, h))

    return degraded_img

# convex lens
def convex(image, factor):
    h, w, c = image.shape
    center_x, center_y = h/2, w/2
    radius = math.sqrt(h**2+w**2)/factor
    degraded_img = image.copy()
    for i in range(h):
        for j in range(w):
            dis = math.sqrt((i-center_x)**2+(j-center_y)**2)
            if dis <= radius:
                new_i = int(np.round(dis/radius*(i-center_x)+center_x))
                new_j = int(np.round(dis/radius*(j-center_y)+center_y))
                degraded_img[i, j] = image[new_i, new_j]
    return degraded_img

# change exposure
def exposure(image, factor):
    degraded_img = image * factor
    degraded_img = np.clip(degraded_img, 0, 255)
    degraded_img = np.uint8(degraded_img)

    return degraded_img

# rotate
def rotate(image, deg):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    angle = deg
    scale = 0.8

    M = cv.getRotationMatrix2D(center, angle, scale)
    image_rotation = cv.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(255, 255, 255))

    return image_rotation

# Custom transform class
class CustomTransform:
    def __init__(self, manipulation_options):
        """
        Args:
            manipulation_options (list): List of manipulation options to apply.
        """
        self.manipulation_options = manipulation_options

    def __call__(self, image):
        # Convert PIL Image to OpenCV format
        image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

        # Randomly choose a manipulation
        opt = random.choice(self.manipulation_options)

        # Apply the chosen manipulation
        image_cv = image_manipulation(image_cv, opt)

        # Convert back to PIL Image
        image_pil = Image.fromarray(cv.cvtColor(image_cv, cv.COLOR_BGR2RGB))
        return image_pil

def image_manipulation(image, opt):
    # Apply the chosen manipulation based on the option
    if opt == 0:
        return stylization(image, 50, 0.6)
    elif opt == 1:
        return stylization(image, 50, 0.3)
    elif opt == 2:
        return stylization(image, 50, 0.1)
    elif opt == 3:
        return add_gaussian_noise(image, 0.2)
    elif opt == 4:
        return add_gaussian_noise(image, 0.4)
    elif opt == 5:
        return add_gaussian_noise(image, 0.8)
    elif opt == 6:
        return crop_resize(image, 4)
    elif opt == 7:
        return crop_resize(image, 3)
    elif opt == 8:
        return crop_resize(image, 2)
    elif opt == 9:
        return quantization(image, 64)
    elif opt == 10:
        return quantization(image, 32)
    elif opt == 11:
        return quantization(image, 8)
    elif opt == 12:
        return gaussian_blur(image, 0.4)
    elif opt == 13:
        return gaussian_blur(image, 0.8)
    elif opt == 14:
        return gaussian_blur(image, 2)
    elif opt == 15:
        return convex(image, 8)
    elif opt == 16:
        return convex(image, 4)
    elif opt == 17:
        return convex(image, 2)
    elif opt == 18:
        return exposure(image, 1.5)
    elif opt == 19:
        return exposure(image, 2.0)
    elif opt == 20:
        return exposure(image, 2.5)
    elif opt == 21:
        return rotate(image, 15)
    elif opt == 22:
        return rotate(image, -15)
    else:  # opt == 23 or any other case
        return image

# Usage example
# transform = CustomTransform(manipulation_options=[0, 1, 3, 9, 12, 21])