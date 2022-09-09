from scipy import ndimage
from scipy.ndimage.filters import convolve

from scipy import misc
import numpy as np
import cv2

weak_pixel = 25
strong_pixel = 75


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g




