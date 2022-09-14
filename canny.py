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



def color_thr(s_img, l_img, s_threshold = (0,255), l_threshold = (0,255)):
    s_binary = np.zeros_like(s_img).astype(np.uint8)
    s_binary[(s_img > s_threshold[0]) & (s_img <= s_threshold[1])] = 1
    l_binary = np.zeros_like(l_img).astype(np.uint8)
    l_binary[(l_img > l_threshold[0]) & (l_img <= l_threshold[1])] = 1
    col = ((s_binary == 1) | (l_binary == 1))
    return col

