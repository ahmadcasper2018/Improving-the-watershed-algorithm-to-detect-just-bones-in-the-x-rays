import numpy as np
import cv2
from scipy import ndimage


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def sobel_filters(img):
    sobel_x = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    theta = np.arctan2(sobel_y, sobel_x)

    return (sobel, theta)


def non_max_suppression(Gm, Gd):
    Gd = np.rad2deg(Gd)
    num_rows, num_cols = Gm.shape[0], Gm.shape[1]
    Gd_bins = 45 * (np.round(Gd / 45))

    G_NMS = np.zeros(Gm.shape)

    neighbor_a, neighbor_b = 0., 0.

    for r in range(1, num_rows - 1):
        for c in range(1, num_cols - 1):
            angle = Gd_bins[r, c]
            if angle == 180. or angle == -180. or angle == 0.:
                neighbor_a, neighbor_b = Gm[r + 1, c], Gm[r - 1, c]
            elif angle == 90. or angle == -90.:
                neighbor_a, neighbor_b = Gm[r, c - 1], Gm[r, c + 1]
            elif angle == 45. or angle == -135.:
                neighbor_a, neighbor_b = Gm[r + 1, c + 1], Gm[r - 1, c - 1]
            elif angle == -45. or angle == 135.:
                neighbor_a, neighbor_b = Gm[r - 1, c + 1], Gm[r + 1, c - 1]
            else:
                print("error")
                return

            if Gm[r, c] > neighbor_a and Gm[r, c] > neighbor_b:
                G_NMS[r, c] = Gm[r, c]

    return G_NMS


# def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
#     highThreshold = img.max() * highThresholdRatio;
#     lowThreshold = highThreshold * lowThresholdRatio;
#
#     M, N = img.shape
#     res = np.zeros((M, N), dtype=np.int32)
#
#     weak = np.int32(25)
#     strong = np.int32(255)
#
#     strong_i, strong_j = np.where(img >= highThreshold)
#     zeros_i, zeros_j = np.where(img < lowThreshold)
#
#     weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
#
#     res[strong_i, strong_j] = strong
#     res[weak_i, weak_j] = weak
#
#     return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    kernal = gaussian_kernel(3)
    identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernal)
    # cv2.imshow('gauss',identity)
    # cv2.waitKey(0)
    edges, theta = sobel_filters(identity)
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)
    supression = non_max_suppression(edges, theta)
    final = hysteresis(edges, lower, upper)
    # edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return final, lower, upper
