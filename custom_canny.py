import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def sHalf(T, sigma):
    temp = -np.log(T) * 2 * (sigma ** 2)
    return np.round(np.sqrt(temp))


def calculate_filter_size(T, sigma):
    return 2 * sHalf(T, sigma) + 1


def MaskGeneration(T, sigma):
    N = calculate_filter_size(T, sigma)
    shalf = sHalf(T, sigma)
    y, x = np.meshgrid(range(-int(shalf), int(shalf) + 1), range(-int(shalf), int(shalf) + 1))
    return x, y


def Gaussian(x, y, sigma):
    temp = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
    return (np.exp(-temp))


def calculate_gradient_X(x, y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((x * np.exp(-temp)) / sigma ** 2)


def calculate_gradient_Y(x, y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((y * np.exp(-temp)) / sigma ** 2)


def pad(img, kernel):
    r, c = img.shape
    kr, kc = kernel.shape
    padded = np.zeros((r + kr, c + kc), dtype=img.dtype)
    insert = np.uint((kr) / 2)
    padded[insert: insert + r, insert: insert + c] = img
    return padded


def smooth(img, kernel=None):
    if kernel is None:
        mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    else:
        mask = kernel
    i, j = mask.shape
    output = np.zeros((img.shape[0], img.shape[1]))
    image_padded = pad(img, mask)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            output[x, y] = (mask * image_padded[x:x + i, y:y + j]).sum() / mask.sum()
    return output


def Create_Gx(fx, fy):
    gx = calculate_gradient_X(fx, fy, sigma)
    gx = (gx * 255)
    return np.around(gx)


def Create_Gy(fx, fy):
    gy = calculate_gradient_Y(fx, fy, sigma)
    gy = (gy * 255)
    return np.around(gy)


def ApplyMask(image, kernel):
    i, j = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    image_padded = pad(image, kernel)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output[x, y] = (kernel * image_padded[x:x + i, y:y + j]).sum()
    return output


def Gradient_Magnitude(fx, fy):
    mag = np.zeros((fx.shape[0], fx.shape[1]))
    mag = np.sqrt((fx ** 2) + (fy ** 2))
    mag = mag * 100 / mag.max()
    return np.around(mag)


def Gradient_Direction(fx, fy):
    g_dir = np.zeros((fx.shape[0], fx.shape[1]))
    g_dir = np.rad2deg(np.arctan2(fy, fx)) + 180
    return g_dir


def Digitize_angle(Angle):
    quantized = np.zeros((Angle.shape[0], Angle.shape[1]))
    for i in range(Angle.shape[0]):
        for j in range(Angle.shape[1]):
            if 0 <= Angle[i, j] <= 22.5 or 157.5 <= Angle[i, j] <= 202.5 or 337.5 < Angle[i, j] < 360:
                quantized[i, j] = 0
            elif 22.5 <= Angle[i, j] <= 67.5 or 202.5 <= Angle[i, j] <= 247.5:
                quantized[i, j] = 1
            elif 67.5 <= Angle[i, j] <= 122.5 or 247.5 <= Angle[i, j] <= 292.5:
                quantized[i, j] = 2
            elif 112.5 <= Angle[i, j] <= 157.5 or 292.5 <= Angle[i, j] <= 337.5:
                quantized[i, j] = 3
    return quantized


def Non_Max_Supp(qn, magni, D):
    M = np.zeros(qn.shape)
    a, b = np.shape(qn)
    for i in range(a - 1):
        for j in range(b - 1):
            if qn[i, j] == 0:
                if magni[i, j - 1] < magni[i, j] or magni[i, j] > magni[i, j + 1]:
                    M[i, j] = D[i, j]
                else:
                    M[i, j] = 0
            if qn[i, j] == 1:
                if magni[i - 1, j + 1] <= magni[i, j] or magni[i, j] >= magni[i + 1, j - 1]:
                    M[i, j] = D[i, j]
                else:
                    M[i, j] = 0
            if qn[i, j] == 2:
                if magni[i - 1, j] <= magni[i, j] or magni[i, j] >= magni[i + 1, j]:
                    M[i, j] = D[i, j]
                else:
                    M[i, j] = 0
            if qn[i, j] == 3:
                if magni[i - 1, j - 1] <= magni[i, j] or magni[i, j] >= magni[i + 1, j + 1]:
                    M[i, j] = D[i, j]
                else:
                    M[i, j] = 0
    return M


def color(quant, mag):
    color = np.zeros((mag.shape[0], mag.shape[1], 3), np.uint8)
    a, b = np.shape(mag)
    for i in range(a - 1):
        for j in range(b - 1):
            if quant[i, j] == 0:
                if mag[i, j] != 0:
                    color[i, j, 0] = 255
                else:
                    color[i, j, 0] = 0
            if quant[i, j] == 1:
                if mag[i, j] != 0:
                    color[i, j, 1] = 255
                else:
                    color[i, j, 1] = 0
            if quant[i, j] == 2:
                if mag[i, j] != 0:
                    color[i, j, 2] = 255
                else:
                    color[i, j, 2] = 0
            if quant[i, j] == 3:
                if mag[i, j] != 0:
                    color[i, j, 0] = 255
                    color[i, j, 1] = 255

                else:
                    color[i, j, 0] = 0
                    color[i, j, 1] = 0
    return color


def _double_thresholding(g_suppressed, low_threshold, high_threshold):
    g_thresholded = np.zeros(g_suppressed.shape)
    for i in range(0, g_suppressed.shape[0]):  # loop over pixels
        for j in range(0, g_suppressed.shape[1]):
            if g_suppressed[i, j] < low_threshold:  # lower than low threshold
                g_thresholded[i, j] = 0
            elif g_suppressed[i, j] >= low_threshold and g_suppressed[i, j] < high_threshold:  # between thresholds
                g_thresholded[i, j] = 128
            else:  # higher than high threshold
                g_thresholded[i, j] = 255
    return g_thresholded


def _hysteresis(g_thresholded):
    g_strong = np.zeros(g_thresholded.shape)
    for i in range(0, g_thresholded.shape[0]):  # loop over pixels
        for j in range(0, g_thresholded.shape[1]):
            val = g_thresholded[i, j]
            if val == 128:  # check if weak edge connected to strong
                if g_thresholded[i - 1, j] == 255 or g_thresholded[i + 1, j] == 255 or g_thresholded[
                    i - 1, j - 1] == 255 or g_thresholded[i + 1, j - 1] == 255 or g_thresholded[i - 1, j + 1] == 255 or \
                        g_thresholded[i + 1, j + 1] == 255 or g_thresholded[i, j - 1] == 255 or g_thresholded[
                    i, j + 1] == 255:
                    g_strong[i, j] = 255  # replace weak edge as strong
            elif val == 255:
                g_strong[i, j] = 255  # strong edge remains as strong edge
    return g_strong


