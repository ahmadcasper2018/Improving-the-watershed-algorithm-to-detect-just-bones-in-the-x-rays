import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import matplotlib.pyplot as plt

def sh(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)

def watershade_algorithm(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #noise removal
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(),255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

def watershade_algorithm(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #noise removal
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(),255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    print('aaaaaaaaaa')
    print(image)
    cv2.imshow('i',image)
    cv2.waitKey(0)
    print('bbbbbbbbb')
    print(markers)
    plt.imshow(markers)
    plt.show()
    cv2.waitKey(0)
    markers = cv2.watershed(image, markers)
    image[markers == -1] = 255

    cv2.imshow('input',sure_fg)
    cv2.waitKey(0)





