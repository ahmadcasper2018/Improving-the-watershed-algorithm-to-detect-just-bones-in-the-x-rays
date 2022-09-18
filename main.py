import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from average_filter import (
    image_padding,
    avg_integral,
    make_integeral
)
from canny import  gaussian_kernel , color_thr
from watershed import watershade_algorithm


"""Apply noise filtering with image integeral"""
original = cv2.imread(r'C:\Users\casper\Desktop\project alpha\dataset\Lower Extremities-Femur AP-1_25_2016-5_34_07 PM-107.JPEG')

padded_image = image_padding(original)
integral = make_integeral(padded_image)
averaged_image = avg_integral(padded_image, integral)
kernal = gaussian_kernel(size=3)
identity = cv2.filter2D(src=averaged_image, ddepth=-1, kernel=kernal)



# auto detect thresholds
ret3, otsu = cv2.threshold(averaged_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lower = otsu.std() * 0.1
upper = otsu.std() * 0.7


# apply canny detection
edges = cv2.Canny(identity, lower, upper)


"""ِApply watershade algorithm"""
#sahded = watershade_algorithm(identity,averaged_image)

img_dilation = cv2.dilate(edges, kernal, iterations=1)



# convert to binary by thresholding
ret, binary_map = cv2.threshold(img_dilation,127,255,0)

# do connected components processing
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:,cv2.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 100:   #keep
        result[labels == i + 1] = 255


sahded = watershade_algorithm(result)

# apply canny detection
edges2 = cv2.Canny(sahded, 255, 255)

# define the kernel
morph_kernal = np.ones((3, 3), np.uint8)

# invert the image
invert = cv2.bitwise_not(edges2)

# erode the image
erosion = cv2.erode(invert, morph_kernal,
                    iterations=1)
# dilate the eroded image (opening)
img_dilation = cv2.dilate(erosion, morph_kernal, iterations=1)

after_morph = cv2.bitwise_not(img_dilation)

resized = cv2.resize(erosion, (500, 500))
cv2.imshow("Result", after_morph)
cv2.waitKey(0)
cv2.destroyAllWindows()

