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
from canny import  gaussian_kernel
from watershed import watershade_algorithm

"""Apply noise filtering with image integeral"""
original = cv2.imread(r'C:\Users\casper\Desktop\project alpha\dataset\Lower Extremities-Femur AP-1_25_2016-5_34_07 PM-107.JPEG')
padded_image = image_padding(original)
integral = make_integeral(padded_image)
averaged_image = avg_integral(padded_image, integral)
kernal = gaussian_kernel(size=3)
identity = cv2.filter2D(src=averaged_image, ddepth=-1, kernel=kernal)


"""ِApply watershade algorithm"""
watershade_algorithm(identity,averaged_image)



# cv2.imshow('ret',identity)
# cv2.waitKey(0)


# auto detect thresholds
ret3, otsu = cv2.threshold(averaged_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lower = otsu.std() * 0.1
upper = otsu.std() * 0.7



# apply canny detection
edges = cv2.Canny(identity, lower, upper)






