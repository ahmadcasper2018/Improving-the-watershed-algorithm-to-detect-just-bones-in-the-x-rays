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
from canny import auto_canny

"""Apply noise filtering with image integeral"""
original = cv2.imread(r'C:\Users\casper\Desktop\project alpha\dataset\Lower Extremities-Knee Lat-1_25_2016-5_40_54 PM-720.JPEG')
padded_image = image_padding(original)
integral = make_integeral(padded_image)
averaged_image = avg_integral(padded_image, integral)
#
# edged_image , l ,u = auto_canny(averaged_image)
# edged_image = edged_image.astype('uint8')
ret3, otsu = cv2.threshold(averaged_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# apply automatic Canny edge detection using the computed median
lower = otsu.std() * 0.3
upper = otsu.std()
edge = cv2.Canny(averaged_image, lower, upper)

# print(averaged_image.shape)
# lolo = auto_canny(averaged_image)
# integral_image = integral / integral.max()
# cv2.imshow('original',edged_image)
# cv2.waitKey(0)
#
# cv2.imshow('original',edge)
# cv2.waitKey(0)

plt.imshow(edge,cmap='binary')
plt.show()