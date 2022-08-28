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
original = cv2.imread(r'test_samples/Lower Extremities-Femur AP-25_01_2016-5_29_55 PM-697.JPEG')
padded_image = image_padding(original)
integral = make_integeral(padded_image)
averaged_image = avg_integral(padded_image, integral)

hell , l ,u = auto_canny(averaged_image)

edge = cv2.Canny(averaged_image, 10, 150)

# print(averaged_image.shape)
# lolo = auto_canny(averaged_image)
# integral_image = integral / integral.max()
cv2.imshow('original',edge)
cv2.waitKey(0)
#

# plt.imshow(auto_canny(averaged_image))
# plt.show()