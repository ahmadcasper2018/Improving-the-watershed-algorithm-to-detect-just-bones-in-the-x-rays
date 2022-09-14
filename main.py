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
original = cv2.imread(r'C:\Users\casper\Desktop\project alpha\dataset\Lower Extremities-Knee AP Cast-1_25_2016-5_48_22 PM-846.JPEG')
padded_image = image_padding(original)
integral = make_integeral(padded_image)
averaged_image = avg_integral(padded_image, integral)
kernal = gaussian_kernel(size=3)
identity = cv2.filter2D(src=averaged_image, ddepth=-1, kernel=kernal)


"""ِApply watershade algorithm"""
sahded = watershade_algorithm(identity,averaged_image)





# auto detect thresholds
ret3, otsu = cv2.threshold(averaged_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lower = otsu.std() * 0.1
upper = otsu.std() * 0.7

hls = cv2.cvtColor(original, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2].astype(np.uint8)
l_channel = hls[:,:,1].astype(np.uint8)

lolo = color_thr(s_channel, l_channel, s_threshold=(0, 255), l_threshold=(140, 220))




plt.imshow(lolo,cmap='gray')
plt.show()

# plt.imshow(sahded)




# apply canny detection
edges = cv2.Canny(identity, lower, upper)

# plt.imshow(edges)

# cv2.imshow('ret',edges)
# cv2.waitKey(0)


# cv2.imshow('ret',sahded)
# cv2.waitKey(0)

