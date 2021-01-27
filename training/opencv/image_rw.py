# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27, 2021
@author: Oleksandr Lytvynenko
# read-write and show images with opencv
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


path = r"..\data\ResultImage.png"
# read image
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
print (type(img))
print (img.shape)

# show image with matplotlib.pyplot and cv2
plt.imshow(img, cmap='gray')
cv2.imshow("ResultImage", img)
plt.show()


# masking a rectangular roi (region of interest) inside and outside
x0 = 25
width = 100
y0 = 50
height = 500

# inside mask
mask = np.zeros(img.shape, dtype = np.int )

mask[x0:x0+width, y0:y0+height] = 1
img1 = np.copy(img) * mask
plt.imshow(img1, cmap='gray')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

# todo outside mask
pass
# todo mask with opencv

