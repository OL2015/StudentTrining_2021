# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08, 2021
@author:
1. Create a mask for fibers on the real image you have.
   Use fiber centers from result.ini file, and guess the radius.
   Read and parse result.ini in you code
2. In-paint the fibers with a background with value 128
3. Add gaussian noise to fibers. The noise should have zero mean and std = 2.

"""

#  import packages

import cv2
import matplotlib.pyplot as plt
import numpy as np
import circular_mask as cm

#  read the image  (lines 13 - 15 in image_rw.py)
path = r"..\data\ResultImage.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

#  read fiber centers and set radius
RADIUS = 62 # in pixels
X = 74.49158877473633,222.69205612957614,371.89252348441596,521.0929908392559,670.2934581940956,819.4939255489355
Y = 74.36447724850615,73.62868797671882,73.8928987049315,74.15710943314417,74.42132016135685,73.68553088956952
XY = zip (X, Y)

#  create mask for fibers
fiber_mask = np.zeros (img.shape, dtype = np.bool)
cm.inpaint_circular_masks(fiber_mask, XY, radius=RADIUS, value = True, inside=True)
plt.imshow(fiber_mask, cmap='gray')
plt.show()

# fill fibers with gray color for the source image
img[fiber_mask] = 128
plt.imshow(img, cmap='gray')
plt.show()

# add noise to fibers
noise = np.random.normal(scale=5., size=img.shape).astype(np.uint8)
img[fiber_mask] += noise[fiber_mask]
plt.imshow(img, cmap='gray')
plt.show()