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
# img = np.ones((149, 894), dtype=np.uint8) * 128
noise1 = np.random.normal(scale=12., size=(149, 894) )
img1 = img + noise1

noise2 = np.random.normal(scale=12., size=(149, 894) )
img2 = img + noise2

deltaimg = img1 - img2
noise_std = (img1 - img2).std()
print (noise_std)
# show image with matplotlib.pyplot and cv2
plt.imshow(img, cmap='gray')
plt.show()
plt.imshow(deltaimg, cmap='gray', vmin=-25, vmax=25)
plt.show()

# img = np.ones((149, 894), dtype=np.uint8) * 128

b = np.zeros((img.shape[0]-2, img.shape[1]-2, 9))
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        b[i-1, j-1, :] = img[(i-1):(i+2), (j-1):(j+2)].reshape((9,))
print(b)


