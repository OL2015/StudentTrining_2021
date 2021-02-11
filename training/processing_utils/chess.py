import cv2
import matplotlib.pyplot as plt
import numpy as np

path = r"..\data\ResultImage.png"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img = np.ones((149, 894), dtype=np.uint8) * 128
a = np.zeros((149, 894))
a[0::2 , 0::2] = 50
img = img + a
plt.imshow(img, cmap='gray')
plt.show()