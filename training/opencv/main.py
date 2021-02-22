import cv2
import matplotlib.pyplot as plt
import numpy as np

path = r"..\data\ResultImage.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

height=149
wight=894
radius = 62
center=(74.49158877473633, 74.36447724850615)

Y, X = np.ogrid[:height , :wight]
dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

mask = dist_from_center <= radius
img1 = np.copy(img) * mask
plt.imshow(img1, cmap='gray')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

fiber_mask = np.zeros (mask.shape, dtype = np.bool)
img[fiber_mask] = 128
plt.imshow(mask, cmap='gray')
plt.show()

