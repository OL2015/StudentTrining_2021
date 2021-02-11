import cv2
import matplotlib.pyplot as plt
import numpy as np


path = r"..\data\ResultImage.png"
# read image
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img = np.ones((149, 894), dtype=np.uint8) * 128
i = 0
j = 0
while i < 149:
    while j < 894:
        if i > 0 and j > 0:
            a = img[(i-1):(i+2):1 , (j-1):(j+2):1].reshape((9,1))
            print(a)
        else :
            print("edge element")
        i = i + 1
        j = j + 1
