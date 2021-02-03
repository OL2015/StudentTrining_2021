# -*- coding: utf-8 -*-
"""
Created on Wed Feb 03, 2021
@author: Oleksandr Lytvynenko
"""

import numpy as np
import matplotlib.pyplot as plt
DEBUG = True

def inpaint_circular_masks(img, centers, radius, value = 255, inside=True):
    if inside:
        for center in centers:
            if (center[0]<=0 or center[1]<=0):
                continue
            inpaint_circular_mask(img, center, radius, value, True)
    else:
        mask = np.zeros(img.shape, dtype = np.uint8)
        for center in centers:
            if (center[0]<=0 or center[1]<=0):
                continue
            inpaint_circular_mask(mask, center, radius, 255, True)
        img = img & (mask)
        if DEBUG :
            fig, ax = plt.subplots (2,1)
            ax[0].imshow(mask)
            ax[1].imshow(img)
            plt.show()

def inpaint_circular_mask(img, center=None, radius=None, value = 255, inside=True):
    mask  = get_circular_mask(img, center, radius)
    if inside:
        img[mask==1] = value
    else:
        raise Exception ('Outside inpaint doesnt work!!!')
    return mask

def get_circular_mask(img, center=None, radius=None):
    h, w = img.shape[0], img.shape[1]
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

if __name__ == '__main__':
    sz = (128,1024)
    img = np.ones(sz, dtype=np.uint8) * (np.random.rand(sz[0],sz[1] )*255)
    img1 = np.copy(img)
    for i in range (7):
        inpaint_circular_mask(img1, (128*(i+1), 64), 32, i*(255-32), inside=True)
    fig, ax = plt.subplots(nrows=2, ncols=1 )
    ax[0].set_title('Src ')
    ax[0].imshow(img, interpolation='none', cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('src masked')
    ax[1].imshow(img1, interpolation='none', cmap='gray', vmin=0, vmax=255)
    plt.show()

