# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11, 2021
@author:  Illia, Oleksandr Lytvynenko
"""
import numpy as np
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt

DEBUG = False

def features_from_image(img, window_size=3):
    b = np.zeros((img.shape[0]-2, img.shape[1]-2, 9))
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            b[i-1, j-1, :] = img[(i-1):(i+2), (j-1):(j+2)].reshape((9,))
    return b

def train_pca(trainset, accum):
    pca = PCA(accum)
    pca.fit(trainset)
    components = pca.transform(trainset)
    explained_variance = pca.explained_variance_ratio_
    if DEBUG:
        print (pca.explained_variance_ratio_)
        range_ = [r+1 for r in range(len(explained_variance))]

        fig, ax  = plt.subplots(1, 1, constrained_layout=True)
        fig.suptitle('Portion of explained variance by component')
        ax.bar(range_,explained_variance, color="b", alpha=0.4, align="center")
        ax.plot(range_,explained_variance,'ro-')
        for pos, pct in enumerate(explained_variance):
            ax.annotate(str(round(pct,2)), (pos+1,pct+0.007))
        plt.xticks(range_)

        plt.show()
    return pca, components, explained_variance

def inverse_pca(pca, fitset):
    components = pca.transform(fitset)
    fitset_res = pca.inverse_transform(components)
    return fitset_res

def denoise_image(img_src, accum):
    ws = 3
    img = np.copy(img_src)
    features = features_from_image(img, window_size=ws)
    org_shape = features.shape
    features = features.reshape(-1, org_shape[-1])
    pca, components, explained_variance = train_pca(features, accum)
    fitset_res = inverse_pca(pca, features)
    fitset_res = fitset_res.reshape(org_shape)
    img[1:img.shape[0]-1, 1:img.shape[1]-1] = fitset_res[:,:,4]
    img = img.astype(np.uint8)
    return img

if __name__ == '__main__':
    path = r"..\data\ResultImage.png"
    accum = 0.83
    noise_dev = 10.
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    noise1 = np.random.normal(scale=noise_dev, size=img.shape)
    img1 = img + noise1

    img_denoise = denoise_image(img1, accum)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    fig.suptitle('Source, noised, and denoised images')
    ax0.imshow(img, interpolation='none', cmap='gray', vmin=0, vmax=255)
    ax1.imshow(img1, interpolation='none',  cmap='gray', vmin=0, vmax=255)
    ax2.imshow(img_denoise, interpolation='none',  cmap='gray', vmin=0, vmax=255)
    plt.show()



