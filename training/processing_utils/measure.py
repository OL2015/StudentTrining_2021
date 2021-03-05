import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA

DEBUG = False

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

def get_dist_mask(img, center=None, radius=None):
    h, w = img.shape[0], img.shape[1]
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    mask1 = mask * dist_from_center
    return mask1

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

def image_sharpness(img, mask):
    img_s = np.copy(img).astype(np.float)
    n = img.shape[0]
    img_s[1:, :] = img_s[0: n - 1, :]
    sharp_img = img.astype(np.float) - img_s
    sharp = np.std(sharp_img[mask])
    return sharp

def image_noise(img1, img2, mask):
    diff1 = img1.astype(np.float) - img2.astype(np.float)
    noise = diff1[mask].std()
    return noise

def plot_images (images, title, subtitles):
    pass


# read images and build masks
path = r"..\data\ResultImage.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
RADIUS = 62
X = 74.49158877473633,222.69205612957614,371.89252348441596,521.0929908392559,670.2934581940956,819.4939255489355
Y = 74.36447724850615,73.62868797671882,73.8928987049315,74.15710943314417,74.42132016135685,73.68553088956952
XY = zip (X, Y)
shp = img.shape
fiber_mask = np.zeros(shp, dtype = np.bool)
inpaint_circular_masks(fiber_mask, XY, radius=RADIUS, value = True, inside=True)
ferrul_mask = np.logical_not(fiber_mask)


# add random noise
mean = 0.
scale = 2.
noise1 = np.random.normal(mean, scale, shp)
img1 = img.astype(np.float) + noise1
noise2 = np.random.normal(mean, scale, shp)
img2 = img.astype(np.float) + noise2

#   denoise images
accum = 6
img_denoise1 = denoise_image(img1, accum).astype(np.float)
img_denoise2 = denoise_image(img2, accum).astype(np.float)

stdnoise = image_noise (img1, img2, fiber_mask)
print(f'stddenoise = {stdnoise}')
stdnoise_denoised =image_noise (img_denoise1, img_denoise2, fiber_mask)
print(f'stdnoise_denoised = {stdnoise_denoised}')

# measure image_sharpness()
src_sharp = image_sharpness(img, ferrul_mask)
print(f'src_sharp = {src_sharp}')

noise_sharp = image_sharpness(img1, ferrul_mask)
print(f'noise_sharp = {noise_sharp}')

denoise_sharp = image_sharpness(img_denoise2, ferrul_mask)
print(f'denoise_sharp = {denoise_sharp}')

