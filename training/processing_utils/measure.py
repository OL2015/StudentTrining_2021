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
 #   plt.show()

path = r"..\data\ResultImage.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
shp = img.shape
mean = 0.
scale = 2.
noise1 = np.random.normal(mean, scale, shp)
img1 = img.astype(np.float) + noise1
noise2 = np.random.normal(mean, scale, shp)
img2 = img.astype(np.float) + noise2
diff = img1 - img2 + 128.

std = diff.std()
print(f'std = {std}')

fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
fig.suptitle('Source, noised, and denoised images')
ax0.imshow(img1, interpolation='none', cmap='gray', vmin=0, vmax=255)
ax1.imshow(img2, interpolation='none', cmap='gray', vmin=0, vmax=255)
ax2.imshow(diff, interpolation='none', cmap='gray', vmin=110, vmax=146)
#plt.show()

accum = 0.83
img_denoise1 = denoise_image(img1, accum).astype(np.float)
img_denoise2 = denoise_image(img2, accum).astype(np.float)

diff1 = img_denoise1 - img_denoise2 + 128.

std1 = diff1.std()
print(f'std1 = {std1}')

fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
fig.suptitle('Noised1, Noised2, diff images')
ax0.imshow(img_denoise1, interpolation='none', cmap='gray', vmin=0, vmax=255)
ax1.imshow(img_denoise1 - img1 + 128, interpolation='none', cmap='gray', vmin=0, vmax=255)
ax2.imshow(diff1, interpolation='none', cmap='gray', vmin=0, vmax=255)
#plt.show()

RADIUS = 62
X = 74.49158877473633,222.69205612957614,371.89252348441596,521.0929908392559,670.2934581940956,819.4939255489355
Y = 74.36447724850615,73.62868797671882,73.8928987049315,74.15710943314417,74.42132016135685,73.68553088956952
XY = zip (X, Y)

fiber_mask = np.zeros(shp, dtype = np.bool)
inpaint_circular_masks(fiber_mask, XY, radius=RADIUS, value = True, inside=True)
fiber_ferul = np.logical_not(fiber_mask)

stdfiber = np.std(img1[fiber_mask])
print(f'stdfiber = {stdfiber}')

stdferul = np.std(img1[fiber_ferul])
print(f'stdferul = {stdferul}')

stdfiber1 = np.std(img_denoise1[fiber_mask])
print(f'stdfiber1 = {stdfiber1}')

stdferul1 = np.std(img_denoise1[fiber_ferul])
print(f'stdferul1 = {stdferul1}')

