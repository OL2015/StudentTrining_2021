import cv2
import matplotlib.pyplot as plt
import numpy as np

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
inpaint_circular_masks(fiber_mask, XY, radius=RADIUS, value = True, inside=True)
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

a = np.zeros((149, 894))
a[0::2 , 0::2] = 50
img = img + a
plt.imshow(img, cmap='gray')
plt.show()