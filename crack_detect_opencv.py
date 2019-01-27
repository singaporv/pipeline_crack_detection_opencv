import cv2
import math
import numpy as np
import scipy.ndimage


def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0],[1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

# start calulcation
gray_image = cv2.imread(r'/home/singaporv/Desktop/ML_DL practice/crack_detection/data/crack2.jpg', 0)

with_nmsup = True #apply non-maximal suppression
fudgefactor = 1.8 #with this threshold you can play a little bit
sigma = 21 #for Gaussian Kernel
kernel = 2*math.ceil(2*sigma)+1 #Kernel size

gray_image = gray_image/255.0
blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
# cv2.imshow('a', blur)
gray_image = cv2.subtract(gray_image, blur)
# cv2.imshow('b', gray_image)

# compute sobel response
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3) # to detect edges in x
# cv2.imshow('a', sobelx)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
# cv2.imshow('b', sobely)
mag = np.hypot(sobelx, sobely)
# cv2.imshow('c', mag)
ang = np.arctan2(sobely, sobelx)
# cv2.imshow('d', ang)

# threshold
threshold = 4 * fudgefactor * np.mean(mag)
mag[mag < threshold] = 0
# cv2.imshow('c', mag)


# edges directly
if with_nmsup is False:
    mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5,5),np.uint8)
    # cv2.imshow('1', mag)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('im', result)
    cv2.waitKey()

#apply a non-maximal suppression
else:

    # non-maximal suppression
    mag = orientated_non_max_suppression(mag, ang)
    # create mask
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)

    kernel = np.ones((5,5),np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('im', result)
    cv2.waitKey()
