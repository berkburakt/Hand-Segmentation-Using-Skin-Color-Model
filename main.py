import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt


def grey_world(nimg):
    # Applying grey_world algorithm
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0] * (mu_g / np.average(nimg[0])), 255)
    nimg[2] = np.minimum(nimg[2] * (mu_g / np.average(nimg[2])), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def plotHistogram(img):
    color = ('b', 'g', 'r')

    # Plotting the histogram values for all R, G, B channels
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def threshold(img, low, high):
    # Detecting lower and upper bounds for threshold
    lowerBound = np.array(low, dtype="uint8")
    upperBound = np.array(high, dtype="uint8")

    # Applying the threshold and return the inverse of it as we need to background to be black
    thresh = cv2.inRange(img, lowerBound, upperBound)
    return cv2.bitwise_not(thresh)


def main():
    for i, filename in enumerate(glob.glob('Dataset-H1/*.png')):
        # Reading all the images from Dataset-H1 folder
        img = cv2.imread(filename)

        # Applying grey world algorithm to each image
        grey_world_img = grey_world(img)

        # Analyzing R, G, B channels to decide threshold values
        # plotHistogram(grey_world_img)

        # Applying the threshold for RGB images
        rgb_thresh = threshold(grey_world_img, [155, 150, 145], [255, 255, 255])

        # Converting the image to YCbCr color space
        ycbCr_img = cv2.cvtColor(grey_world_img, cv2.COLOR_RGB2YCrCb)

        # Analyzing R, G, B channels to decide threshold values
        # plotHistogram(ycbCr_img)

        # Applying the threshold for YCbCr images
        ycbCr_thresh = threshold(ycbCr_img, [150, 125, 120], [255, 140, 255])

        cv2.imshow('Image RGB {}'.format(i), rgb_thresh)
        cv2.imshow('Image YCbCr {}'.format(i), ycbCr_thresh)

        # YCbrCr color space is better for detecting the color ranges for thresholding

    cv2.waitKey()


if __name__ == "__main__":
    main()
