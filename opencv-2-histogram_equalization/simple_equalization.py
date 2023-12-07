# https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/?_ga=2.53960875.2080009457.1701883228-1842902230.1698424416
import argparse
import cv2
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("original grayscale", gray)

hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.title("grayscale histogram")
plt.xlabel("bins")
plt.ylabel("# pixels")
plt.plot(hist_original)
plt.xlim([0, 256])

# This function will equalize the input image
# Only grayscale images are valid inputs
# "equalize", meaning, the intensity distribution of the pixels will be evenly spread out over all 255 bins
equalized = cv2.equalizeHist(gray)
cv2.imshow("equalized grayscale", equalized)
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
plt.figure()
plt.title("equalized histogram")
plt.xlabel("bins")
plt.ylabel("# pixels")
plt.plot(hist_equalized)
plt.xlim([0, 256])
plt.show()