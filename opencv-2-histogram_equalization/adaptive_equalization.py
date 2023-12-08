# https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/?_ga=2.53960875.2080009457.1701883228-1842902230.1698424416
import argparse
import cv2
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to image")
ap.add_argument("-c", "--clip", type=float, default=2.0, help="threshold for contrast limiting")
ap.add_argument("-t", "--tile", type=int, default=8, help="tile grid size == divides image into tile x time cells")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("original gray scale", gray)

hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.title("grayscale histogram")
plt.xlabel("bins")
plt.ylabel("# pixels")
plt.plot(hist_original)
plt.xlim([0, 256])

# CLAHE adaptive equalization:
# The idea is to divide an input image into MxN sub regions and to apply equalization onto each sub region
# inputs:
# 1. clip limit: threshold for "contrast limiting". normally, this value should be between 2 - 5. By default, CLAHE uses 20, which is quite high
# If the clip threshold is too high, then local contrast will be maximized, thereby maximizing noise
# 2. tileGridSize: the MxN "tiles" that the input image will be divided into
clahe = cv2.createCLAHE(clipLimit=args["clip"], tileGridSize=(args["tile"], args["tile"]))
equalized = clahe.apply(gray)
cv2.imshow("adaptively equalized", equalized)
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

plt.figure()
plt.title("adaptively equalized histogram")
plt.xlabel("bins")
plt.ylabel("# pixels")
plt.plot(hist_equalized)
plt.xlim([0, 256])
cv2.waitKey(0)
plt.show()