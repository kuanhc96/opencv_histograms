# https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/?_ga=2.254649999.680671124.1701662763-1842902230.1698424416
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

# split the image into its channels
image = cv2.imread(args["image"])
plt.figure()
plt.axis("off")
# imutils.opencv2matplotlib is a convenience function that will convert an image
# that is suitable in cv2 format (BGR) into an image suitable in matplotlib format (RGB)
plt.imshow(imutils.opencv2matplotlib(image))
plt.title("original image")

channels = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("flattened color histogram")
plt.xlabel("bins")
plt.ylabel("# pixels")

for (channel, color)  in zip(channels, colors):
    # create histogram for current channel and plot it
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

# Create new figure and plot 2D color histogram for the green and blue channels
fig = plt.figure()
ax = fig.add_subplot(131)
hist = cv2.calcHist([channels[1], channels[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)

# Create new figure and plot 2D color histogram for the green and red channels
ax = fig.add_subplot(132)
hist = cv2.calcHist([channels[1], channels[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)
# Create new figure and plot 2D color histogram for the blue and red channels
ax = fig.add_subplot(133)
hist = cv2.calcHist([channels[0], channels[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)

print(f"2D histogram shape: {hist.shape} with {hist.flatten().shape[0]} values")

hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print(f"3D histogram shape: {hist.shape}, with {hist.flatten().shape} values")
# p = ax.imshow(hist, interpolation="nearest") This is an invalid shape for plotting
plt.show()
