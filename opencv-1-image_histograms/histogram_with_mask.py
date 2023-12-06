# https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/?_ga=2.254649999.680671124.1701662763-1842902230.1698424416
from matplotlib import pyplot as plt
import numpy as np
import cv2

def plot_histogram(image, title, mask=None):
    # split image into its channels
    channels = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("# of pixels")

    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    
image = cv2.imread("bts.jpg")
plot_histogram(image, "Histogram of original image")
cv2.imshow("original", image)

mask= np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (800, 1360), (1250, 2000), 255, -1)
cv2.imshow("Mask", mask)

# display masked region
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Applying the Mask", masked)
cv2.waitKey(0)

plot_histogram(image, "Histogram of masked region", mask=mask)

(H, S, V) = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
plt.figure()
plt.title("histogram in Hue (HSV space)")
plt.xlabel("bins")
plt.ylabel("# of pixels")
hist = cv2.calcHist([H], [0], mask, [181], [0, 181])
plt.plot(hist)
plt.xlim([0, 181])
plt.show()