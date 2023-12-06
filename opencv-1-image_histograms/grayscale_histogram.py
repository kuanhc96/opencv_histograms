# https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/?_ga=2.254649999.680671124.1701662763-1842902230.1698424416
from matplotlib import pyplot as plt
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.calcHist:
# Used to construct a histogram for an input image. The input image can be a grayscale image
# with only one channel, or a colored image with 3 channels
# inputs:
# 1 image: the image that the histogram will be computed for
# 2 channels: a list of indices, where the indices of the channels that the histogram is computed for is specified
# 3 mask: a mask that can be used to mask the input region. If a mask is provided, the histogram will be computed for the un-masked region. 
# input mask as None if there is no mask
# 4 histSize: a list of the number of bins used when computing the histogram. the Nth item in the list refers to the bins in the Nth dimension of the histogram
# 5 ranges: the range of possible values in the histogram, expressed as [a, b], where the value b is NON-inclusive
# output: an array of values representing the "frequency"
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# matplotlib expects RGB images
plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

plt.figure()
plt.title("grayscale histogram")
plt.xlabel("bins")
plt.ylabel("# pixels")
plt.plot(hist)
plt.xlim([0, 256])

# divide each value in the histogram by its sum to get a percentage
hist = hist / hist.sum()
plt.figure()
plt.title("grayscale histogram (normalized)")
plt.xlabel("bins")
plt.ylabel("% pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

