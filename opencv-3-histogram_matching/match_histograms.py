from skimage import exposure
import matplotlib.pyplot as plt
import imutils
import argparse
import cv2

# Idea: take the clor distribution in the reference image and transfer it to the source image
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="path to input source image")
ap.add_argument("-r", "--reference", required=True, help="path to input reference image")
args = vars(ap.parse_args())

src = cv2.imread(args["source"])
ref = cv2.imread(args["reference"])

is_multi_channel = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=is_multi_channel)

cv2.imshow("source", src)
cv2.imshow("reference", ref)
cv2.imshow("matched", matched)
cv2.waitKey(0)

(fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

for (i, image) in enumerate((src, ref, matched)):
    rgb = imutils.opencv2matplotlib(image)

    for (j, color) in enumerate(("red", "green", "blue")):
        # compute histogram for current color channel
        (hist, bins) = exposure.histogram(rgb[..., j], source_range="dtype")
        axs[j, i].plot(bins, hist / hist.max())

        # compute the cumulative distribution function for the current channel and plot it
        (cdf, bins) = exposure.cumulative_distribution(rgb[..., j])
        axs[j, i].plot(bins, cdf)

        # set the y-axis label of the current plot to be the name
        axs[j, 0].set_ylabel(color)

axs[0, 0].set_title("Source")
axs[0, 1].set_title("Reference")
axs[0, 2].set_title("Matched")

plt.tight_layout()
plt.show()