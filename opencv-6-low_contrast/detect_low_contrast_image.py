# skimage.exposre.is_low_contrast:
# This function is used to detect low contrast images by examining an image's historgram
# and determining if the range of brightness spans less than a fractional amount of the full range
from skimage import exposure
from skimage.exposure import is_low_contrast
# imutils.paths.list_images:
# This function is used to grab the paths to images in a directory
from imutils.paths import list_images
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-t", "--thresh", type=float, default=0.35, help="threshold for low contrast")
# The default for the threshold input variable is 0.35, meaning, an image will be considered low contrast
# when the range of brightness spans less than 35% of [0, 255]
args = vars(ap.parse_args())

image_paths = sorted(list(list_images(args["input"])))

for (i, image_path) in enumerate(image_paths):
    # load the input image from disk, and convert to grayscale
    print(f"processing {image_path}")
    image = cv2.imread(image_path)
    # cv2.imshow("original", image)
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("original gray", gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("blurred", blurred)
    edged = cv2.Canny(blurred, 30, 150)
    # cv2.imshow("edged", edged)
    # cv2.waitKey(0)

    text = "Low Contrast: No"
    text_color = (0, 255, 0)

    if is_low_contrast(gray, fraction_threshold=args["thresh"]):
        # update the text to "warning form" when the contrast is low
        text_color = (0, 0, 255)
        text = "Low Contrast: Yes"
    else:
        # otherwise, continue processing

        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # find contours in the edge map generated from `edge`

    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)
    cv2.imshow("original", image)
    cv2.imshow("edged", edged)
    (hist, bins) = exposure.histogram(gray, source_range="dtype")
    hist = hist / hist.max()
    (cdf, bins) = exposure.cumulative_distribution(gray)
    # matplotlib expects RGB images
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    plt.figure()
    plt.title("grayscale histogram")
    plt.xlabel("bins")
    plt.ylabel("# pixels")
    plt.plot( hist)
    plt.plot( cdf)
    plt.xlim([0, 256])
    cv2.waitKey(0)
    plt.show()
