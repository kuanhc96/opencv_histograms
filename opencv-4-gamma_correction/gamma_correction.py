from __future__ import print_function
import numpy as np
import argparse
import cv2

def adjust_gamma(image, gamma=1.0):
    # build a look up table that can be used to find
    # the proper pixel value AFTER applying gamma correction
    inv_gamma = 1.0 / gamma
    table = np.empty(256).astype("uint8")
    for i in np.arange(0, 256):
        table[i] = (i / 255.0) ** inv_gamma * 255.0

    return cv2.LUT(image, table)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-g", "--gamma", default=1.0, type=float, help="gamma correction parameter")
args = vars(ap.parse_args())

original = cv2.imread(args["image"])
input_gamma = args["gamma"]
range = np.append(np.arange(0.0, 3.5, 0.5), input_gamma)
for gamma in range:
    if gamma == 1:
        continue
    
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original, gamma=gamma)
    cv2.putText(adjusted, f"g={gamma}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 225), 3)
    cv2.imshow("Images", np.hstack([original, adjusted]))
    cv2.waitKey(0)

