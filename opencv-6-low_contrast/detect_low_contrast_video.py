from skimage.exposure import is_low_contrast
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
# By default, the webcam will be accessed (default=""). Otherwise, the video stream provided will be accessed
ap.add_argument("-i", "--input", type=str, default="", help="optional path to video file")
ap.add_argument("-t", "--thresh", type=float, default=0.35, help="threshold for low contrast")
args = vars(ap.parse_args())

# 0 as input for cv2.VideoCapture will allow the webcam to be accessed
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

# read frame from video
(grabbed, frame) = vs.read()
key = cv2.waitKey(1) & 0xFF

# ord() converts a single Unicode character into its integer representation
while grabbed and key != ord("q"):

    # process the frames
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    text = "Low contrast: No"
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
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        # find contours in the edge map generated from `edge`

    cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # stack the output frame and edge map next to each other:

    # The `edged` frame is in binary form (1 channel), whereas the original input frame is in BGR form (3 channels)
    output = np.dstack([edged] * 3) # This step is to make sure that the `edged` image has 3 channels (BGR) to match the input frame's channels
    output = np.hstack([frame, output]) # Concatenate the two frames horizontally

    # show output to screen
    cv2.imshow("Output", output)
    (grabbed, frame) = vs.read()
    key = cv2.waitKey(1) & 0xFF

