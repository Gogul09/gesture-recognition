#------------------------------------------------------------
# SEGMENT, RECOGNIZE and COUNT fingers from a single frame
#------------------------------------------------------------

# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, grayimage, threshold=75):
    # threshold the image to get the foreground (hand)
    thresholded = cv2.threshold(grayimage, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image (OpenCV 4.x style)
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None if no contours detected
    if len(cnts) == 0:
        return None
    else:
        # based on contour area, get the maximum contour (the hand)
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(image, thresholded, segmented):
    # convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # maximum euclidean distance from center to extreme points
    distances = pairwise.euclidean_distances(
        [(cX, cY)], 
        Y=[extreme_left, extreme_right, extreme_top, extreme_bottom]
    )[0]
    max_distance = distances[distances.argmax()]

    # radius of circular ROI (80% of max distance)
    radius = int(0.8 * max_distance)

    # circumference of circular ROI
    circumference = (2 * np.pi * radius)

    # circular ROI mask
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # apply circular ROI on thresholded hand
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # find contours in circular ROI (OpenCV 4.x style)
    cnts, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count_fingers = 0

    # loop through contours
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        # count finger only if:
        # 1. Not the wrist
        # 2. Small enough relative to circumference
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count_fingers += 1

    return count_fingers

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # load a hand image
    frame = cv2.imread("resources/hand-sample.jpg")

    # resize the frame
    frame = imutils.resize(frame, width=700)

    # clone the frame
    clone = frame.copy()

    # convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # segment the hand region
    hand = segment(clone, gray)

    if hand is not None:
        (thresholded, segmented) = hand

        # count fingers
        fingers = count(clone, thresholded, segmented)

        # display result
        cv2.putText(clone, f"This is {fingers}", (70, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # show output
    cv2.imshow("Image", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
