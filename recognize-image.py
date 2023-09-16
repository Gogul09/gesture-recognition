import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

# Constants for frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Minimum contour area to consider as a finger
MIN_CONTOUR_AREA = 1000

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# Create a window to display the camera feed
cv2.namedWindow("Finger Counting", cv2.WINDOW_NORMAL)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, grayimage, threshold=75):
    # threshold the image to get the foreground which is the hand
    thresholded = cv2.threshold(grayimage, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:     
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        
        return (thresholded, segmented)

#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(image, thresholded, segmented):
    # find the convex hull of the segmented hand region
    # which is the maximum contour with respect to area
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distances = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    max_distance = distances[distances.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * max_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # initialize circular_roi with same shape as thresholded image
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    # draw the circular ROI with radius and center point of convex hull calculated above
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    # approach 1 - eliminating wrist
    #cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
    #print(len(cntsSorted[1:])) # gives the count of fingers

    # approach 2 - eliminating wrist
    # loop through the contours found
    for i, c in enumerate(cnts):

        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

# Initialize variables
prev_finger_count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the frame to create a binary image
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finger_count = 0

    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, _ = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    # Calculate the lengths of sides of the triangle
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                    # Calculate the angle using the cosine rule
                    angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

                    # If the angle is less than 90 degrees, it's a finger
                    if angle < np.pi / 2:
                        finger_count += 1
                        cv2.circle(frame, far, 5, [0, 0, 255], -1)

    # Display the finger count on the frame
    cv2.putText(frame, f"Finger Count: {finger_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow("Finger Counting", frame)

    # Check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
