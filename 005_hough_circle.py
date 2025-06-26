import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread("IMG5.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)  # Reduce noise

# Hough Circle Transform
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=0,
    maxRadius=30
)

# Draw the detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)       # circle outline
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)       # center point

cv2.imshow("Hough Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
