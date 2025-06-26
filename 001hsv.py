import cv2
import numpy as np

# Load image
image = cv2.imread("IMG1.jpg")
cv2.imshow("Original", image)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for a color (e.g., blue)
lower_blue = np.array([100, 150, 50])   # Lower bound of blue in HSV
upper_blue = np.array([140, 255, 255])  # Upper bound of blue in HSV

# Create mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Show mask
cv2.imshow("HSV Mask", mask)

# Apply mask to original image
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
