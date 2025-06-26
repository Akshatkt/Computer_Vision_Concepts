import cv2
import numpy as np

# Load image and convert to grayscale
image = cv2.imread("IMG4.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detector
edges = cv2.Canny(gray, 50, 150)

# Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

# Draw the lines on a copy of the original image
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Calculate line start and end points
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Hough Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Read image
# image = cv2.imread('IMG4.jpeg')

# # Convert image to grayscale
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# # Use canny edge detection
# edges = cv2.Canny(gray,50,150,apertureSize=3)

# # Apply HoughLinesP method to 
# # to directly obtain line end points
# lines_list =[]
# lines = cv2.HoughLinesP(
#             edges, # Input edge image
#             1, # Distance resolution in pixels
#             np.pi/180, # Angle resolution in radians
#             threshold=100, # Min number of votes for valid line
#             minLineLength=5, # Min allowed length of line
#             maxLineGap=10 # Max allowed gap between line for joining them
#             )

# # Iterate over points
# for points in lines:
#       # Extracted points nested in the list
#     x1,y1,x2,y2=points[0]
#     # Draw the lines joing the points
#     # On the original image
#     cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
#     # Maintain a simples lookup list for points
#     lines_list.append([(x1,y1),(x2,y2)])
    
# # Save the result image
# cv2.imwrite('detectedLines.png',image)
# # Display the result
# cv2.imshow('Detected Lines',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
