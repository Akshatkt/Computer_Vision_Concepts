import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Load and convert to grayscale
image = cv2.imread("IMG1.jpg", cv2.IMREAD_GRAYSCALE)

# Parameters for LBP
radius = 1  # Distance from the center pixel
n_points = 8 * radius  # Number of surrounding points

# Compute LBP
lbp = local_binary_pattern(image, n_points, radius, method="uniform")

# Display
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(lbp, cmap='gray')
plt.title("LBP Image")

plt.show()
