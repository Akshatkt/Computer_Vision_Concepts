import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert BGR to RGB
image = cv2.imread('IMG2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape image to 2D pixel array (rows*cols, channels)
pixels = image.reshape((-1, 3)).astype(np.float32)

# K-means clustering
k = 6  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Rebuild segmented image
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()].reshape(image.shape)

# Display
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1); plt.title("Original Image"); plt.imshow(image)
plt.subplot(1, 2, 2); plt.title("Segmented Image"); plt.imshow(segmented_image)
plt.show()
