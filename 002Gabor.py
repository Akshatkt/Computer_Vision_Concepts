import cv2
import numpy as np

# Load and convert to grayscale
image = cv2.imread("IMG1.jpg", cv2.IMREAD_GRAYSCALE)

# Create Gabor kernel
ksize = 31
sigma = 4.0
theta = np.pi / 4  # Orientation of Gabor (e.g., 45 degrees)
lamda = 10.0       # Wavelength of the sinusoidal factor
gamma = 0.5        # Spatial aspect ratio
phi = 0            # Phase offset

gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)

# Apply Gabor filter
filtered_img = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)

cv2.imshow("Original Grayscale", image)
cv2.imshow("Gabor Texture", filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
