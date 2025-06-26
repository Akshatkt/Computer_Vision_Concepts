# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.feature import graycomatrix, graycoprops
# from sklearn.cluster import KMeans

# # Load grayscale image
# image = cv2.imread('IMG3.jpg', cv2.IMREAD_GRAYSCALE)

# # Compute GLCM texture matrix
# glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# # Extract features
# contrast = graycoprops(glcm, 'contrast').flatten()
# correlation = graycoprops(glcm, 'correlation').flatten()
# texture_features = np.column_stack((contrast, correlation))

# # Apply K-Means
# kmeans = KMeans(n_clusters=3, random_state=42)
# labels = kmeans.fit_predict(texture_features)

# # Reshape to image
# segmented_image = labels.reshape(image.shape)

# # Show images
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1); plt.title("Original Image"); plt.imshow(image, cmap='gray')
# plt.subplot(1,2,2); plt.title("Texture-Based Segmentation"); plt.imshow(segmented_image, cmap='jet')
# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load grayscale image
image = cv2.imread('IMG1.jpg', cv2.IMREAD_GRAYSCALE)

# Define patch size
patch_size = 16  # Adjust as needed
h, w = image.shape
texture_features = []
patch_indices = []

# Extract GLCM features for each patch
for i in range(0, h, patch_size):
    for j in range(0, w, patch_size):
        patch = image[i:i+patch_size, j:j+patch_size]
        if patch.size == 0:  # Skip empty patches
            continue
        glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        texture_features.append([contrast, correlation])
        patch_indices.append((i, j))

# Normalize features
texture_features = np.array(texture_features)
scaler = StandardScaler()
texture_features = scaler.fit_transform(texture_features)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(texture_features)

# Reconstruct segmented image
segmented_image = np.zeros((h, w), dtype=np.uint8)
for idx, (i, j) in enumerate(patch_indices):
    segmented_image[i:i+patch_size, j:j+patch_size] = labels[idx]

# Show images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Texture-Based Segmentation")
plt.imshow(segmented_image, cmap='jet')
plt.show()