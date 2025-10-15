import matplotlib.pyplot as plt
import cv2

# Read as grayscale
img = cv2.imread('rice.jpg', cv2.IMREAD_GRAYSCALE)

# Plot histogram
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title('Grayscale Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()
