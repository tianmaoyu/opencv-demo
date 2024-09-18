import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'water/unet-2-7.jpg'

# 读取图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图像

# Binarize the image
ret, binary_image = cv2.threshold(image, 25, 255, 0)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing colored contours
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Draw green contours

# Display images
plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(binary_image, cmap='gray')
plt.title('Binarized Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(contour_image)
plt.title('Image with Contours'), plt.xticks([]), plt.yticks([])

plt.show()