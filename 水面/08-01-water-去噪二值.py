import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'water/unetplus-5-4.jpg'
# 读取图像
image = cv2.imread(image_path)
ret1, threshold = cv2.threshold(image, 30, 255, 0)


# # 将图像转换为灰度图像
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊平滑图像
gaussian = cv2.GaussianBlur(image, (5, 5), 5)
ret2, threshold_gaussian = cv2.threshold(gaussian, 30, 255, 0)

medianBlur = cv2.medianBlur(image, 5)
ret3, threshold_medianBlur = cv2.threshold(medianBlur, 30, 255, 0)


# 将BGR格式转换为RGB格式以适应matplotlib
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

# 使用matplotlib显示图像
plt.figure(figsize=(30,25))

#
plt.subplot(3, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.subplot(3, 2,2)
plt.imshow(threshold)
plt.axis('off')


plt.subplot(3, 2, 3)
plt.imshow(gaussian)
plt.title('gaussian')
plt.axis('off')
plt.subplot(3, 2, 4)
plt.imshow(threshold_gaussian)
plt.axis('off')


plt.subplot(3, 2, 5)
plt.imshow(medianBlur)
plt.title('medianBlur')
plt.axis('off')
plt.subplot(3, 2, 6)
plt.imshow(threshold_medianBlur)
plt.axis('off')



plt.show()
