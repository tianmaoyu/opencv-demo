import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'src/W.JPG'
# 读取图像并转换为灰度图
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# 预处理 - 使用高斯模糊去除噪声
image = cv2.GaussianBlur(image, (5, 5), 0)
# 使用Sobel算子计算梯度
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 计算x方向梯度
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 计算y方向梯度

# 使用Laplacian算子计算梯度
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# 可视化结果
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()