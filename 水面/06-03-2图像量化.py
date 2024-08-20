import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'src/W1.JPG'

# 读取图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图像

# 定义量化级别
levels = 2
bins = np.linspace(0, 255, levels + 1).astype(int)
quantized_image = np.digitize(image, bins=bins, right=True) * (256 // levels)
quantized_image = quantized_image.astype(np.uint8)
quantized_image[quantized_image == 0] = 255
quantized_image[quantized_image == 128] = 0

# 使用Canny边缘检测
edges = cv2.Canny(quantized_image, 100, 200)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算每个轮廓的面积，并排序
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 创建一个空白图像用于绘制轮廓
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 画出最大的5个面积的轮廓
for i in range(min(5, len(contours))):
    contour = contours[i]
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(contour_image, [approx], 0, (0, 255, 0), 10)  # 画出轮廓，颜色为绿色，厚度为2

# 显示原始图像、量化图像、边缘检测结果和轮廓图像
plt.figure(figsize=(24, 6))
plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2), plt.imshow(quantized_image, cmap='gray')
plt.title('Quantized Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3), plt.imshow(edges, cmap='gray')
plt.title('Edges'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 4), plt.imshow(contour_image)
plt.title('Largest 5 Contours'), plt.xticks([]), plt.yticks([])
plt.show()
