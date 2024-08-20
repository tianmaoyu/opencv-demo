import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'src/W.JPG'

# 加载图像
image = cv2.imread(image_path, 0)  # 读取为灰度图像

# 应用 Canny 边缘检测
edges = cv2.Canny(image, threshold1=50, threshold2=200)

# 应用中值滤波
median_filtered = cv2.medianBlur(edges, 3)  # 使用 5x5 的窗口
# # 定义结构元素
# kernel = np.ones((3, 3), np.uint8)
# # 腐蚀操作
# eroded = cv2.erode(edges, kernel, iterations=1)
# # 膨胀操作
# dilated = cv2.dilate(eroded, kernel, iterations=1)

# 创建一个绘图窗口
plt.figure(figsize=(12, 8))

# 显示原始图像
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')  # 使用灰度 colormap
plt.title('原始图像')
plt.axis('off')

# 显示边缘图像
plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('边缘检测')
plt.axis('off')

# 显示腐蚀后的图像
plt.subplot(2, 2, 3)
plt.imshow(median_filtered, cmap='gray')
plt.title('腐蚀')
plt.axis('off')
#
# # 显示膨胀后的图像
# plt.subplot(2, 2, 4)
# plt.imshow(dilated, cmap='gray')
# plt.title('膨胀')
# plt.axis('off')

# 显示图像
plt.show()