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

quantized_image=quantized_image.astype(np.uint8)
# 查找轮廓
contours, hierarchy = cv2.findContours(quantized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 创建彩色图像以绘制结果
output_image_polygon = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
output_image_hull = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# 按面积排序，找到前五个面积最大的轮廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
# 为每个轮廓进行拟合并绘制结果
for contour in contours:
    # 多边形拟合
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(output_image_polygon, [approx], 0, (0, 255, 0), 5)

    # 计算凸包
    hull = cv2.convexHull(contour)
    cv2.drawContours(output_image_hull, [hull], 0, (0, 255, 0), 5)


# 显示原始图像、量化后的图像、二值图像以及带有轮廓的图像
plt.figure(figsize=(16, 8))

plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 4, 2), plt.imshow(quantized_image, cmap='gray')
plt.title('Quantized Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 4, 3), plt.imshow(output_image_polygon, cmap='gray')
plt.title('Binary Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 4, 4), plt.imshow(output_image_hull)
plt.title('Image with Contours'), plt.xticks([]), plt.yticks([])

plt.show()