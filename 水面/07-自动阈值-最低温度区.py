import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'src/T.JPG'
# 读取红外图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 自适应阈值分割低温区，并反转二值图像
adaptive_thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# 使用Otsu's 方法进行全局阈值分割，并反转二值图像
_, otsu_thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 查找轮廓
contours_adaptive, _ = cv2.findContours(adaptive_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_otsu, _ = cv2.findContours(otsu_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 按面积排序，找到前五个面积最大的轮廓
contours_adaptive = sorted(contours_adaptive, key=cv2.contourArea, reverse=True)[:5]
contours_otsu = sorted(contours_otsu, key=cv2.contourArea, reverse=True)[:5]

# 创建彩色图像以绘制结果
output_image_adaptive = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
output_image_otsu = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 为每个轮廓进行拟合并绘制结果
for contour in contours_adaptive:
    # 多边形拟合
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(output_image_adaptive, [approx], 0, (0, 255, 0), 2)

    # 计算凸包
    hull = cv2.convexHull(contour)
    cv2.drawContours(output_image_adaptive, [hull], 0, (0, 0, 255), 2)

for contour in contours_otsu:
    # 多边形拟合
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(output_image_otsu, [approx], 0, (0, 255, 0), 2)

    # 计算凸包
    hull = cv2.convexHull(contour)
    cv2.drawContours(output_image_otsu, [hull], 0, (0, 0, 255), 2)

# 显示结果
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(adaptive_thresholded, cmap='gray')
plt.title('Adaptive Thresholded Image')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(output_image_adaptive, cv2.COLOR_BGR2RGB))
plt.title('Adaptive Threshold - Polygon and Convex Hull')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(output_image_otsu, cv2.COLOR_BGR2RGB))
plt.title('Otsu\'s Method - Polygon and Convex Hull')

plt.show()  
