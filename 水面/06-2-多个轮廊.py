import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'src/T.JPG'
# 读取红外图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 使用全局阈值分割低温区
_, thresholded = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 按面积排序，找到前五个面积最大的轮廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# 创建彩色图像以绘制结果
output_image_polygon = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
output_image_hull = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
output_image_edge = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 获取图像尺寸
height, width = image.shape

# 为每个轮廓进行拟合并绘制结果
for contour in contours:
    # 多边形拟合
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(output_image_polygon, [approx], 0, (0, 255, 0), 2)

    # 计算凸包
    hull = cv2.convexHull(contour)
    cv2.drawContours(output_image_hull, [hull], 0, (0, 255, 0), 2)

    # 绘制靠近图像边缘的部分
    for i in range(len(contour)):
        x1, y1 = contour[i][0]
        print(x1, y1) # 如果有边处于 图像边缘
        x2, y2 = contour[(i + 1) % len(contour)][0]
        if (x1 == 0 or x1 == width - 1 or y1 == 0 or y1 == height - 1) or (
                x2 == 0 or x2 == width - 1 or y2 == 0 or y2 == height - 1):
            cv2.line(output_image_edge, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 显示结果
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(output_image_polygon, cv2.COLOR_BGR2RGB))
plt.title('Polygon Approximation')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(output_image_hull, cv2.COLOR_BGR2RGB))
plt.title('Convex Hull')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(output_image_edge, cv2.COLOR_BGR2RGB))
plt.title('Contour Edges')
plt.show()  
