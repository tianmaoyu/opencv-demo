import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'src/T.JPG'
# 读取红外图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 使用全局阈值分割低温区
# 这里假设低温区较暗，所以我们不需要反转二值化结果
_, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 或者使用自适应阈值
# thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 查找轮廓
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到面积最大的轮廓
max_contour = max(contours, key=cv2.contourArea)

# 拟合最小外接椭圆
ellipse = cv2.fitEllipse(max_contour)
output_image_ellipse = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.ellipse(output_image_ellipse, ellipse, (0, 255, 0), 2)

# 计算凸包
hull = cv2.convexHull(max_contour)
output_image_hull = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image_hull, [hull], 0, (0, 255, 0), 2)

# 拟合最小外接矩形
rect = cv2.minAreaRect(max_contour)
box = cv2.boxPoints(rect)
box = np.int0(box)
output_image_rect = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image_rect, [box], 0, (0, 255, 0), 2)

# 显示结果
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(output_image_ellipse, cv2.COLOR_BGR2RGB))
plt.title('Min Enclosing Ellipse')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(output_image_hull, cv2.COLOR_BGR2RGB))
plt.title('Convex Hull')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(output_image_rect, cv2.COLOR_BGR2RGB))
plt.title('Min Area Rectangle')

plt.show()
