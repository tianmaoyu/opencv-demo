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
output_image_ellipse = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
output_image_hull = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
output_image_rect = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# 为每个轮廓进行拟合并绘制结果  
for contour in contours:
    # 拟合最小外接椭圆  
    if len(contour) >= 5:  # 拟合椭圆要求至少有5个点  
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(output_image_ellipse, ellipse, (0, 255, 0), 2)

    # 计算凸包
    hull = cv2.convexHull(contour)
    cv2.drawContours(output_image_hull, [hull], 0, (0, 255, 0), 2)



    # 拟合最小外接矩形  
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
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
