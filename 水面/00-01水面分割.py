import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'src/W.JPG'
# 读取图像
image = cv2.imread(image_path)

# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊平滑图像
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, 100, 150)

# 找到轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建黑色遮罩
mask = np.zeros_like(gray)

# 假设最大的轮廓是水面
largest_contour = max(contours, key=cv2.contourArea)

# 在遮罩上绘制轮廓
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# 创建一个输出图像，仅包含水面部分
water_surface = cv2.bitwise_and(image, image, mask=mask)

# 创建岸边的反遮罩
mask_inv = cv2.bitwise_not(mask)

# 创建一个输出图像，仅包含岸边部分
shore = cv2.bitwise_and(image, image, mask=mask_inv)

# 将BGR格式转换为RGB格式以适应matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
water_surface_rgb = cv2.cvtColor(water_surface, cv2.COLOR_BGR2RGB)
shore_rgb = cv2.cvtColor(shore, cv2.COLOR_BGR2RGB)

# 使用matplotlib显示图像
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(water_surface_rgb)
plt.title('Water Surface')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(shore_rgb)
plt.title('Shore')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.tight_layout()
plt.show()
