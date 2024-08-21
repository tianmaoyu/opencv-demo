import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# 加载图像
image_path = 'src/W1.JPG'
image = cv2.imread(image_path)
new_width = 400
new_height = 300

# 调整图像大小
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 将图像从 BGR 转换为 RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像缩小以减少计算量
# 将图像展平
flat_image = np.reshape(image, [-1, 3])

# 估计带宽
bandwidth = estimate_bandwidth(flat_image, quantile=0.1, n_samples=500)

# 进行 Mean Shift 聚类
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(flat_image)

# 获取聚类标签
labels = mean_shift.labels_

# 获取聚类中心
cluster_centers = mean_shift.cluster_centers_

# 将标签映射到图像像素
segmented_image = cluster_centers[labels]

# 重塑回原始图像形状
segmented_image = np.reshape(segmented_image, image.shape)

# 转换为灰度图
gray_image = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

# 二值化
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 找到轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 按面积排序，找到前五个面积最大的轮廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
# 在原始图像上绘制轮廓
contour_image = image.copy()

# 多边形拟合
polygon_contours = []
for contour in contours:
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon_contours.append(approx)
    cv2.drawContours(contour_image, [approx], -1, (0, 0, 255), 2)  # 蓝色边框

# 显示结果
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image.astype(np.uint8))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Contour Image')
plt.imshow(contour_image)
plt.axis('off')

plt.show()
