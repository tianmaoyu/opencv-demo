import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# 加载图像
image_path = 'src/W2.JPG'  # 替换为你的图像路径
image = cv2.imread(image_path)
new_width = 400
new_height = 300

# 调整图像大小
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 将图像从 BGR 转换为 RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像展平以减少计算量
flat_image = np.reshape(image, [-1, 3])

# 估计带宽
bandwidth = estimate_bandwidth(flat_image, quantile=0.1, n_samples=1000)

# 进行 Mean Shift 聚类
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(flat_image)

# 获取聚类标签
labels = mean_shift.labels_

# 将标签映射到图像像素
segmented_image = mean_shift.cluster_centers_[labels]

# 重塑回原始图像形状
segmented_image = np.reshape(segmented_image, image.shape)

# 转换为灰度图
gray_image = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

# 找到最大的三个类别并进行多边形拟合
contour_image = image.copy()

# 找到每个类别的面积
unique_labels, counts = np.unique(labels, return_counts=True)
sorted_indices = np.argsort(-counts)  # 从大到小排序

# 只处理最大的三个类别
for i in range(min(3, len(sorted_indices))):
    label = unique_labels[sorted_indices[i]]

    # 创建一个掩码
    mask = np.zeros_like(gray_image)
    mask[labels.reshape(image.shape[:2]) == label] = 255

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到了轮廓，则进行多边形拟合
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        epsilon =0.8*0.01 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        cv2.drawContours(contour_image, [approx], -1, (0, 0, 255), 2)  # 红色边框

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
