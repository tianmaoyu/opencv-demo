import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
# 图像文件路径
image_path = 'src/W.JPG'
# 加载图像
image = cv2.imread(image_path)

# 将图像从 BGR 转换为 RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像展平
flat_image = np.reshape(image, [-1, 3])

# 估计带宽
bandwidth = estimate_bandwidth(flat_image, quantile=0.2, n_samples=500)

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

# 显示结果
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image.astype(np.uint8))
plt.axis('off')

plt.show()

