import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 加载图像
image_path = 'src/W.JPG'
image = cv2.imread(image_path)
new_width = 400
new_height = 300

# 调整图像大小
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 将图像从 BGR 转换为 RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像展平以减少计算量
flat_image = np.reshape(image, [-1, 3])

# 使用 K-Means 聚类分割图像
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(flat_image)
labels = kmeans.labels_

# 将标签映射到图像像素
segmented_image = kmeans.cluster_centers_[labels]
segmented_image = np.reshape(segmented_image, image.shape).astype(np.uint8)

# 转换为灰度图
gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

# 找到最大的三个类别并进行多边形拟合
contour_image = image.copy()
largest_contour = None
max_area = 0

for i in range(3):
    # 创建一个掩码
    mask = np.zeros_like(gray_image)
    mask[labels.reshape(image.shape[:2]) == i] = 255

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到了轮廓，则找到面积最大的轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

        # 如果找到了面积最大的轮廓，进行多边形拟合
if largest_contour is not None:
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    cv2.drawContours(contour_image, [approx], -1, (0, 0, 255), 2)  # 红色边框

# 显示结果
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Contour Image')
plt.imshow(contour_image)
plt.axis('off')

plt.show()
