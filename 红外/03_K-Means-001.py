import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



def read_temperature(raw_path: str, width=640, height=512) -> np.ndarray:
    bit_depth = 16

    # 假设字节序为小端
    byte_order = '<'  # 小端

    # 读取 .raw 文件
    with open(raw_path, 'rb') as f:
        raw_data = f.read()

    # 解析 .raw 数据
    data = np.frombuffer(raw_data, dtype=f'{byte_order}u{bit_depth // 8}')
    data = data.reshape((height, width))
    return data


# raw_path = "raw/DJI_0005_R2.raw"
raw_path = "raw/0002_T.raw"
# 读取图像

data = read_temperature(raw_path)
max_value = np.max(data)
min_value = np.min(data)
average_value = np.mean(data)
print(f"最大值: {max_value}")
print(f"最小值: {min_value}")
print(f"平均值: {average_value}")

image = 255 * (data - min_value) / (max_value - min_value)
image = np.uint8(image)

# 将图像从 BGR 转换为 RGB
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# 将图像展平以减少计算量
flat_image = np.reshape(image, [-1, 3])

# 使用 K-Means 聚类分割图像
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(flat_image)
labels = kmeans.labels_

# 将标签映射到图像像素
segmented_image = kmeans.cluster_centers_[labels]
segmented_image = np.reshape(segmented_image, image.shape).astype(np.uint8)

# 转换为灰度图
gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255),(126, 0, 255), (126, 255, 0), (255, 126, 0),(126, 0, 126), (126, 126, 0), (126, 126, 0)]

# 找到最大的三个类别并进行多边形拟合
contour_image = image.copy()
for i in range(1):
    # 创建一个掩码
    mask = np.zeros_like(gray_image)
    mask[labels.reshape(image.shape[:2]) == i] = 255
    #
    # contour_image[:, :][labels.reshape(image.shape[:2]) != i] = 255

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到了轮廓，则进行多边形拟合
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.001 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        cv2.drawContours(contour_image, [approx], -1, color_list[i], 5)  # 红色边框

# 显示结果
plt.figure(figsize=(15, 5))
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
