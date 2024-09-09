import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.graph import rag_mean_color, cut_threshold
from skimage.io import imread
from skimage.util import img_as_float

# 加载图像
image_path = 'src/W.JPG'
image = imread(image_path)
new_width = 400
new_height = 300

# 调整图像大小
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# 将图像转换为浮点数格式
image = img_as_float(image)

# 使用 Felzenszwalb 算法进行初步分割
segments = segmentation.felzenszwalb(image, scale=100, sigma=0.9, min_size=50)

# 创建区域相邻图（RAG）
rag = rag_mean_color(image, segments, mode='distance')

# 使用阈值切割进行分割
labels = cut_threshold(segments, rag, 29)

# 将标签映射到图像像素
segmented_image = color.label2rgb(labels, image, kind='avg', bg_label=0)

# 转换为灰度图
gray_image = cv2.cvtColor((segmented_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

# 找到最大的三个类别并进行多边形拟合
contour_image = (image * 255).astype(np.uint8)
largest_contour = None
max_area = 0

unique_labels = np.unique(labels)
for i in unique_labels:
    # 创建一个掩码
    mask = np.zeros_like(gray_image)
    mask[labels == i] = 255

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
