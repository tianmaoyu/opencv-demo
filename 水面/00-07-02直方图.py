import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
image_path = 'src/W1.JPG'
# 加载图像
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = np.array(image)

# 计算直方图
hist, bin_edges = np.histogram(gray_image, bins=256, range=(0, 256))

# 将直方图分成10类
num_classes = 10
class_hist, class_bin_edges = np.histogram(gray_image, bins=num_classes, range=(0, 256))

# 打印每个类别的边界值， 每个边界，可以求得 平均值，最大值等
print("Class boundaries:", class_bin_edges)

# 可视化原始直方图和分类后的直方图
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# 绘制原始直方图
ax[0].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', color='blue')
ax[0].set_title('Original Histogram')
ax[0].set_xlabel('Pixel Value')
ax[0].set_ylabel('Frequency')

# 绘制分类后的直方图，class_hist 每个类别的数量
ax[1].bar(class_bin_edges[:-1], class_hist, width=np.diff(class_bin_edges), align='edge', color='green')
ax[1].set_title(f'Classified into {num_classes} Classes')
ax[1].set_xlabel('Class (Pixel Range)')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()