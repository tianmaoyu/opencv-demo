import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'src/W1.JPG'
# 读取图像
# image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图像

# 定义量化级别
levels = 2
bins = np.linspace(0, 255, levels + 1).astype(int)
quantized_image = np.digitize(image, bins=bins, right=True) * (256 // levels)
quantized_image=quantized_image.astype(np.uint8)

quantized_image[quantized_image==0]=255
quantized_image[quantized_image==128]=0
# 显示原始图像和增强后的图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(quantized_image, cmap='gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])

plt.show()