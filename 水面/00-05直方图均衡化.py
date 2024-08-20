import cv2
import numpy as np
import matplotlib.pyplot as plt


import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'src/W1.JPG'
# 读取图像并转换为灰度图
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 使用直方图均衡化增强对比度
equ = cv2.equalizeHist(image)

# 显示原始图像和增强后的图像
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(equ, cmap='gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])

plt.show()