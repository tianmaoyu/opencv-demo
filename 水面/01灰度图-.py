import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path="src/T.JPG"
img = cv2.imread(image_path)
# 转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


plt.figure(figsize=(12, 6))

plt.subplot(121),
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
plt.title('Original Image')
plt.axis('off')

plt.subplot(122),
plt.imshow(gray_img, cmap='gray'),
plt.title('Edges')
plt.axis('off')


plt.show()  # 显示图像