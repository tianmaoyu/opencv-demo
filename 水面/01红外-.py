import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "src/T.JPG"

# 读取图像
infrared_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 显示红外图像
plt.figure(figsize=(12, 6))
plt.imshow(infrared_img, cmap='gray', vmin=0, vmax=255)
plt.colorbar()  # 添加色条以表示温度范围
plt.title('Infrared Image')
plt.axis('off')
plt.show()