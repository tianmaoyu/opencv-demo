import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open('imgs/T2.JPG')
# 将图像转换为 numpy 数组
image_array = np.array(image)

# 检查图像的维度
print("Image shape:", image_array.shape)
# 假设近红外波段在红色通道，绿光波段在绿色通道
nir_band = image_array[:, :, 0]  # 红色通道
green_band = image_array[:, :, 1]  # 绿色通道

# 计算 NDWI
ndwi = (green_band.astype(float) - nir_band.astype(float)) / ((green_band + nir_band))

# 显示 NDWI 图像
plt.imshow(ndwi, cmap='BrBG')  # 使用绿色-棕色渐变色板
plt.colorbar()
plt.title('NDWI')
plt.show()

# 假彩色合成：将红外通道 (红色) 映射到R，红色映射到G，绿色映射到B
false_color_image = np.dstack((nir_band, image_array[:, :, 0], image_array[:, :, 1]))
# 归一化
false_color_image = false_color_image / np.max(false_color_image)

# 显示假彩色图像
plt.imshow(false_color_image)
plt.title('False Color Composite')
plt.axis('off')
plt.show()
