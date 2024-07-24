import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 读取图像
image = np.array(Image.open("test.jpg"))  # 替换为你的图像路径

# 降采样函数
def downsample(image, scale_factor):
    new_shape = (int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor), image.shape[2])
    downsampled_image = np.zeros(new_shape, dtype=image.dtype)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            downsampled_image[i, j] = image[int(i / scale_factor), int(j / scale_factor)]
    return downsampled_image

# 上采样函数
def upsample(image, scale_factor):
    new_shape = (int(image.shape[0] / scale_factor), int(image.shape[1] / scale_factor), image.shape[2])
    upsampled_image = np.zeros(new_shape, dtype=image.dtype)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            upsampled_image[i, j] = image[int(i * scale_factor), int(j * scale_factor)]
    return upsampled_image

# 设定降采样和上采样的比例
downscale_factor = 0.5
upscale_factor = 2

# 进行降采样和上采样操作
downsampled_image = downsample(image, downscale_factor)
upsampled_image = upsample(image, upscale_factor)

# 放大或缩小图片
image = Image.fromarray(image)
downsampled_image = Image.fromarray(downsampled_image)
upsampled_image = Image.fromarray(upsampled_image)

# 显示结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Downsampled Image')
plt.imshow(downsampled_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Upsampled Image')
plt.imshow(upsampled_image)
plt.axis('off')

plt.show()
