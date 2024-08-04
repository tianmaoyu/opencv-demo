import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取图像（这里假设图像已经以numpy数组的形式读入，可以使用其他库读取）
# 这里使用随机生成的图像作为示例

image = Image.open("test.jpg")

image=np.array(image)

# image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)

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
image_down = Image.fromarray(downsampled_image)
image_down.save('downsampled_image_output.jpg')

plt.subplot(1, 3, 3)
plt.title('Upsampled Image')
plt.imshow(upsampled_image)
plt.axis('off')
image_up = Image.fromarray(upsampled_image)
image_up.save('upsampled_image_output.jpg')

plt.show()
