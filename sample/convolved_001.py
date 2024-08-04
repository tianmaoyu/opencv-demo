from PIL import Image
import numpy as np

# 加载图像并转换为灰度
img = Image.open('test.jpg').convert('L')
img_array = np.array(img)

# 定义一个3x3的卷积核
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# 获取图像和核的尺寸
image_height, image_width = img_array.shape
kernel_size = kernel.shape[0]

# 计算零填充的大小
pad_size = kernel_size // 2

# 创建一个填充后的图像数组
padded_image = np.pad(img_array, pad_size, mode='constant')

# 初始化卷积后的图像数组
convolved_image = np.zeros_like(img_array)

# 执行卷积操作
for i in range(image_height):
    for j in range(image_width):
        # 提取当前窗口
        window = padded_image[i:i + kernel_size, j:j + kernel_size]

        # 计算卷积
        convolved_image[i, j] = np.sum(window * kernel)/9

# 正则化结果，确保值在0-255之间
convolved_image = np.clip(convolved_image, 0, 255).astype(np.uint8)

# 将结果转换为PIL图像并保存
result_image = Image.fromarray(convolved_image)
result_image.save('convolved_image.jpg')

# 显示原图和处理后的图
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(convolved_image, cmap='gray')
plt.title('Convolved Image')
plt.axis('off')

plt.show()