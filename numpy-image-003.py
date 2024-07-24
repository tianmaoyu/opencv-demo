import numpy as np
from PIL import Image

def transform_image(input_image_path, output_image_path, transform_matrix):
    # 读取图片
    image = Image.open(input_image_path)
    image_array = np.array(image)
    height, width, channels = image_array.shape

    # 生成图片的像素点坐标
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(x_coords)
    coords = np.stack([ones, x_coords, y_coords, ones], axis=2).reshape(-1, 4)

    # 对坐标进行矩阵变换
    transformed_coords = coords @ transform_matrix.T

    # 将变换后的坐标限制在图片尺寸范围内
    transformed_coords = np.clip(transformed_coords, [0, 0, 0, 0], [1, width - 1, height - 1, 1]).astype(int)

    # 创建一个新的空白图片
    transformed_image_array = np.zeros_like(image_array)

    # 将变换后的坐标映射到新图片上
    src_x, src_y = coords[:, 1], coords[:, 2]
    dst_x, dst_y = transformed_coords[:, 1], transformed_coords[:, 2]
    transformed_image_array[dst_y, dst_x] = image_array[src_y, src_x]

    # 将变换后的图片保存
    transformed_image = Image.fromarray(transformed_image_array)
    transformed_image.save(output_image_path)


# 定义变换矩阵 (例如：旋转矩阵)
theta = np.radians(30)  # 旋转角度
transform_matrix = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta), -np.sin(theta), 0],
    [0, np.sin(theta), np.cos(theta), 0],
    [0, 0, 0, 1]
])

# 输入图片路径和输出图片路径
input_image_path = 'pixel_to_gen_img/32.jpeg'
output_image_path = 'pixel_to_gen_img/numpy-image3.jpg'

# 调用函数进行图片变换
transform_image(input_image_path, output_image_path, transform_matrix)
exit(1)
