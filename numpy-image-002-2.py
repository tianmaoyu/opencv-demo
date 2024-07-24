import numpy as np
from PIL import Image

def calculate_new_image_size(width, height, transform_matrix):
    # 定义原图的四个角点
    corners = np.array([
        [ 0, 0],
        [ width - 1, 0],
        [ 0, height - 1],
        [ width - 1, height - 1]
    ])

    # 变换角点
    transformed_corners = corners @ transform_matrix.T

    # 计算新图像的边界
    min_x = np.min(transformed_corners[:, 0])
    max_x = np.max(transformed_corners[:, 0])
    min_y = np.min(transformed_corners[:, 1])
    max_y = np.max(transformed_corners[:, 1])

    return int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)), min_x, min_y

def transform_image(input_image_path, output_image_path, transform_matrix):
    # 读取图片
    image = Image.open(input_image_path)
    image_array = np.array(image)
    height, width, channels = image_array.shape

    # 生成图片的像素点坐标
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    # 计算新图像的尺寸
    new_width, new_height, min_x, min_y = calculate_new_image_size(width, height, transform_matrix)

    # 对坐标进行矩阵变换
    transformed_coords = coords @ transform_matrix.T


    # 将变换后的坐标限制在新图片尺寸范围内 -平移
    transformed_coords[:, 0] -= min_x
    transformed_coords[:, 1] -= min_y
    # 将变换后的坐标限制在图片尺寸范围内
    transformed_coords = np.clip(transformed_coords, [0, 0], [new_width - 1, new_height - 1]).astype(int)

    # 创建一个新的空白图片
    transformed_image_array = np.zeros((new_height, new_width, channels), dtype=image_array.dtype)

    # 将变换后的坐标映射到新图片上
    # 使用矢量化操作代替逐像素循环
    src_x, src_y = coords[:, 0], coords[:, 1]
    dst_x, dst_y = transformed_coords[:, 0], transformed_coords[:, 1]
    transformed_image_array[dst_y, dst_x] = image_array[src_y, src_x]

    # 将变换后的图片保存
    transformed_image = Image.fromarray(transformed_image_array)
    transformed_image.save(output_image_path)


# 定义变换矩阵 (例如：旋转矩阵)
theta = np.radians(30)  # 旋转角度
transform_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# 输入图片路径和输出图片路径
input_image_path = 'pixel_to_gen_img/32.jpeg'
output_image_path = 'pixel_to_gen_img/numpy-image2-2.jpg'

# 调用函数进行图片变换
transform_image(input_image_path, output_image_path, transform_matrix)
exit(1)
