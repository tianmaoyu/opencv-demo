import math

import numpy as np
from PIL import Image


def get_T1():
    x = 12624110.30580023
    y = 2532740.070827089
    z = 50.247

    T1 = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    return T1


def get_Ri():
    Ri = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return Ri


def get_R1():
    yaw = -34.3
    pitch = -47.4
    roll = 0.0

    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def get_S1():
    xo = 12624110.30580023
    yo = 2532740.070827089
    zo = 50.247

    S1 = np.array([
        [-zo, 0, xo, 0],
        [0, -zo, yo, 0],
        [0, 0, 0, 0],
        [0, 0, 1, -zo]
    ], dtype=np.float64)
    return S1


def get_R2():
    yaw = -34.3
    pitch = -90
    roll = 0.0

    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def get_T2():
    x2 = 12624065.1656054
    y2 = 2532806.24386257
    z2 = 50.247

    T2 = np.array([
        [1, 0, 0, x2],
        [0, 1, 0, y2],
        [0, 0, 1, z2],
        [0, 0, 0, 1]
    ])
    return T2


def get_I2():
    pixel = 1.6 / 1000000
    width = 4000
    heigth = 3000

    Scale = np.array([
        [1 / pixel, 0, 0, 0],
        [0, 1 / pixel, 0, 0],
        [0, 0, 1 / pixel, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    # 相片的width 对应传感器的 y, 相片heigth 对应传感器的z 轴
    Translation = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, width/2],
        [0, 0, 1, heigth/2],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    return Translation @ Scale


def get_S2():
    f = 0.0044

    S2 = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, f, 0],
        [1, 0, 0, 0],
    ])

    return S2


def get_I1():
    f = 0.0044
    pixel = 1.6 / 1000000
    width = 4000
    height = 3000

    I1 = np.array([
        [0, 0, 0, f],
        [0, pixel, 0, -width * pixel / 2],
        [0, 0, pixel, -height * pixel / 2],
        [0, 0, 0, 1],
    ])
    return I1


def get_pixelcoord_to_geocoord_matrix():
    I1 = get_I1()
    print(f"I1: {I1.shape}")
    R1 = get_R1()
    print(f"R1: {R1.shape}")
    Ri = get_Ri()
    print(f"Ri: {Ri.shape}")
    T1 = get_T1()
    print(f"T1: {T1.shape}")
    S1 = get_S1()
    print(f"S1: {S1.shape}")
    A1 = T1 @ Ri @ R1
    return S1 @ A1 @ I1


def geocoord_to_pixelcoord_matrix():
    R2 = get_R2()
    print(f"R2: {R2.shape}")
    Ri = get_Ri()
    print(f"Ri: {Ri.shape}")
    T2 = get_T2()
    print(f"T2: {T2.shape}")
    S2 = get_S2()
    print(f"S2: {S2.shape}")
    I2 = get_I2()
    print(f"I2: {I2.shape}")

    A2 = T2 @ Ri @ R2
    return I2 @ S2 @ np.linalg.inv(A2)


def calculate_new_image_size(width, height, matrix):
    # 定义原图的四个角点
    corners = np.array([
        [1, 0, 0, 1],
        [1, width - 1, 0, 1],
        [1, 0, height - 1, 1],
        [1, width - 1, height - 1, 1]
    ])
    # 变换角点
    new_corners = corners @ matrix.T

    new_corners[:, 0] = new_corners[:, 0] / new_corners[:, 3]
    new_corners[:, 1] = new_corners[:, 1] / new_corners[:, 3]
    new_corners[:, 2] = new_corners[:, 2] / new_corners[:, 3]
    new_corners[:, 3] = new_corners[:, 3] / new_corners[:, 3]

    print(new_corners.tolist())

    # 计算新图像的边界
    min_x = np.min(new_corners[:, 1])
    max_x = np.max(new_corners[:, 1])
    min_y = np.min(new_corners[:, 2])
    max_y = np.max(new_corners[:, 2])

    new_width = max_x - min_x
    new_height = max_y - min_y

    return new_width, new_height, min_x, min_y



def transform_image(input_image_path, output_image_path, matrix):

    image = Image.open(input_image_path)
    image_array = np.array(image)
    height, width, channels = image_array.shape

    # 计算新图像的尺寸
    new_width, new_height, min_x, min_y = calculate_new_image_size(width, height, matrix)
    new_width = int(np.ceil(new_width))
    new_height = int(np.ceil(new_height))
    # 生成图片的像素点坐标
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(x_coords)
    coords = np.stack([ones, x_coords, y_coords, ones], axis=2).reshape(-1, 4)

    # 对坐标进行矩阵变换
    new_corners = coords @ matrix.T

    # 归一化 第4维
    new_corners[:, 0] = new_corners[:, 0] / new_corners[:, 3]
    new_corners[:, 1] = new_corners[:, 1] / new_corners[:, 3]
    new_corners[:, 2] = new_corners[:, 2] / new_corners[:, 3]
    new_corners[:, 3] = new_corners[:, 3] / new_corners[:, 3]

    # 将变换后的坐标限制在新图片尺寸范围内
    new_corners[:, 1] -= min_x
    new_corners[:, 2] -= min_y

    new_corners = np.clip(new_corners, [0, 0, 0, 0], [1, new_width - 1, new_height - 1, 1]).astype(int)

    # 创建一个新的空白图片
    new_image_array = np.zeros((new_height, new_width, channels), dtype=image_array.dtype)

    # 将变换后的坐标映射到新图片上，第2，3维
    src_x, src_y = coords[:, 1], coords[:, 2]
    dst_x, dst_y = new_corners[:, 1], new_corners[:, 2]
    new_image_array[dst_y, dst_x] = image_array[src_y, src_x]

    # 将变换后的图片保存
    new_image = Image.fromarray(new_image_array)
    new_image.save(output_image_path)




matrix1 = get_pixelcoord_to_geocoord_matrix()
matrix2 = geocoord_to_pixelcoord_matrix()
transform_matrix = matrix2 @ matrix1
print(transform_matrix.tolist())
# 输入图片路径和输出图片路径
input_image_path = 'pixel_to_gen_img/32.jpeg'
output_image_path = 'pixel_to_gen_img/numpy-image7-透视.jpg'

# 调用函数进行图片变换
transform_image(input_image_path, output_image_path, transform_matrix)
exit(1)
