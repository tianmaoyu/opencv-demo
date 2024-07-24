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


def get_C():
    f = 0.0044
    C = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, f, 0],
        [1, 0, 0, 0],
    ], dtype=np.float64)
    return C


def get_S():
    f = 0.0044
    pixel = 1.6 / 1000000
    width = 4000
    heigth = 3000

    S = np.array([
        [0, 0, 0, f],
        [0, pixel, 0, -width * pixel / 2],
        [0, 0, pixel, -heigth * pixel / 2],
        [0, 0, 0, 1],
    ])
    return S

def get_I():
    f = 0.0044
    pixel = 1.6 / 1000000
    width = 4000
    heigth = 3000

    S2 = np.array([
        [1, 0, 0, 0],
        [0, 1/pixel, 0, width/2],
        [0, 0, 1/pixel, heigth/2],
        [0, 0, 0, 1],
    ])
    return S2
def get_H():
    C = get_C()
    print(f"C: {C.shape}")

    Ri = get_Ri()
    print(f"Ri: {Ri.shape}")

    T2 = get_T2()
    print(f"T2: {T2.shape}")

    R2 = get_R2()
    print(f"R2: {R2.shape}")

    R1_inv = np.linalg.inv(get_R1())
    print(f"R1_inv: {R1_inv.shape}")

    T1_inv = np.linalg.inv(get_T1())
    print(f"T1_inv: {T1_inv.shape}")

    Ri_inv = np.linalg.inv(get_Ri())
    print(f"Ri_inv: {Ri_inv.shape}")

    S = get_S()
    print(f"S: {S.shape}")

    I = get_I()
    print(f"S_inv: {I.shape}")

    T=get_T()

    H = I @ C @ T2 @ Ri @  R2 @ R1_inv @ Ri_inv @ T1_inv @ S
    # H = I @ C @ Ri  @ R2 @ R1_inv  @ Ri_inv @ S
    print(H)
    return H

def get_T():
    x2 = 12624065.1656054
    y2 = 2532806.24386257
    z2 = 50.247
    x = 12624110.30580023
    y = 2532740.070827089
    z = 50.247
    dif_x=x2-x
    dif_y=y2-y
    dif_z=0
    T = np.array([
        [1, 0, 0, dif_x],
        [0, 1, 0, dif_y],
        [0, 0, 1, dif_z],
        [0, 0, 0, 1]
    ])
    return T



def calculate_new_image_size(width, height, transform_matrix, scale):
    # 定义原图的四个角点
    corners = np.array([
        [1, 0, 0, 1],
        [1, width - 1, 0, 1],
        [1, 0, height - 1, 1],
        [1, width - 1, height - 1, 1]
    ])

    # 变换角点
    transformed_corners = corners @ transform_matrix.T

    # transformed_corners[:, 0] = transformed_corners[:, 0] / transformed_corners[:, 3]
    transformed_corners[:, 1] = transformed_corners[:, 1] / transformed_corners[:, 0]
    transformed_corners[:, 2] = transformed_corners[:, 2] / transformed_corners[:, 0]
    transformed_corners[:, 3] = transformed_corners[:, 3] / transformed_corners[:, 0]

    # 计算新图像的边界
    min_x = np.min(transformed_corners[:, 1])
    max_x = np.max(transformed_corners[:, 1])
    min_y = np.min(transformed_corners[:, 2])
    max_y = np.max(transformed_corners[:, 2])

    new_width = max_x - min_x
    new_height = max_y - min_y

    # return new_width, new_height, min_x, min_y

    print(f'new_width:{new_width} new_height: {new_height} min_x:{min_x} max_y: {min_y}')
    print(f"scale:{scale}")

    new_width *= scale
    new_height *= scale
    min_x *= scale
    min_y *= scale
    print(f'new_width:{new_width} new_height: {new_height} min_x:{min_x} max_y: {min_y}')
    return new_width,new_height,min_x,min_y


def transform_image(input_image_path, output_image_path, transform_matrix):
    scale = 100* 50.247 / 0.0044
    # 读取图片
    image = Image.open(input_image_path)
    image_array = np.array(image)
    height, width, channels = image_array.shape

    # 计算新图像的尺寸
    new_width, new_height, min_x, min_y = calculate_new_image_size(width, height, transform_matrix, scale)

    new_width = int(np.ceil(new_width))
    new_height = int(np.ceil(new_height))

    # 生成图片的像素点坐标
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(x_coords)
    coords = np.stack([ones, x_coords, y_coords, ones], axis=2).reshape(-1, 4)

    # 对坐标进行矩阵变换
    transformed_coords = coords @ transform_matrix.T

    # transformed_coords[:, 0] = transformed_coords[:, 0] / transformed_coords[:, 3]
    transformed_coords[:, 1] = transformed_coords[:, 1] / transformed_coords[:, 0]
    transformed_coords[:, 2] = transformed_coords[:, 2] / transformed_coords[:, 0]
    transformed_coords[:, 3] = transformed_coords[:, 3] / transformed_coords[:, 0]
    transformed_coords = transformed_coords * scale

    # 将变换后的坐标限制在新图片尺寸范围内
    transformed_coords[:, 1] -= min_x
    transformed_coords[:, 2] -= min_y

    transformed_coords = np.clip(transformed_coords, [0, 0, 0, 0], [1, new_width - 1, new_height - 1, 1]).astype(int)

    # 创建一个新的空白图片
    transformed_image_array = np.zeros((new_height, new_width, channels), dtype=image_array.dtype)

    # 将变换后的坐标映射到新图片上
    src_x, src_y = coords[:, 1], coords[:, 2]
    dst_x, dst_y, dst_z = transformed_coords[:, 1], transformed_coords[:, 2], transformed_coords[:, 3]
    transformed_image_array[dst_y, dst_x] = image_array[src_y, src_x]

    # 将变换后的图片保存
    transformed_image = Image.fromarray(transformed_image_array)
    transformed_image.save(output_image_path)





s = 50.247 / 0.0044
s = 100
transform_matrix = get_H()* s
print(transform_matrix.tolist())
# 输入图片路径和输出图片路径
input_image_path = 'pixel_to_gen_img/32.jpeg'
output_image_path = 'pixel_to_gen_img/numpy-image6-透视.jpg'

# 调用函数进行图片变换
transform_image(input_image_path, output_image_path, transform_matrix)
exit(1)
