# 把一张照片- 转到正射视角下

from pyproj import CRS, Transformer
from sympy import symbols, Matrix, cos, sin
import numpy as np
import math
import pyexiv2


def string_to_float(str: str) -> float:
    str_split = str.split('/')
    return int(str_split[0]) / int(str_split[1])


# 经纬度的 十分秒 转 度
def dms_to_deg(dms_str: str) -> float:
    dms_split = dms_str.split(" ")
    d = string_to_float(dms_split[0])
    m = string_to_float(dms_split[1]) / 60
    s = string_to_float(dms_split[2]) / 3600
    deg = d + m + s
    return deg


# 得到旋转矩阵的定义，旋转轴顺序 ZYX (内旋)
def rotate_matrix(yaw_degree: float, pitch_degree: float, roll_degree) -> Matrix:
    yaw_value = math.radians(yaw_degree)
    pitch_value = math.radians(pitch_degree)
    roll_value = math.radians(roll_degree)

    yaw, pitch, roll = symbols('yaw pitch roll')

    Rz = Matrix([[cos(yaw), -sin(yaw), 0],
                 [sin(yaw), cos(yaw), 0],
                 [0, 0, 1]])

    Ry = Matrix([[cos(pitch), 0, sin(pitch)],
                 [0, 1, 0],
                 [-sin(pitch), 0, cos(pitch)]])

    Rx = Matrix([[1, 0, 0],
                 [0, cos(roll), -sin(roll)],
                 [0, sin(roll), cos(roll)]])

    definition = Rz * Ry * Rx

    matrix = definition.subs({yaw: yaw_value, pitch: pitch_value, roll: roll_value})

    return matrix, definition;


# 相机坐标系 到 地面坐标系的 旋转矩阵 ,先绕Z轴90度，X轴180度
def rotate_matrix_camera_to_geo() -> Matrix:
    matrix, definition = rotate_matrix(90, 0, 180)

    return matrix, definition


# 三点共线，已经知道 O,P 坐标 以及 G（x,y,z）坐标中的z值，计算 x,y 并返回G
def point_from_collinearity(O: np.ndarray, P: np.ndarray, z: float) -> np.ndarray:
    if (P[2] - O[2]) == 0 or (z - P[2]) == 0:
        print("线和平面平行,没有交点")
        return (float('inf'), float('inf'), z)

    λ = (P[2] - O[2]) / (z - P[2])
    x = (P[0] - O[0]) / λ + P[0]
    y = (P[1] - O[1]) / λ + P[1]
    return np.array([x, y, z])


def wgs84_to_web(longitude: float, latitude: float) -> float:
    # WGS84坐标系统
    wgs84 = CRS("EPSG:4326")
    # Web Mercator坐标系统
    web_mercator = CRS("EPSG:3857")
    transformer = Transformer.from_crs(wgs84, web_mercator, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    return x, y


def web_to_wgs84(x: float, y: float) -> float:
    wgs84 = CRS("EPSG:4326")
    web_mercator = CRS("EPSG:3857")
    transformer = Transformer.from_crs(web_mercator, wgs84)
    longitude, latitude = transformer.transform(x, y)
    return longitude, latitude


# 定义，把图像坐标（左上角是原点）像素点- 转到 相机云台初始坐标系下 3维
def image_to_camera(width: int, heigth: int, x: int, y: int, pixel: float, focal_length: float) -> np.ndarray:
    c_x = focal_length
    c_y = (x - width / 2) * pixel
    c_z = (y - heigth / 2) * pixel
    return np.array([c_x, c_y, c_z])


# 定义，把相机云台坐标系下3维  转到  图像坐标（左上角是原点）像素点
def camera_to_image(width: int, height: int, point: np.ndarray, pixel: float) -> np.ndarray:
    c_y = point[1]
    c_z = point[2]
    x = c_y / pixel + (width / 2)
    y = c_z / pixel + (height / 2)
    return np.array([x, y])


# 像素坐标（左上角是原点） 转地理经纬坐标，image_file：图片，pixel_x：像素x,pixel_y:像素y ,pixel像素物理单位m;
def pixelcoord_to_geocoord(image_file: str, pixel: float, pixel_x: int, pixel_y: int) -> dict:
    image = pyexiv2.Image(image_file)
    exif = image.read_exif()
    xmp = image.read_xmp()

    # 焦距,长，宽
    focal_length = string_to_float(exif['Exif.Photo.FocalLength']) / 1000
    width = float(exif['Exif.Photo.PixelXDimension'])
    hegth = float(exif['Exif.Photo.PixelYDimension'])

    # 经纬度，相对高低
    longitude = dms_to_deg(exif["Exif.GPSInfo.GPSLongitude"])
    latitude = dms_to_deg(exif["Exif.GPSInfo.GPSLatitude"])
    altitude = float(xmp['Xmp.drone-dji.RelativeAltitude'])

    # 滚角，俯仰角，偏航角
    yaw = float(xmp['Xmp.drone-dji.GimbalYawDegree'])
    pitch = float(xmp['Xmp.drone-dji.GimbalPitchDegree'])
    roll = float(xmp['Xmp.drone-dji.GimbalRollDegree'])

    #  云台坐标系下的坐标

    pixel_point_in_camera = image_to_camera(width, hegth, pixel_x, pixel_y, pixel, focal_length)
    print(f"相机坐标下：pixel_x:{pixel_x}  pixel_y:{pixel_y} pixel_point_in_camera: {pixel_point_in_camera} ")

    # 云台坐标进行云台旋转，得到旋转后的坐标
    camera_matrix, _ = rotate_matrix(yaw_degree=yaw, pitch_degree=pitch, roll_degree=roll)
    pixel_point_rotated_in_camera = camera_matrix * Matrix(pixel_point_in_camera)
    print(f"旋转后：pixel_point_rotated_in_camera: {pixel_point_rotated_in_camera} ")

    # 相机的坐标，也是平移向量
    x, y = wgs84_to_web(longitude, latitude)
    camera_point_in_geo = np.array([x, y, altitude])

    # 把云台坐标系下的坐标 转到 地面坐标系：旋转+平移向量，
    geo_matrix, _ = rotate_matrix_camera_to_geo()
    pixel_point_in_geo = geo_matrix * pixel_point_rotated_in_camera + Matrix(camera_point_in_geo)

    object_point_in_geo = point_from_collinearity(camera_point_in_geo, pixel_point_in_geo, 0)
    longitude, latitude = web_to_wgs84(object_point_in_geo[0], object_point_in_geo[1])
    return longitude, latitude


# 地理坐标 转像素 坐标
def geocoord_to_pixelcoord(image_file: str, pixel: float, target_longitude: float, target_latitude: float) -> dict:
    image = pyexiv2.Image(image_file)
    exif = image.read_exif()
    xmp = image.read_xmp()

    # 焦距,长，宽
    focal_length = string_to_float(exif['Exif.Photo.FocalLength']) / 1000
    width = float(exif['Exif.Photo.PixelXDimension'])
    heigth = float(exif['Exif.Photo.PixelYDimension'])

    # 经纬度，相对高低
    longitude = dms_to_deg(exif["Exif.GPSInfo.GPSLongitude"])
    latitude = dms_to_deg(exif["Exif.GPSInfo.GPSLatitude"])
    altitude = float(xmp['Xmp.drone-dji.RelativeAltitude'])

    # 滚角，俯仰角，偏航角
    yaw = float(xmp['Xmp.drone-dji.GimbalYawDegree'])
    pitch = float(xmp['Xmp.drone-dji.GimbalPitchDegree'])
    roll = float(xmp['Xmp.drone-dji.GimbalRollDegree'])

    # 相机的平移向量
    x, y = wgs84_to_web(longitude, latitude)
    move_vector = np.array([x, y, altitude])

    # 目标经纬转 web
    target_x, target_y = wgs84_to_web(target_longitude, target_latitude)
    target_point = np.array([target_x, target_y, 0])

    # 把云台坐标系下的坐标 转到 地面坐标系：旋转+平移向量，
    geo_matrix, _ = rotate_matrix_camera_to_geo()
    # 旋转矩阵
    camera_matrix, _ = rotate_matrix(yaw_degree=yaw, pitch_degree=pitch, roll_degree=roll)
    # 逆矩阵 -注意顺序
    target_point = camera_matrix.inv() * geo_matrix.inv() * Matrix(target_point - move_vector)
    target_point = np.array(target_point, dtype=float).T.tolist()[0]

    t = focal_length / target_point[0]
    x_p = t * target_point[0]
    y_p = t * target_point[1]
    z_p = t * target_point[2]
    target_point = np.array([x_p, y_p, z_p]) / pixel

    # print("x:",target_point[1])
    # print("y:",target_point[2])
    # print("x:",target_point[1]+width/2)
    # print("y:",target_point[2]+heigth/2)

    pixel_x = target_point[1] + width / 2
    pixel_y = target_point[2] + heigth / 2
    return pixel_x, pixel_y


# # 批量转化, 图片，传感器像元大小
# def imagecoord_to_geocoord_batch(image_file: str, pixel: float,image_points:np.ndarray) -> np.ndarray:
#     image = pyexiv2.Image(image_file)
#     exif = image.read_exif()
#     xmp = image.read_xmp()
#
#     # 焦距,长，宽
#     focal_length = string_to_float(exif['Exif.Photo.FocalLength']) / 1000
#     width = float(exif['Exif.Photo.PixelXDimension'])
#     height = float(exif['Exif.Photo.PixelYDimension'])
#
#     # 经纬度，相对高低
#     longitude = dms_to_deg(exif["Exif.GPSInfo.GPSLongitude"])
#     latitude = dms_to_deg(exif["Exif.GPSInfo.GPSLatitude"])
#     altitude = float(xmp['Xmp.drone-dji.RelativeAltitude'])
#
#     # 滚角，俯仰角，偏航角
#     yaw = float(xmp['Xmp.drone-dji.GimbalYawDegree'])
#     pitch = float(xmp['Xmp.drone-dji.GimbalPitchDegree'])
#     roll = float(xmp['Xmp.drone-dji.GimbalRollDegree'])
#
#
#     sensor_matrix_1 = np.array([
#         [pixel, 0, -width * pixel / 2],
#         [0, pixel, -height * pixel / 2],
#         [0, 0, focal_length]
#     ])
#
#     sensor_matrix_2 = np.array([
#         [0, 0, 1],
#         [1, 0, 0],
#         [0, 1, 0]
#     ])
#     # 初始相机下的传感器坐标
#     sensor_matrix = sensor_matrix_2 @ sensor_matrix_1
#
#     # 相机- 到参考下坐标原点
#     x, y = wgs84_to_web(longitude, latitude)
#     move_vector = np.array([x, y, altitude])
#
#     Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                    [np.sin(yaw), np.cos(yaw), 0],
#                    [0, 0, 1]])
#     Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                    [0, 1, 0],
#                    [-np.sin(pitch), 0, np.cos(pitch)]])
#     Rx = np.array([[1, 0, 0],
#                    [0, np.cos(roll), -np.sin(roll)],
#                    [0, np.sin(roll), np.cos(roll)]])
#     # 相机坐标系矩阵
#     rotate_matrix = Rz @ Ry @ Rx
#     # 投影矩阵-这个是 点到 相片，无法求其逆矩阵- 无法 相片到点
#     projection_matrix = np.array([
#         [focal_length, 0, 0, 0],
#         [0, focal_length, 0, 0],
#         [0, 0, focal_length, 0],
#         [1, 0, 0, 0],
#     ])
#     # 北-东-地
#     camera_init_matrix = np.array([
#         [0, 1, 0],
#         [1, 0, 0],
#         [0, 0, -1]
#     ])
#
#     # 相机坐标系-矩阵
#     camera_matrix =camera_init_matrix @ rotate_matrix
#     #相机下传感器坐标
#     ccs_sensor_points = sensor_matrix @ image_points
#     #投影后的坐标? 从相片投影到地面 --
#     ccs_object_points = projection_matrix @ ccs_sensor_points
#
#
#     # 转到参考坐标系
#     rcs_object_points = camera_matrix @  ccs_object_points + move_vector
#
#     return rcs_object_points


# 像素大小



pixel = 1.6 / 1000_000
image_file = "pixel_to_gen_img/3.jpeg"
