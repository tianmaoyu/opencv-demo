from pyproj import CRS, Transformer
from sympy import symbols, Matrix,cos,sin
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
    return np.array([x,y,z])


# def point_from_collinearity(O: np.ndarray, P: np.ndarray, z: float) -> np.ndarray:
#     # 获取O和P的Z坐标
#     O_z = O[2]
#     P_z = P[2]
#
#     if (P_z - O_z) == 0 or (z - P_z) == 0:
#         print("线和平面平行,没有交点")
#         return np.array([float('inf'), float('inf'), z])
#
#     λ = (P_z - O_z) / (z - P_z)
#
#     # 计算x和y坐标（这里假设O和P在同一平面上，即它们的z坐标不同时才进行计算）
#     x = O[0] + λ * (P[0] - O[0])
#     y = O[1] + λ * (P[1] - O[1])
#     # 返回numpy数组形式的结果
#     return np.array([x, y, z])

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
    print(f"相机坐标下：pixel_x:{ pixel_x }  pixel_y:{pixel_y} pixel_point_in_camera: {pixel_point_in_camera} ")

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




# 像素大小
pixel=1.6/1000_000
image_file="pixel_to_gen_img/3.jpeg"
#图片右上角
#(0,0) (4000,0) (4000,3000) (0,3000) (2000,1500)
pixel_x=4000
pixel_y=3000
latitude,longitude=pixelcoord_to_geocoord(image_file,pixel,pixel_x,pixel_y)
print(f"lon={longitude};")
print(f"lat={latitude};")
