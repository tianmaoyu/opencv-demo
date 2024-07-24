import cv2
import numpy as np
from pyproj import CRS, Transformer


def web_to_wgs84(x: float, y: float) -> float:
    wgs84 = CRS("EPSG:4326")
    web_mercator = CRS("EPSG:3857")
    transformer = Transformer.from_crs(web_mercator, wgs84)
    longitude, latitude = transformer.transform(x, y)
    return [longitude, latitude]


# 计算 gsd,根据图片区域中心位置计算 四个角
pixel = 1.6 / 1000_000
altitude = 56.24
focal_length = 0.0044
gsd = (altitude / focal_length) * pixel
print("gsd:", gsd)
# scale_gsd=gsd/scale
scale_gsd = 0.04979240727272728
print("scale_gsd:", scale_gsd)
new_width, new_height = 4833, 4496
# 无人机位置
points = [12624065.1656054, 2532806.24386257, 50.247]
center_x = points[0]
center_y = points[1]

# 左下角位置
x1 = center_x - new_width * scale_gsd / 2;
y1 = center_y - new_height * scale_gsd / 2;
# 右上角
x2 = center_x + new_width * scale_gsd / 2;
y2 = center_y + new_height * scale_gsd / 2;

lon_lat_1= web_to_wgs84(x1,y1)
lon_lat_2= web_to_wgs84(x2,y2)
print("左下:",lon_lat_1)
print("右上:",lon_lat_2)

