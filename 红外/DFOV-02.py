import math

import numpy
from numpy import arctan

# 红外镜头信息
t_pixel = 12 / 1_000_000  # 像元间距 (米)
t_focal_length = 9.1 / 1000  # 焦距 (米)

w_width = 640  # 图像宽度 (像素)
w_height = 512  # 图像高度 (像素)

# 计算对角距离 (dy) 和焦距 (dx)
dy = math.sqrt((320 * t_pixel) ** 2 + (256 * t_pixel) ** 2)  # 半对角线
dx = t_focal_length

# 计算对角视场角 (DFOV)
dfov = math.atan(dy / dx) * 2 * 180 / math.pi
print(f"对角视场角 (DFOV): {dfov:.2f}°")

# 计算水平视场角 (HFOV)
hfov = math.atan((320 * t_pixel) / dx) * 2 * 180 / math.pi
print(f"水平视场角 (HFOV): {hfov:.2f}°")

# 计算垂直视场角 (VFOV)
vfov = math.atan((256 * t_pixel) / dx) * 2 * 180 / math.pi
print(f"垂直视场角 (VFOV): {vfov:.2f}°")


hfov =numpy.rad2deg(2* arctan(320 * t_pixel/t_focal_length))
vfov =numpy.rad2deg(2* arctan(256 * t_pixel/t_focal_length))
print(hfov,vfov)



import math

# 已知参数
dfov = 61  # 对角视场角 (degrees)
width = 5
height = 4

# 计算对角线的半视场角
half_dfov = dfov / 2
tan_half_dfov = math.tan(math.radians(half_dfov))

# 计算宽高比
aspect_ratio = width / height

# 通过宽高比和对角视场角，求解水平和垂直视场角的半角
tan_half_hfov = tan_half_dfov / math.sqrt(1 + (1 / aspect_ratio ** 2))
tan_half_vfov = tan_half_hfov / aspect_ratio

# 通过反三角函数计算完整的视场角
hfov = 2 * math.degrees(math.atan(tan_half_hfov))
vfov = 2 * math.degrees(math.atan(tan_half_vfov))

# 输出结果
print(f"水平视场角 (HFOV): {hfov:.2f}°")
print(f"垂直视场角 (VFOV): {vfov:.2f}°")
