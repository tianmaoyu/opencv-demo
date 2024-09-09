import math

# 像素大小
# t_pixel = 12 / 1000_000
# t_focal_length = 9.1 / 1000

# w_pixel = 1.6 / 1000_000
# w_focal_length = 4.4 / 1000
# w_width = 4000
# w_height = 3000
# dy = math.sqrt((2000 * w_pixel) ** 2 + (1500 * w_pixel) ** 2)
# dx =w_focal_length
#
# dfov=math.atan(dy / dx) * 2 * 180 / math.pi
# print(dfov)
# #水平
# hfov=math.atan((2000 * w_pixel) / dx) * 2 * 180 / math.pi
# #垂直
# vfov =math.atan((1500 * w_pixel) / dx) * 2 * 180 / math.pi
# print(vfov,hfov)
#
# print("*"*20)


# 红外镜头信息：  DFOV：61° ;焦距：9.1 mm（等效焦距：40mm）;照片尺寸：640×512;像元间距: 12 um
t_pixel = 12 / 1000_000
t_focal_length = 9.1 / 1000

w_width = 640
w_height = 512
dy = math.sqrt((320 * t_pixel) ** 2 + (256 * t_pixel) ** 2)
dx =t_focal_length

dfov=math.atan(dy / dx) * 2 * 180 / math.pi
print(dfov)
#水平
hfov=math.atan((320 * t_pixel) / dx) * 2 * 180 / math.pi
print(hfov)
#垂直
vfov =math.atan((256 * t_pixel) / dx) * 2 * 180 / math.pi
print(vfov)