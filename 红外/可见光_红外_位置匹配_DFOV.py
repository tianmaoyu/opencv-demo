import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_hfov_vfov(dfov, aspect_ratio):
    # 根据 DFOV 计算出 HFOV 和 VFOV
    half_dfov = dfov / 2
    tan_half_dfov = math.tan(math.radians(half_dfov))

    # 宽高比
    tan_half_hfov = tan_half_dfov / math.sqrt(1 + (1 / aspect_ratio ** 2))
    tan_half_vfov = tan_half_hfov / aspect_ratio

    hfov = 2 * math.degrees(math.atan(tan_half_hfov))
    vfov = 2 * math.degrees(math.atan(tan_half_vfov))

    return hfov, vfov


# A 图片参数
dfov_A = 84
aspect_ratio_A = 4 / 3  # 长宽比 4:3
width_A = 4000
height_A = 3000

# B 图片参数
dfov_B = 61
aspect_ratio_B = 5 / 4  # 长宽比 5:4
width_B = 1280
height_B = 1024

# 计算 A 和 B 的 HFOV 和 VFOV
hfov_A, vfov_A = calculate_hfov_vfov(dfov_A, aspect_ratio_A)
hfov_B, vfov_B = calculate_hfov_vfov(dfov_B, aspect_ratio_B)

print(f"A 图 HFOV: {hfov_A:.2f}°，VFOV: {vfov_A:.2f}°")
print(f"B 图 HFOV: {hfov_B:.2f}°，VFOV: {vfov_B:.2f}°")

# 计算 B 图相对于 A 图中心的角度范围（假设两者中心对齐）
half_hfov_B = hfov_B / 2
half_vfov_B = vfov_B / 2

# 计算 A 图每像素对应的角度（水平和垂直）
angle_per_pixel_h_A = hfov_A / width_A
angle_per_pixel_v_A = vfov_A / height_A

# 计算 B 图在 A 图中的四个角点位置
center_x_A = width_A // 2
center_y_A = height_A // 2

# 左上角
left_x_A = center_x_A - int(half_hfov_B / angle_per_pixel_h_A)
top_y_A = center_y_A - int(half_vfov_B / angle_per_pixel_v_A)

# 右上角
right_x_A = center_x_A + int(half_hfov_B / angle_per_pixel_h_A)
top_y_A = center_y_A - int(half_vfov_B / angle_per_pixel_v_A)

# 左下角
left_x_A = center_x_A - int(half_hfov_B / angle_per_pixel_h_A)
bottom_y_A = center_y_A + int(half_vfov_B / angle_per_pixel_v_A)

# 右下角
right_x_A = center_x_A + int(half_hfov_B / angle_per_pixel_h_A)
bottom_y_A = center_y_A + int(half_vfov_B / angle_per_pixel_v_A)

# 得到四个角点
corners_A = np.float32([
    [left_x_A, top_y_A],  # 左上角
    [right_x_A, top_y_A],  # 右上角
    [right_x_A, bottom_y_A], # 右下角
    [left_x_A, bottom_y_A],  # 左下角
])

# 打印角点
print("对应 A 图中的角点坐标：")
for i, corner in enumerate(corners_A):
    print(f"角点 {i + 1}: {corner}")

# 创建 A 图片（用空白图代替）
image_A = np.zeros((height_A, width_A, 3), dtype=np.uint8)

# # 绘制四个角点及连接它们的线
# for corner in corners_A:
#     cv2.circle(image_A, tuple(corner.astype(int)), 10, (0, 0, 255), 20)  # 用红点标记角点
#
# # 连接四边形
# cv2.line(image_A, tuple(corners_A[0].astype(int)), tuple(corners_A[1].astype(int)), (0, 255, 0), 2)
# cv2.line(image_A, tuple(corners_A[1].astype(int)), tuple(corners_A[3].astype(int)), (0, 255, 0), 2)
# cv2.line(image_A, tuple(corners_A[3].astype(int)), tuple(corners_A[2].astype(int)), (0, 255, 0), 2)
# cv2.line(image_A, tuple(corners_A[2].astype(int)), tuple(corners_A[0].astype(int)), (0, 255, 0), 2)
# 显示结果
# plt.figure(figsize=(10, 7.5))
# plt.imshow(cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB))
# plt.title('B 的四个角在 A 图上的投影')
# plt.show()


t_iamge_path = "./imgs2/DJI_20240822122651_0027_T.JPG"
w_image_path = "./imgs2/DJI_20240822122652_0027_W.JPG"

w_image = cv2.imread(w_image_path)
t_image = cv2.imread(t_iamge_path)

cv2.polylines(w_image, [corners_A.astype(int)], isClosed=True, color=(0,0, 255), thickness=20)


plt.figure(figsize=(28, 10))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(w_image, cv2.COLOR_BGR2RGB))
plt.title('W')
plt.axis('off')

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(t_image, cv2.COLOR_BGR2RGB))
plt.title('T')
plt.axis('off')
plt.show()
