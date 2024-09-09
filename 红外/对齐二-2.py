import numpy as np
import cv2
import matplotlib.pyplot as plt


t_iamge_path = "./imgs2/T.JPG"
w_image_path = "./imgs2/W.JPG"

# 第一张图片的参数
width_A = 4000
height_A = 3000
dfov_A = 84  # 第一张图片的视场角

# 第二张图片的参数
width_B = 1280
height_B = 1024
dfov_B = 61  # 第二张图片的视场角

# 计算每个像素在水平视角下的角度（per-pixel angle）
angle_per_pixel_A = dfov_A / width_A
angle_per_pixel_B = dfov_B / width_B

# 第二张图片的半宽覆盖的角度
half_fov_B = dfov_B / 2

# 假设两张图片的中心点是相同的
# 从中心向外扩展，计算第二张图片的左右边缘在第一张图片中的角度范围
left_angle_B = -half_fov_B
right_angle_B = half_fov_B

# 计算对应第一张图片中对应的像素位置（以水平为例）
left_pixel_A = width_A // 2 + int(left_angle_B / angle_per_pixel_A)
right_pixel_A = width_A // 2 + int(right_angle_B / angle_per_pixel_A)

# 对应第二张图片高度（上下方向）的视角可以用类似的方法计算
vertical_fov_A = dfov_A * (height_A / width_A)  # 计算第一张图片的垂直视角
vertical_fov_B = dfov_B * (height_B / width_B)  # 计算第二张图片的垂直视角

angle_per_pixel_vertical_A = vertical_fov_A / height_A
angle_per_pixel_vertical_B = vertical_fov_B / height_B

# 第二张图片的上下边缘角度
top_angle_B = vertical_fov_B / 2
bottom_angle_B = -vertical_fov_B / 2

# 计算上下边缘在第一张图片中的对应像素
top_pixel_A = height_A // 2 - int(top_angle_B / angle_per_pixel_vertical_A)
bottom_pixel_A = height_A // 2 - int(bottom_angle_B / angle_per_pixel_vertical_A)

# 得到第一张图片中对应第二张图片的四个角点（左上，右上，左下，右下）
corners_A = np.float32([
    [left_pixel_A, top_pixel_A],     # 左上角
    [right_pixel_A, top_pixel_A],    # 右上角
    [right_pixel_A, bottom_pixel_A],  # 右下角
    [left_pixel_A, bottom_pixel_A],  # 左下角
])
corners_A=corners_A.astype(int)
# 打印出对应的角点坐标
print("对应的角点坐标（在第一张图片中）：")
for i, corner in enumerate(corners_A):
    print(f"角点 {i+1}: {corner}")

# 绘制第一张图片，并标记出对应角点
# image_A = np.zeros((height_A, width_A, 3), dtype=np.uint8)  # 创建一个空白图像模拟第一张图片
w_image = cv2.imread(w_image_path)
t_image = cv2.imread(t_iamge_path)

# for corner in corners_A:
#     cv2.circle(image_A, tuple(corner.astype(int)), 10, (0, 0, 255), -1)  # 用红点标记角点
cv2.polylines(w_image, [corners_A], isClosed=True, color=(0,0, 255), thickness=20)
# 使用 plt 显示图像
plt.figure(figsize=(7, 2.5))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(w_image, cv2.COLOR_BGR2RGB))
plt.title('W')
plt.axis('off')

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(t_image, cv2.COLOR_BGR2RGB))
plt.title('T')
plt.axis('off')
plt.show()
