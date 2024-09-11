import math
import numpy as np
import cv2
import matplotlib.pyplot as plt



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

#  d1/d2= tan1/tan2
tan_a = np.tan(np.deg2rad(dfov_A / 2))
tan_b = np.tan(np.deg2rad(dfov_B / 2))
#  c^2=a^2+b^2
c1= np.sqrt(width_A*width_A +height_A*height_A)
c2= (c1/2) * (tan_b/tan_a)
# c2^2= (5x)^2+ (4x)^2 =>
offset_pixel= np.sqrt(c2*c2/41)
offset_x=5* offset_pixel
offset_y=4* offset_pixel


# 得到四个角点
corners_A = np.float32([
    [width_A/2 -offset_x, height_A/2-offset_y],  # 左上角
    [width_A/2 + offset_x,  height_A/2-offset_y],  # 右上角
    [width_A/2 + offset_x,  height_A/2+offset_y],  # 右下角
    [width_A/2 - offset_x,  height_A/2+offset_y],  # 左下角
])

# 打印角点
print("对应 A 图中的角点坐标：")
for i, corner in enumerate(corners_A):
    print(f"角点 {i + 1}: {corner}")

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

cv2.polylines(w_image, [corners_A.astype(int)], isClosed=True, color=(0, 0, 255), thickness=20)

plt.figure(figsize=(28, 10))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(w_image, cv2.COLOR_BGR2RGB))
plt.title('W')
plt.axis('off')

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(t_image, cv2.COLOR_BGR2RGB))
plt.title('T')
plt.axis('off')
plt.show()
