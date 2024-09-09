import numpy as np
import cv2
import matplotlib.pyplot as plt

# 第一张图片的尺寸 (宽度, 高度)
size_A = (4000, 3000)

# 第二张图片的尺寸 (宽度, 高度)
size_B = (1280, 1024)

# 计算图片的中心点 (因为两张图片的中心相同)
center_A = (size_A[0] // 2, size_A[1] // 2)  # 第一张图片的中心点 (2000, 1500)
center_B = (size_B[0] // 2, size_B[1] // 2)  # 第二张图片的中心点 (640, 512)

# 比例因子：宽度和高度方向上的缩放比例
scale_x = size_A[0] / size_B[0]  # 水平方向的比例
scale_y = size_A[1] / size_B[1]  # 垂直方向的比例

# 第二张图片的四个角相对于中心点的偏移量
# 左上 (0, 0), 右上 (1280, 0), 左下 (0, 1024), 右下 (1280, 1024)
offsets_B = np.float32([
    [-center_B[0], -center_B[1]],  # 左上角相对于中心的偏移
    [center_B[0], -center_B[1]],   # 右上角相对于中心的偏移
    [-center_B[0], center_B[1]],   # 左下角相对于中心的偏移
    [center_B[0], center_B[1]]     # 右下角相对于中心的偏移
])

# 通过比例缩放得到第一张图片中对应的角点
# 偏移量通过 scale_x 和 scale_y 缩放，再加回第一张图片的中心点
corners_A = np.float32([
    [center_A[0] + scale_x * offset[0], center_A[1] + scale_y * offset[1]]
    for offset in offsets_B
])

# 打印出对应的角点坐标
print("对应的角点坐标（在第一张图片中）：")
for i, corner in enumerate(corners_A):
    print(f"角点 {i+1}: {corner}")

# 绘制第一张图片，并标记出对应角点
image_A = np.zeros((size_A[1], size_A[0], 3), dtype=np.uint8)  # 创建一个空白图像模拟第一张图片
# for corner in corners_A:
#     cv2.circle(image_A, tuple(corner.astype(int)), 10, (0, 0, 255), 10)  # 用红点标记角点
cv2.polylines(image_A, [corners_A], isClosed=True, color=(0, 255, 0), thickness=10)

# 使用 plt 显示图像
plt.figure(figsize=(10, 7.5))
plt.imshow(cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB))
plt.title('Corresponding Corners in First Image (4000x3000)')
plt.show()
