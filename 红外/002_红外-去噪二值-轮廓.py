import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 根据raw  生成 numpy array
def read_temperature(raw_path: str, width=640, height=512) -> np.ndarray:
    bit_depth = 16

    # 假设字节序为小端
    byte_order = '<'  # 小端

    # 读取 .raw 文件
    with open(raw_path, 'rb') as f:
        raw_data = f.read()

    # 解析 .raw 数据
    data = np.frombuffer(raw_data, dtype=f'{byte_order}u{bit_depth // 8}')
    data = data.reshape((height, width))
    return data


raw_path = "raw/0002_T.raw"
# 读取图像

data = read_temperature(raw_path)
# data=cv2.medianBlur(data, 3 )
max_value = np.max(data)
min_value = np.min(data)
average_value = np.mean(data)
print(f"最大值: {max_value}")
print(f"最小值: {min_value}")
print(f"平均值: {average_value}")

image = 255 * (data - min_value) / (max_value - min_value)
image = np.uint8(image)
image= 255-image
# 温度差 0.5 度差
temperature_value= 1
threshold_value = 255 * (average_value - min_value) / (max_value - min_value)

# 使用高斯模糊平滑图像
gaussian = cv2.GaussianBlur(image, (5, 5), 0)
ret, threshold = cv2.threshold(image, threshold_value, 255, 0)

color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0),(126, 0, 255), (126, 255, 0), (255, 126, 0),(126, 0, 126), (126, 126, 0), (126, 126, 0)]

# 查找轮廓
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 筛选出最大的三个轮廓
if len(contours) >= 5:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
else:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 创建一个新的图像用于绘制轮廓
contour_image = threshold.copy()
contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
# 对每个轮廓进行多边形拟合并绘制
for i, contour in enumerate(contours):
    # 像素为单位
    area = cv2.contourArea(contour)
    if (area < 500):
        continue
    print(f"{i}的面积:{area}")
    epsilon = 0.004 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(contour_image, [approx], -1, color_list[i], 5)

# 使用matplotlib显示图像
plt.figure(figsize=(30, 10))

plt.subplot(2, 3, 1)
plt.title('Image')
plt.axis('off')
plt.imshow(image, cmap="gray")

plt.subplot(2, 3, 2)
plt.title('gaussian')
plt.axis('off')
plt.imshow(gaussian, cmap="gray")

plt.subplot(2, 3, 3)
plt.title('threshold')
plt.axis('off')
plt.imshow(threshold, cmap="gray")

plt.subplot(2, 3, 4)
plt.title('contour_image')
plt.axis('off')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

plt.show()
