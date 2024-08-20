import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_contours_and_convex_hulls(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用 Canny 边缘检测
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 准备一个空白图像来绘制轮廓
    result = np.zeros_like(image)

    for cnt in contours:
        # 计算凸包
        hull = cv2.approxPolyDP(cnt)

        # 在结果图像上绘制轮廓
        cv2.drawContours(result, [hull], 0, (0, 255, 0), 2)

    return result, edges


# 假设有一个图像文件路径
image_path = 'src/T.JPG'

# 读取图像
image = cv2.imread(image_path)

# 执行边缘检测和凸包拟合
result, edges = find_contours_and_convex_hulls(image)

# 显示原始图像、边缘图像和凸包拟合结果
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.title('Edges')
plt.axis('off')
plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Convex Hulls')
plt.axis('off')

plt.show()