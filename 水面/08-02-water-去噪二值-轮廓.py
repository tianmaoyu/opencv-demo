import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


image_path = 'water/test.jpg'
# 读取图像
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

# 使用高斯模糊平滑图像
gaussian = cv2.GaussianBlur(image, (5, 5), 0)
ret, threshold = cv2.threshold(gaussian, 30, 255, 0)

color_list=[(0, 0, 255),(0, 255, 0),(255, 0, 0)]

# 查找轮廓
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 筛选出最大的三个轮廓
if len(contours) >= 3:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
else:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)


# 创建一个新的图像用于绘制轮廓
contour_image = threshold.copy()
contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
# 对每个轮廓进行多边形拟合并绘制
for i,contour in enumerate(contours) :
    #像素为单位
    area= cv2.contourArea(contour)
    if(area<2500):
        continue
    print(f"{i}的面积:{area}")
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(contour_image, [approx], -1, color_list[i], 10)



# 使用matplotlib显示图像
plt.figure(figsize=(30,10))

plt.subplot(2, 3, 1)
plt.title('Image')
plt.axis('off')
plt.imshow(image,cmap="gray")

plt.subplot(2, 3,2)
plt.title('gaussian')
plt.axis('off')
plt.imshow(gaussian,cmap="gray")

plt.subplot(2, 3, 3)
plt.title('threshold')
plt.axis('off')
plt.imshow(threshold,cmap="gray")

plt.subplot(2, 3, 4)
plt.title('contour_image')
plt.axis('off')
plt.imshow( cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))


plt.show()
