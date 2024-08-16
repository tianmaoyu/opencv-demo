import cv2
import numpy as np

# 读取图像
image = cv2.imread('../pixel_to_gen_img/3.jpeg')

# 灰度化和高斯模糊
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 颜色空间转换和阈值分割
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 轮廓检测
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imwrite("../pixel_to_gen_img/water_3.jpeg",image)
# # 显示结果
# cv2.imshow('Detected Water', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
exit(1)