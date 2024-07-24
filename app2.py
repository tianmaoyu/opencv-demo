import cv2
import numpy as np
from matplotlib import pyplot

# 初始化SIFT检测器

# 读取图片
img2 = cv2.imread('0029_W.jpg', 1)
img1 = cv2.imread('0028_W.jpg', 1)

# 读取灰度图片
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

# 找到关键点和描述符
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 使用FLANN进行匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 仅保存良好的匹配
good = []
for m, n in matches:
    if m.distance < 0.3 * n.distance:
        good.append(m)

    # 绘制匹配点
img3 = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, good, None, flags=2)
    # 显示图片
cv2.imshow('img3', img3)

    # 计算单应性矩阵
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 计算图像2变换后的四个角点的位置
h, w = img2_gray.shape
corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
transformed_corners = cv2.perspectiveTransform(corners, M)

# 找到变换后的四个角点的最小和最大x，y坐标
min_x, min_y = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
max_x, max_y = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)

# 计算最终拼接图像的宽度和高度
width = max_x - min_x
height = max_y - min_y

# 调整单应性矩阵，使得变换后的图像在新的拼接图像中居中
M[0, 2] += -min_x
M[1, 2] += -min_y

# 使用单应性矩阵对齐图像
h, w = img1_gray.shape
aligned_img = cv2.warpPerspective(img2, M, (width, height))

cv2.imshow('aligned_img', aligned_img)
cv2.imshow('img1', img1)

cv2.imwrite("aligned_img2.jpg",aligned_img)



# 显示结果
# cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
