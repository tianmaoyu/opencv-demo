import cv2
import numpy as np

# 读取彩色图片
img1_color = cv2.imread('r_0028_W.jpeg')
img2_color = cv2.imread('r_0029_W.jpeg')

# 转为灰度图片
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

# 使用SIFT算法找出特征点和描述符
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 用KNN算法匹配特征点
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 选出距离较近的匹配
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

    # 计算单应性矩阵
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 计算图像2的四个角经过变换后的坐标
h1, w1 = img1.shape
h2, w2 = img2.shape
corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
transformed_corners = cv2.perspectiveTransform(corners, M)

# 计算图像1和变换后的图像2的四个角的最小和最大坐标
min_x = int(min(np.min(transformed_corners[:, :, 0]), 0))
min_y = int(min(np.min(transformed_corners[:, :, 1]), 0))
max_x = int(max(np.max(transformed_corners[:, :, 0]), w1))
max_y = int(max(np.max(transformed_corners[:, :, 1]), h1))

# 计算新图像的大小
width = int(max_x - min_x)
height = int(max_y - min_y)

# 调整单应性矩阵，使得变换后的图像在新的图像中正确显示
M[0, 2] += -min_x
M[1, 2] += -min_y

# 创建新的图像
result = np.zeros((height, width, 3), dtype=img1_color.dtype)

# 将图像1复制到新的图像中
result[-min_y: h1 - min_y, -min_x: w1 - min_x] = img1_color

# 使用调整后的单应性矩阵对齐图像2，并将其添加到新图像中
aligned_img2 = cv2.warpPerspective(img2_color, M, (width, height))

# all = cv2.addWeighted(result, 0.5, aligned_img2, 0.5, 0)
# aligned_img1 = cv2.warpPerspective(img1_color, M, (width, height))

all=np.where(result==[0,0,0],aligned_img2,result)
# cv2.imshow('result', result)
# cv2.imshow('aligned_img2', aligned_img2)
# cv2.imshow('aligned_img1', aligned_img1)
cv2.imshow('Stitched Image', all)
cv2.waitKey(0)
cv2.destroyAllWindows()
