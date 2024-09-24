import cv2
import numpy as np

# 读取两张图像
img1 = cv2.imread('imgs2/DJI_20240822122651_0027_T.JPG', 0)  # 灰度模式
img2 = cv2.imread('imgs2/DJI_20240822122652_0027_W.JPG', 0)  # 灰度模式

# 初始化 SIFT 特征检测器
sift = cv2.SIFT_create()

# 检测和计算特征点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用 FLANN 匹配器进行特征点匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 使用 Lowe's ratio test 筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 0.7 是一个常用的阈值
        good_matches.append(m)

# 提取匹配点
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用 RANSAC 去除错误匹配并计算透视变换矩阵
matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# 应用透视变换到 img1
height, width = img2.shape
aligned_img = cv2.warpPerspective(img1, matrix, (width, height))

# 绘制匹配结果（只绘制通过 RANSAC 筛选的匹配点）
matches_mask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0),  # 绿色为正确匹配
                   singlePointColor=None,
                   matchesMask=matches_mask,  # 只显示通过 RANSAC 筛选的匹配点
                   flags=2)

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

# 显示结果
cv2.imshow('Aligned Image', aligned_img)
cv2.imshow('Matched keypoints', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
