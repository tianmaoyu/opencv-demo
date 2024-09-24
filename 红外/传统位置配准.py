import cv2
import numpy as np

# 读取两张图像
img1 = cv2.imread('imgs2/DJI_20240822122651_0027_T.JPG', 0)  # 灰度模式
img2 = cv2.imread('imgs2/DJI_20240822122652_0027_W.JPG', 0)  # 灰度模式

# 初始化 ORB 特征检测器
orb = cv2.ORB_create()

# 检测和计算特征点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)



# 使用 BFMatcher 进行特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 根据距离对匹配结果进行排序
matches = sorted(matches, key=lambda x: x.distance)


# 绘制匹配结果
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示匹配结果
cv2.imshow('Matched keypoints', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
