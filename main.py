import cv2
import numpy as np
from matplotlib import pyplot

# 读取两张图片
img1 = cv2.imread('0029_W.jpg')
img2 = cv2.imread('0028_W.jpg')

# 转换为灰度图像
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 创建SIFT对象
sift = cv2.xfeatures2d.SIFT_create()

# 对两张图片进行SIFT特征提取
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 创建FLANN匹配器
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=10)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 对两张图片的SIFT特征进行匹配
matches = flann.knnMatch(des1, des2, k=2)


# 筛选出最佳匹配
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)



# 获取匹配点坐标
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# 计算变换矩阵
M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# 对第二张图片进行变换
h, w = gray2.shape
result = cv2.warpPerspective(img2, M, (w + img1.shape[1], h))

# 将第一张图片拼接到变换后的第二张图片上
result[0:img1.shape[0], 0:img1.shape[1]] = img1

# #保存图片
cv2.imwrite('all.jpg', result)

# 显示拼接后的图片
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
