import os

import numpy as np
import cv2
import glob


# 设置棋盘格尺寸
checkerboard_size = (11,8)
square_size = 60  # mm

# 设置终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

# 准备棋盘格的世界坐标点
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# 用于存储所有图像的世界坐标点和图像坐标点
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# 获取所有图片文件
images = glob.glob('chess/*.jpg')

for image_name in images:
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if not ret:
        print(f"失败:{image_name}")
        continue
    # 如果找到足够的角点，添加到列表中
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)
    print(f"完成检测:{image_name}")

    # 画出并显示角点
    img = cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
    cv2.imwrite(f"./draw_images/{os.path.basename(image_name)}", img)


# 标定相机
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 评估标定结果
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print("总误差: ", total_error / len(objpoints))

# 保存标定参数
np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# 使用标定结果进行去畸变
img = cv2.imread('calibration_images/test_image.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 去畸变
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 裁剪图像
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibration_images/calibresult.png', dst)

cv2.imshow('calibresult', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
