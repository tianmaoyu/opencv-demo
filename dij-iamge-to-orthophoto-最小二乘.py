
import cv2
import numpy as np

# source
x1, y1 = 0, 0
x2, y2 = 4000, 0
x3, y3 = 0, 3000
x4, y4 = 4000, 3000
# target
tx1, ty1 = 500, 500
tx2, ty2 = 3500, 500
tx3, ty3 = 0, 3000
tx4, ty4 = 4000, 3000

source_list=np.array([[0, 0], [4000, 0], [0, 3000], [4000, 3000]])
# target_list=np.array([[-3451, -6583], [7451, -6583], [191, 815], [3809, 815]])
target_list=np.array([[-3451, -2199], [7451, -2199], [191, 5199], [3809, 5199]])

points= np.hstack((source_list,target_list))
print(points)

# # 输入你的数据 样子
# points = [
#     (x1, y1, tx1, ty1),
#     (x2, y2, tx2, ty2),
#     (x3, y3, tx3, ty3),
#     (x4, y4, tx4, ty4),
#     # (x5, y5, tx5, ty5),
# ]
# 构建矩阵
# Ax=Y
A = []
Y = []
for x, y, tx, ty in points:
    A.append([x, y, 1, 0, 0, 0, -x * tx, -y * tx])
    A.append([0, 0, 0, x, y, 1, -x * ty, -y * ty])
    Y.append(tx)
    Y.append(ty)

A = np.array(A)
Y=np.array(Y)


x =np.linalg.inv(A.T@A)@A.T@Y  # 最小二乘法求解公式
matrix2 =np.append(x,1).reshape(3, 3)

x, residuals, rank, singular = np.linalg.lstsq(A, Y, rcond=None)
matrix =np.append(x,1).reshape(3, 3)

img = cv2.imread('pixel_to_gen_img/3.jpeg')
dst = cv2.warpPerspective(img, matrix, (8000,12000))
cv2.imwrite('pixel_to_gen_img/3-1-dst.jpeg', dst)

print("最小二乘解matrix:", matrix.tolist())
print("最小二乘解:", x)
print("残差:", residuals)
print("矩阵的秩:", rank)
print("奇异值:", singular)
exit(1)

