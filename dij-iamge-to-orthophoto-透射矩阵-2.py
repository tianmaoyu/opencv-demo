
import cv2
import numpy as np


source_list=np.array([[0, 0], [4000, 0], [0, 3000], [4000, 3000]])
target_list=np.array([[500, 500], [3500, 500], [0, 3000], [4000, 3000]])
target_list=np.array([[-3451, -2199], [7451, -2199], [191, 5199], [3809, 5199]])

matrix,var = cv2.findHomography(source_list, target_list)

img = cv2.imread('pixel_to_gen_img/3.jpeg')
dst = cv2.warpPerspective(img, matrix, (8000,11000))
cv2.imwrite('pixel_to_gen_img/3-Homography.jpeg', dst)

print("透射矩阵:", matrix.tolist())

exit(1)

