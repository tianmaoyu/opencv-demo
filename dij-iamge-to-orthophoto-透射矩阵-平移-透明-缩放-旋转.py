import cv2
import numpy as np

source_list = np.array([[0, 0], [4000, 0], [0, 3000], [4000, 3000]])
target_list = np.array([[-3451, -2199], [7451, -2199], [191, 5199], [3809, 5199]])

homography_matrix, var = cv2.findHomography(source_list, target_list)
img = cv2.imread('pixel_to_gen_img/3.jpeg')
source_height, source_width = img.shape[:2]

# 取得透射矩阵 中的 第三列的 平移 量，x,y- (由于透射变换第三行第三列最后一个是 1，如果不为1 理论上对应缩放)
t_x = homography_matrix[0, 2]
t_y = homography_matrix[1, 2]

target_max = np.max(target_list, axis=0)
target_min = np.min(target_list, axis=0)
target_width = target_max[0] - target_min[0]
target_height = target_max[1] - target_min[1]
target_size=(target_width, target_height)


#缩放后的大小
scale= source_width/target_width
scale_size=(int(target_width*scale), int(target_height*scale))

# 创建平移矩阵
trans_matrix = np.array([
    [1, 0, abs(t_x)],
    [0, 1, abs(t_y)],
    [0, 0, 1]
])
#缩放矩阵
scale_matrix= np.array([
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1]
])
matrix =scale_matrix @trans_matrix @ homography_matrix

output_image = cv2.warpPerspective(img, matrix,scale_size )

# 将图像从BGR转换到BGRA
output_image_bgra = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
# 找到所有黑色像素并设置alpha值为0
black_pixels_mask = np.all(output_image_bgra == [0, 0, 0, 255], axis=-1)
output_image_bgra[black_pixels_mask] = [0, 0, 0, 0]



width,height = scale_size[0],scale_size[1]

rad=np.deg2rad(-34.3)

cos_val = abs(np.cos(rad))
sin_val = abs(np.sin(rad))
# 旋转后新的 长高
new_width = int(width * cos_val+height * sin_val)
new_height = int(height* cos_val + width * sin_val)
# 像回退到原点
trans_matrix_sub= np.array([
    [1, 0, -width/2 ],
    [0, 1, -height/2],
    [0, 0, 1]
])
rotate_matrix= np.array([
    [np.cos(rad), -np.sin(rad),0],
    [np.sin(rad), np.cos(rad),0],
    [0, 0,1]
])
# 平移到图片中间
trans_matrix_add= np.array([
    [1, 0, new_width/2],
    [0, 1, new_height/2],
    [0, 0, 1]
])
result =  trans_matrix_add@rotate_matrix@trans_matrix_sub
affine_matrix = result[:2, :]
rotated_img = cv2.warpAffine(src=output_image_bgra, M=affine_matrix, dsize=(new_width, new_height))

# 中心点开始计算
pixel = 1.6 / 1000_000
altitude=50.24
focal_length=0.0044
gsd= (altitude/focal_length)*pixel
print("gsd:",gsd)
scale_gsd=gsd/scale
print("scale_gsd:",scale_gsd)

# 保存图像
cv2.imwrite('pixel_to_gen_img/rotated_img2.png', rotated_img)
exit(1)