import cv2
import numpy as np
import matplotlib.pyplot as plt


import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = 'src/W1.JPG'
# 读取图像并转换为灰度图
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
hist, bins = np.histogram(image.flatten(), 256, [0, 256])
color = ('b','g','r')
for i,col in enumerate(color):
    h = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(h, color = col)
    plt.xlim([0,256])
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
plt.show()