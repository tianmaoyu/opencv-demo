import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
image_path = 'src/W1.JPG'
# 加载图像
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 计算直方图
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# 找到峰值
peaks, _ = find_peaks(hist, height=0)

# 找到谷值
inverted_hist = -hist + hist.max()  # 对直方图取反
valleys, _ = find_peaks(inverted_hist, height=0)

# 绘制直方图以及峰值和谷值
plt.figure(figsize=(10, 5))
plt.plot(bins[:-1], hist, label='Histogram')
plt.plot(peaks, hist[peaks], "x", label='Peaks')
plt.plot(valleys, -inverted_hist[valleys], "o", label='Valleys')
plt.legend(loc='upper right')
plt.title("Histogram with Peaks and Valleys")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])
plt.show()

# 输出峰值和谷值的信息
print("Peaks:")
for peak in peaks:
    print(f"Peak at bin {peak}: {hist[peak]}")

print("\nValleys:")
for valley in valleys:
    print(f"Valley at bin {valley}: {-inverted_hist[valley]}")