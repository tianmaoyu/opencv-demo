import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 定义图像尺寸
width = 640
height = 512

# 打开 .raw 文件
with open('raw/process.raw', 'rb') as f:
    # 读取所有字节数据
    raw_data = f.read()

# 计算每个像素的字节数
bytes_per_pixel = 3  # RGB888

# 将字节流转换为 numpy 数组
data = np.frombuffer(raw_data, dtype=np.uint8)

# 重塑数组为正确的图像形状
# 注意，PIL 和 Matplotlib 默认的图像坐标是从左上角开始，y 轴向下增加
# 因此，我们需要先将数据按列分割，再按行重组
data = data.reshape((height, width, bytes_per_pixel))

# # 使用 PIL 显示图像
# image = Image.fromarray(data)
# image.show()

# 或者使用 Matplotlib 显示图像
plt.imshow(data)
plt.axis("off")
plt.show()

# # 如果需要保存图像
# image = Image.fromarray(data)
# image.save('output.png')