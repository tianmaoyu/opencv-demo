
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 假设图像大小为 640x480
width = 640
height = 512

# 假设位深度为 16 位
bit_depth = 16

# 假设字节序为小端
byte_order = '<'  # 小端

# 读取 .raw 文件
with open('raw/extract_p_0.raw', 'rb') as f:
    raw_data = f.read()

# 解析 .raw 数据
data = np.frombuffer(raw_data, dtype=f'{byte_order}u{bit_depth//8}')
data = data.reshape((height, width))

# 显示图像
plt.imshow(data, cmap='gray')
plt.show()
