import numpy
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
with open('raw/measure2.raw', 'rb') as f:
    raw_data = f.read()

# 解析 .raw 数据
data = np.frombuffer(raw_data, dtype=f'{byte_order}u{bit_depth//8}')
data = data.reshape((height, width))
# data=data/10

max_value=np.max(data)
min_value=np.min(data)
average_value=np.mean(data)
average_value2=np.average(data)
print(f"最大值: {max_value}")
print(f"最小值: {min_value}")
print(f"平均值: {average_value}")
print(f"平均值: {average_value2}")
# 显示图像
mapped_arr = 255 * (data - min_value) / (max_value - min_value)
mapped_arr = np.uint8(mapped_arr)
image = Image.fromarray(mapped_arr)
image.save('measure2.jpg')
plt.imshow(data, cmap='gray')
plt.axis("off")
plt.show()
