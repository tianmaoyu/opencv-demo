import os.path
import subprocess
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt


# 生成 raw 文件
def measure_raw(image_path: str, raw_path: str) -> bool:
    program_path = 'lib/dji_irp.exe'

    if not os.path.exists(image_path):
        raise Exception(f"文件不存在:{image_path}")

    result = subprocess.run([
        program_path,
        '-a', "measure",
        '-s', image_path,
        '-o', raw_path,
    ], capture_output=True, text=True)

    print(result.stdout)

    # 如果程序有错误输出，可以打印错误信息
    if result.stderr:
        print(result.stderr)
        return False
    return True


def show_temperature(raw_path: str, width, height, image_path):
    # 假设位深度为 16 位
    bit_depth = 16

    # 假设字节序为小端
    byte_order = '<'  # 小端

    # 读取 .raw 文件
    with open(raw_path, 'rb') as f:
        raw_data = f.read()

    # 解析 .raw 数据
    data = np.frombuffer(raw_data, dtype=f'{byte_order}u{bit_depth // 8}')
    data = data.reshape((height, width))
    # data = data / 10
    max_value = np.max(data)
    min_value = np.min(data)
    average_value = np.mean(data)

    print(f"最大值: {max_value}")
    print(f"最小值: {min_value}")
    print(f"平均值: {average_value}")
    # 显示图像

    mapped_arr = 255 * (data - min_value) / (max_value - min_value)
    mapped_arr = np.uint8(mapped_arr)
    image = Image.fromarray(mapped_arr)
    image.save(image_path)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()


# 根据raw  生成 numpy array
def read_temperature(raw_path: str, width=640, height=512) -> np.ndarray:
    bit_depth = 16

    # 假设字节序为小端
    byte_order = '<'  # 小端

    # 读取 .raw 文件
    with open(raw_path, 'rb') as f:
        raw_data = f.read()

    # 解析 .raw 数据
    data = np.frombuffer(raw_data, dtype=f'{byte_order}u{bit_depth // 8}')
    data = data.reshape((height, width))
    return data


if __name__ == '__main__':
    image_path = "imgs2/DJI_20240423132515_0002_T.JPG"
    raw_path = "raw/0002_T.raw"

    measure_raw(image_path, raw_path)
    image = Image.open(image_path)
    width, height = image.size
    width = 640
    height = 512

    temper_data = read_temperature(raw_path, width, height)


    max_value = np.max(temper_data)
    min_value = np.min(temper_data)
    average_value = np.mean(temper_data)


    threshold_image = np.zeros_like(temper_data)  # 初始化一个全0的数组，与 temper_data 形状相同
    threshold_image[temper_data > average_value] = 255  # 大于平均值的设置为255
    threshold_image[temper_data <= average_value] = 0  # 小于或等于平均值的设置为0

    print(f"最大值: {max_value}")
    print(f"最小值: {min_value}")
    print(f"平均值: {average_value}")

    plt.subplot(1, 2, 1)
    plt.imshow(temper_data, cmap='gray')
    plt.imsave("test.jpg",temper_data, cmap='gray')
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(threshold_image, cmap='gray')
    plt.colorbar()
    plt.axis("off")

    plt.show()
