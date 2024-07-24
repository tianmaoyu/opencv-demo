import numpy as np

# 定义矩阵A和B
A = np.array([
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 1]
])
B = np.array([
    [1, 0, 13],
    [0, 1, 12],
    [0, 0, 1]
])

# 计算矩阵乘积AB
AB = np.dot(A,B)
print(AB)

from sympy import Matrix, cos, sin, symbols, expand, latex

x, y, z = symbols('x y z')

y = expand((x + 1) ** 2)

print(latex(y))
# 定义符号变量
theta_yaw, theta_pitch, theta_roll = symbols('theta_yaw theta_pitch theta_roll')

Rz = Matrix([[cos(theta_yaw), -sin(theta_yaw), 0],
             [sin(theta_yaw), cos(theta_yaw), 0],
             [0, 0, 1]])

Ry = Matrix([[cos(theta_pitch), 0, sin(theta_pitch)],
             [0, 1, 0],
             [-sin(theta_pitch), 0, cos(theta_pitch)]])

Rx = Matrix([[1, 0, 0],
             [0, cos(theta_roll), -sin(theta_roll)],
             [0, sin(theta_roll), cos(theta_roll)]])

# 组合为总旋转矩阵，假设按照Z-Y-X顺序
R = Rz * Ry * Rx

Rz.subs()
