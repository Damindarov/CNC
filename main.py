import numpy as np
from math import *
from converter import *

p0 = np.array([0, 0, 0]) # в этой точке строится нормаль
p1 = np.array([0, 2, 0])
p2 = np.array([2, 0, 0])

length_tools = 10
d_Xrot = 5
m1 = np.array([[p1[1] - p0[1], p2[1] - p0[1]], [p1[2] - p0[2], p2[2] - p0[2]]])
m2 = np.array([[p1[0] - p0[0], p2[0] - p0[0]], [p1[2] - p0[2], p2[2] - p0[2]]])
m3 = np.array([[p1[0] - p0[0], p2[0] - p0[0]], [p1[1] - p0[1], p2[1] - p0[1]]])

det_m1 = np.linalg.det(m1)
det_m2 = np.linalg.det(m2)
det_m3 = np.linalg.det(m3)

# A, B, C, D ур-е плоскости
equation = np.array([det_m1, -det_m2, det_m3, -p0[0]*det_m1 + p0[1]*det_m2 - p0[2]*det_m3])

# вектор нормали к плоскости
normal = np.array([equation[0], equation[1], equation[2]])

# длина нормального ветора
length_normal = sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

# направляющие косинусы
direct_cos = np.array([normal[0]/length_normal, normal[1]/length_normal, normal[2]/length_normal])

# print(p0[0] - length_tools*direct_cos[0], p0[1] - length_tools*direct_cos[1], p0[2] - length_tools*direct_cos[2])
# координата центра вращающейся головы станка
x_head = np.array([p0[0] - length_tools*direct_cos[0], p0[1] - length_tools*direct_cos[1], p0[2] - length_tools*direct_cos[2]])
# print(x_head)
# угол для головы станка
angle_head = np.array([np.arcsin(direct_cos[0]), np.arcsin(direct_cos[1])])
# print(angle_head)

# print(np.linalg.det(m1), np.linalg.det(m2), np.linalg.det(m3))

import sympy as sp
x, y = sp.symbols('x, y')
# paraboloid = sp.lambdify((x, y), -(0.1*x**2 + 0.1*y**2))
# points = np.linspace(-10, 10, 90)
x = np.linspace(-10, 10, 16)
y = np.linspace(-10, 10, 16)
#
# z = 0.1*x**2 + 0.1*y**2
# print(z)
data = []
data_head_points = []
data_head_points_1 = []
names_cos = []
# data_converted = []
# data_converted_head_points = []
for i in range(len(x)-1):
    for j in range(len(y)-1):
        data.append([x[i], y[j], -(0.1*x[i]**2 + 0.1*y[j]**2)])
        p0 = np.array([x[i], y[j], -(0.1*x[i]**2 + 0.1*y[j]**2)])
        p1 = np.array([x[i+1], y[j], -(0.1*x[i+1]**2 + 0.1*y[j]**2)])
        p2 = np.array([x[i], y[j+1], -(0.1*x[i]**2 + 0.1*y[j+1]**2)])

        m1 = np.array([[p1[1] - p0[1], p2[1] - p0[1]], [p1[2] - p0[2], p2[2] - p0[2]]])
        m2 = np.array([[p1[0] - p0[0], p2[0] - p0[0]], [p1[2] - p0[2], p2[2] - p0[2]]])
        m3 = np.array([[p1[0] - p0[0], p2[0] - p0[0]], [p1[1] - p0[1], p2[1] - p0[1]]])

        det_m1 = np.linalg.det(m1)
        det_m2 = np.linalg.det(m2)
        det_m3 = np.linalg.det(m3)

        # A, B, C, D ур-е плоскости
        equation = np.array([det_m1, -det_m2, det_m3, -p0[0] * det_m1 + p0[1] * det_m2 - p0[2] * det_m3])

        # вектор нормали к плоскости
        normal = np.array([equation[0], equation[1], equation[2]])

        # длина нормального ветора
        length_normal = sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)

        # направляющие косинусы
        direct_cos = np.array([normal[0] / length_normal, normal[1] / length_normal, normal[2] / length_normal])

        # print(p0[0] - length_tools*direct_cos[0], p0[1] - length_tools*direct_cos[1], p0[2] - length_tools*direct_cos[2])
        # координата центра вращающейся головы станка
        # x_head = np.array([p0[0] + length_tools * direct_cos[0], p0[1] + length_tools * direct_cos[1],
        #                    p0[2] + length_tools * direct_cos[2]])
        data_head_points.append([p0[0] + length_tools * direct_cos[0], p0[1] + length_tools * direct_cos[1],
                           p0[2] + length_tools * direct_cos[2]])

        length_new_vect = sqrt(length_tools**2 + d_Xrot**2)

        cosdFi = d_Xrot/length_new_vect
        
        sin_directCos = 0
        names_cos.append(str(round(acos(direct_cos[0]),2)) + str(round(acos(direct_cos[1]),2)) + str(round(acos(direct_cos[2]),2)))


        # delta_cos = direct_cos[2]*cosdFi -







        # data_head_points_1.append([0, 0, 0])
b1 = np.array(data)
b2 = np.array(data_head_points)

# print(b[:])
# x, y = np.meshgrid(points, points)
# z = paraboloid(x, y)
# print(x[0,0:10])
# print(y[0,0:10])
# print(z[0,0:10])
#
# -------------------------------------------------------------------
# пока закоментил отрисовку
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(x=b1[:,0], y=b1[:,1], z=b1[:,2],
                                   mode='markers'),
                      go.Scatter3d(x=b2[:, 0], y=b2[:, 1], z=b2[:, 2],
                                   mode='markers+text', text=names_cos)
                      ])
fig.show()
# -------------------------------------------------------------------
