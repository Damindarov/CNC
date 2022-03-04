import numpy as np
from math import *
from converter import *

p0 = np.array([0, 0, 0]) # в этой точке строится нормаль
p1 = np.array([0, 2, 0])
p2 = np.array([2, 0, 0])

length_tools = 5
offset_y = 0.09
offset_x_liner = 0.516
offset_x_rot = 0.078

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
point_1 = []
point_2 = []
point_3 = []


names_cos = []
# data_converted = []
# data_converted_head_points = []

for i in range(len(x)-1):
    for j in range(len(y)-1):
        a, b, c = convert(x[i], y[j], -(0.1 * x[i] ** 2 + 0.1 * y[j] ** 2), 10, 10, pi/4, 10)
        data.append([a,b,c])
        a,b,c = convert(x[i], y[j], -(0.1*x[i]**2 + 0.1*y[j]**2),10, 10,pi/4,10)
        p0 = np.array([a, b, c])
        a, b, c = convert(x[i+1], y[j], -(0.1 * x[i+1] ** 2 + 0.1 * y[j] ** 2),10, 10,pi/4,10)
        p1 = np.array([a, b, c])
        a, b, c = convert(x[i], y[j+1], -(0.1 * x[i] ** 2 + 0.1 * y[j+1] ** 2),10, 10,pi/4,10)
        p2 = np.array([a, b, c])

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
        names_cos.append(str(round(acos(direct_cos[0]),2)) + str(round(acos(direct_cos[1]),2)) + str(round(acos(direct_cos[2]),2)))

        katet1 = sqrt(offset_x_rot**2 - (offset_x_rot*direct_cos[2])**2)
        katet2 = sqrt(offset_x_rot**2 - katet1**2)
        z1 = p0[2] + length_tools * direct_cos[2] + katet1
        x1 = p0[0] + length_tools * direct_cos[0] - katet1 * direct_cos[0] #+ length_tools * direct_cos[0] - katet * direct_cos[0]
        y1 = p0[1] + length_tools * direct_cos[1] - katet1 * direct_cos[1] #+ length_tools * direct_cos[1] - katet * direct_cos[1]

        print('Katet1 ',katet1, 'Katet2 ',katet2)
        point_1.append([x1,y1,z1])
        x11 = x1 - offset_x_liner * direct_cos[0]
        y11 = y1 - offset_x_liner * direct_cos[1]
        z11 = z1

        point_2.append([x11, y11, z11])

        x111 = x11 - offset_y * direct_cos[1]
        y111 = y11 + offset_y * direct_cos[0]
        z111 = z11

        point_3.append([x111, y111, z111])
        teta2 = acos(direct_cos[1])
        teta1 = acos(direct_cos[0])
        # data_head_points_1.append([0, 0, 0])
b1 = np.array(data)
b2 = np.array(data_head_points)
b3 = np.array(point_1)
b4 = np.array(point_2)
b5 = np.array(point_3)
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
                                   mode='markers'),
                      go.Scatter3d(x=b3[:, 0], y=b3[:, 1], z=b3[:, 2],
                                   mode='markers'),
                      go.Scatter3d(x=b4[:, 0], y=b4[:, 1], z=b4[:, 2],
                                   mode='markers'),
                      go.Scatter3d(x=b5[:, 0], y=b5[:, 1], z=b5[:, 2],
                                   mode='markers'),
                      # go.Scatter3d(x=b5[:, 0], y=b5[:, 1], z=b5[:, 2],
                      #              mode='markers')
                      ])
fig.show()
# -------------------------------------------------------------------
