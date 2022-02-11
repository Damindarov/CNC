import numpy as np
from math import *
p0 = np.array([0, 0, 0]) # в этой точке строится нормаль
p1 = np.array([0, 2, 0])
p2 = np.array([2, 0, 0])

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
print(normal)
# длина нормального ветора
length_normal = sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
# направляющие косинусы
direct_cos = np.array([normal[0]/length_normal, normal[1]/length_normal, normal[2]/length_normal])
print(sqrt(direct_cos[0]**2+direct_cos[1]**2 + direct_cos[2]**2))

# координата центра вращающейся головы станка
x_head = np.array([-(normal[0] - p0[0]), -(normal[1] - p0[1]), -(normal[2] - p0[2])])

print(normal)



# print(np.linalg.det(m1), np.linalg.det(m2), np.linalg.det(m3))



# print(m1)               # Вывод: [1, 2, 3]
# print(type(p1))