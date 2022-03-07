import numpy as np
from math import *
from converter import *

data, data_head_points = [], []
point_1, point_2, point_3 = [], [], []
names_cos = []
ang1, ang2 = [], []

length_tools = 5
offset_y = 0.09
offset_x_liner = 0.516
offset_x_rot = 0.078

def IK(x, y, z, ang_C1,l1, ang_C2, l2):
    print('len ',np.shape(x), len(x))
    for i in range(0,len(x)):
        for j in range(0, len(y)):
            i_pr,j_pr = 0,0
            signx,signy = 1,1
            if i == len(x)-1:
                i_pr = i - 2
                signy = -1
            else:
                i_pr = i + 1
                signy = 1

            if j == len(y)-1:
                j_pr = j - 2
                signx = -1
            else:
                j_pr = j + 1
                signx = 1
            a, b, c = convert(x[i][j], y[i][j], z[i][j], ang_C1, l1, ang_C2, l2)
            data.append([a, b, c])

            a, b, c = convert(x[i][j], y[i][j], z[i][j], ang_C1, l1, ang_C2, l2)
            p0 = np.array([a, b, c])
            a, b, c = convert(x[i_pr][j], y[i_pr][j], z[i_pr][j], ang_C1, l1, ang_C2, l2)
            p1 = np.array([a, b, c])
            a, b, c = convert(x[i][j_pr], y[i][j_pr], z[i][j_pr], ang_C1, l1, ang_C2, l2)
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
            direct_cos = np.array([signx*signy*normal[0] / length_normal, signx*signy*normal[1] / length_normal, signx*signy*normal[2] / length_normal])

            # print(p0[0] - length_tools*direct_cos[0], p0[1] - length_tools*direct_cos[1], p0[2] - length_tools*direct_cos[2])
            # координата центра вращающейся головы станка
            # x_head = np.array([p0[0] + length_tools * direct_cos[0], p0[1] + length_tools * direct_cos[1],
            #                    p0[2] + length_tools * direct_cos[2]])
            data_head_points.append([p0[0] + length_tools * direct_cos[0], p0[1] + length_tools * direct_cos[1],
                                     p0[2] + length_tools * direct_cos[2]])
            names_cos.append(str(round(acos(direct_cos[0]), 2)) + str(round(acos(direct_cos[1]), 2)) + str(
                round(acos(direct_cos[2]), 2)))

            katet1 = sqrt(offset_x_rot ** 2 - (offset_x_rot * direct_cos[2]) ** 2)
            katet2 = sqrt(offset_x_rot ** 2 - katet1 ** 2)
            z1 = p0[2] + length_tools * direct_cos[2] + katet1
            x1 = p0[0] + length_tools * direct_cos[0] - katet1 * direct_cos[
                0]  # + length_tools * direct_cos[0] - katet * direct_cos[0]
            y1 = p0[1] + length_tools * direct_cos[1] - katet1 * direct_cos[
                1]  # + length_tools * direct_cos[1] - katet * direct_cos[1]

            # print('Katet1 ', katet1, 'Katet2 ', katet2)
            point_1.append([x1, y1, z1])
            x11 = x1 - offset_x_liner * direct_cos[0]
            y11 = y1 - offset_x_liner * direct_cos[1]
            z11 = z1

            point_2.append([x11, y11, z11])

            x111 = x11 - offset_y * direct_cos[1]
            y111 = y11 + offset_y * direct_cos[0]
            z111 = z11

            point_3.append([x111, y111, z111])
            ang2.append(acos(direct_cos[1]))
            ang1.append(acos(direct_cos[0]))
            teta2 = acos(direct_cos[1])
            teta1 = acos(direct_cos[0])

    b1 = np.array(data)
    b2 = np.array(data_head_points)
    b3 = np.array(point_1)
    b4 = np.array(point_2)
    b5 = np.array(point_3)
    ang1_ = np.array(ang1)
    ang2_ = np.array(ang2)
    return b5[:, 0], b5[:, 1], b1[:,2], ang1_, ang2_
