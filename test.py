import numpy as np
from math import *
def angle_between(v1, v2):
    dot_pr = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)

    return np.rad2deg(np.arccos(dot_pr / norms))
def Rx(q):
  T = np.array([[1,         0,          0, 0],
                [0, np.cos(q), -np.sin(q), 0],
                [0, np.sin(q),  np.cos(q), 0],
                [0,         0,          0, 1]], dtype=float)
  return T

def Ry(q):
  T = np.array([[ np.cos(q), 0, np.sin(q), 0],
                [         0, 1,         0, 0],
                [-np.sin(q), 0, np.cos(q), 0],
                [         0, 0,         0, 1]], dtype=float)
  return T

def Rz(q):
  T = np.array([[np.cos(q), -np.sin(q), 0, 0],
                [np.sin(q),  np.cos(q), 0, 0],
                [        0,          0, 1, 0],
                [        0,          0, 0, 1]], dtype=float)
  return T

def Tx(x):
  T = np.array([[1, 0, 0, x],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=float)
  return T

def Ty(y):
  T = np.array([[1, 0, 0, 0],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=float)
  return T

def Tz(z):
  T = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, z],
                [0, 0, 0, 1]], dtype=float)
  return T

def convert(x,y,z,ang0,tr0,ang1,tr1):
    arr = Rz(ang0)@Tz(tr0)@Ry(ang1)@Tz(tr1)@Tx(x)@Ty(y)@Tz(z)
    # print(arr[:,3:4])
    return arr[:,3:4][0][0], arr[:,3:4][1][0], arr[:,3:4][2][0]

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

            # A, B, C, D ????-?? ??????????????????
            equation = np.array([det_m1, -det_m2, det_m3, -p0[0] * det_m1 + p0[1] * det_m2 - p0[2] * det_m3])

            # ???????????? ?????????????? ?? ??????????????????
            normal = np.array([equation[0], equation[1], equation[2]])

            # print(normal)
            # ?????????? ?????????????????????? ????????????
            length_normal = sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)

            # ???????????????????????? ????????????????
            vect1 = [signy * signx * normal[0] / length_normal, signy * signx * normal[1] / length_normal,
                     signy * signx * normal[2] / length_normal]
            direct_cos = np.array(vect1)

            # print(round(direct_cos[0],2), round(round(direct_cos[1]),2), round(direct_cos[2],2),round(p0[0] + length_tools * direct_cos[0],2),
            #       round(p0[1] + length_tools * direct_cos[1],2), round(p0[2] + length_tools * direct_cos[2],2),
            #       round(p0[0],2), round(p0[1],2), round(p0[2],2))
            # print(p0[0] - length_tools*direct_cos[0], p0[1] - length_tools*direct_cos[1], p0[2] - length_tools*direct_cos[2])
            # ???????????????????? ???????????? ?????????????????????? ???????????? ????????????
            # x_head = np.array([p0[0] + length_tools * direct_cos[0], p0[1] + length_tools * direct_cos[1],
            #                    p0[2] + length_tools * direct_cos[2]])
            vect2 = [p0[0] + length_tools * direct_cos[0], p0[1] + length_tools * direct_cos[1],
                     p0[2] + length_tools * direct_cos[2]]
            data_head_points.append(vect2)
            v1 = [p0[0] - vect2[0], p0[1] - vect2[1], p0[2] - vect2[2]]
            v2 = [0, 0, -length_tools]

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
            ang2.append(angle_between(np.array(v1), np.array(v2)))
            deltas = [length_tools * direct_cos[0], length_tools * direct_cos[1], length_tools * direct_cos[2]]
            ang1.append(np.rad2deg(atan2(deltas[1], deltas[0])))

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



# ?????????????????? ?????????????????? ????
Rap = 999999999999
KoniKonst = 0.0
A2 = 0.0
A3 = 0.0
A4 = 0.0

# ?????? ????????????
Scritt_X_Y = 3

# ???????????? ????????????
Drchm = 30.0

# ???????????? ?????????????? ?????????? ?? ?????????????????? ?????? ?????????????? ?????????????? ??????????????
aa = np.float32(np.arange(-Drchm / 2, Drchm / 2 + Scritt_X_Y, Scritt_X_Y))
a = aa[np.newaxis, :]

Punkten1 = [[]]
for i in range(len(a)):
    for j in range(len(aa)):
        res = Punkten1.append(((a[i]) ** 2 + (aa[j]) ** 2) ** 0.5)

Punkten2 = [[]]
for i in range(len(a)):
    for j in range(len(aa)):
        res = Punkten2.append((a[i]))

# ???????????????????? ?????????? ???? ???? Y
Y_Linse = np.float32(Punkten2[1:])

# ???????????????????? ?????????? ???? ?????????????????????????? ???????????? ????. ?????? ?????????? ?????? ???????????????????? ?????????????????? ???? ?? ???????????????????? ???????????????????????? ???????????? ?????? ??2
Koordin_X = -1 * Y_Linse.transpose()

# ?????????????? ?????????????? ?? ?????????????? ???????????? ???? ?????????????? ????
Punkten = np.float32(Punkten1[1:])
Pfeilh??he = Punkten * Punkten / (Rap * (1 + (1 - (
            1 + KoniKonst) * Punkten * Punkten / Rap / Rap) ** 0.5)) + A2 * Punkten ** 2 + A3 * Punkten ** 3 + A4 * Punkten ** 4
#x,y,z ???????? ??1, ????????????????1, ???????? ??2, ????????????????2. ???????? ?? ????????????????
# print(np.shape(Koordin_X))

# print(Y_Linse)
# print(-Y_Linse)
x_cnc, y_cnc, z_cnc, ang1_cnc, ang2_cnc = IK(Koordin_X, -Y_Linse, Pfeilh??he,pi/4,400,pi/4,10)
tex = []
for i in range(len(ang1_cnc)):
    tex.append(str(round(ang1_cnc[i],2))+' '+str(round(ang2_cnc[i],2)))
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(x=x_cnc, y=y_cnc, z=z_cnc,
                                   mode='markers+text', text = tex),
                      go.Scatter3d(x=Koordin_X[:, 0], y=Y_Linse[:, 1], z=Pfeilh??he[:, 2],
                                   mode='markers'),
                      # go.Scatter3d(x=b3[:, 0], y=b3[:, 1], z=b3[:, 2],
                      #              mode='markers'),
                      # go.Scatter3d(x=b4[:, 0], y=b4[:, 1], z=b4[:, 2],
                      #              mode='markers'),
                      # go.Scatter3d(x=b5[:, 0], y=b5[:, 1], z=b5[:, 2],
                      #              mode='markers'),
                      # go.Scatter3d(x=b5[:, 0], y=b5[:, 1], z=b5[:, 2],
                      #              mode='markers')
                      ])
fig.show()