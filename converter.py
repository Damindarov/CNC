import numpy as np
from math import *
from matrix import *
def convert(x,y,z,ang0,tr0,ang1,tr1):
    arr = Rz(ang0)@Tz(tr0)@Ry(ang1)@Tz(tr1)@Tx(x)@Ty(y)@Tz(z)
    print(arr[:,3:4])
    return arr[:,3:4][0][0], arr[:,3:4][1][0], arr[:,3:4][2][0]
# p0 = np.array([0, 0, 0]) # в этой точке строится нормаль
# p1 = np.array([0, 2, 0])
# p2 = np.array([2, 0, 0])
#
a,b,c = convert(0,0,0,0,10,pi/2,10)
print(a,b,c)


# data = []
# x = np.linspace(0, 10, 5)
# y = np.linspace(0, 10, 5)
# for i in range(len(x)-1):
#     for j in range(len(y)-1):
#         data.append([x[i], y[j], -(0.1*x[i]**2 + 0.1*y[j]**2)])
#         r = sqrt(x[i]**2 + y[j]**2, (-(0.1*x[i]**2 + 0.1*y[j]**2)**2))
#         fi = atan2(y[j],x[i])
#         # if y > 0:
#         #
#         # w = acos(-(0.1*x[i]**2 + 0.1*y[j]**2)/)
