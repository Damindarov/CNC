import numpy as np
from math import *
from matrix import *
def convert(x,y,z,ang0,tr0,ang1,tr1,ang2,tr2):
    arr = Rz(ang0)@Tz(tr0)@Ry(ang1)@Tz(tr1)@Ry(ang2)@Tz(tr2)
    print(arr)
    return arr[:,3:4][0][0] + x, arr[:,3:4][1][0] + y, arr[:,3:4][2][0] + z
# p0 = np.array([0, 0, 0]) # в этой точке строится нормаль
# p1 = np.array([0, 2, 0])
# p2 = np.array([2, 0, 0])
#
a,b,c = convert(0,0,0,0,10,pi/2,10,0,4)
print(a,b,c)
