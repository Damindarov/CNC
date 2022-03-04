
import numpy as np


# Параметры уравнения АП
Rap=52.0
KoniKonst=0.0
A2=0.0
A3=0.0
A4=0.0

# Шаг растра
Scritt_X_Y=0.5

# Размер детали
Drchm=30.0


# Массив узловых точек с радиусами для расчета стрелок прогиба
aa = np.float32(np.arange(-Drchm/2, Drchm/2+Scritt_X_Y, Scritt_X_Y))
a= aa[np.newaxis, :]

Punkten1 =[[]]
for i in range(len(a)):
   for j in range (len(aa)):
      res = Punkten1.append(((a[i])**2+(aa[j])**2)**0.5)
      
Punkten2 =[[]]
for i in range(len(a)):
   for j in range (len(aa)):
      res = Punkten2.append((a[i]))

# Координаты точек АП по Y
Y_Linse = np.float32(Punkten2[1:])

# Координаты точек АП отностительно центра АП. Это нужно для прересчета координат АП в координаты относительно нижней оси С2
Koordin_X = -1 * Y_Linse.transpose()

# Стрелки прогиба в узловых точках по формуле АП
Punkten = np.float32(Punkten1[1:])
Pfeilhöhe = Punkten*Punkten/(Rap*(1+(1-(1+KoniKonst)*Punkten*Punkten/Rap/Rap)**0.5)) + A2*Punkten**2 + A3*Punkten**3+A4*Punkten**4
print(Y_Linse)
print(Koordin_X)

print(Pfeilhöhe)
