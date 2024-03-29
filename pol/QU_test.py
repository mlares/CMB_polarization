import numpy as np
from matplotlib import pyplot as plt
from math import atan2

d = np.genfromtxt('../dat/experiment_8/Shear_all_quantities.dat',
                  skip_header=1, unpack=True)

"""
0) alfa_x   ---> componente x del ángulo theta 
1) delta_z ---> componente z del ángulo theta 

2) gamma ---> el módulo sqrt(Q^2 + U^2) 
3) var_phi ---> el ángulo de la rayita respecto del eje x 

4) E_shear ---> Q_r 
5) B_shear ---> U_r 

6) gamma_1 ---> El equivalente a Q 
7) gamma_2 ---> El equivalente a U 
"""

N = d.shape[1]

x = d[0, :]
y = d[1, :]

angles = []
for i in range(N):
    a = atan2(y[i], x[i])
    angles.append(a)


Q = d[6,:]
U = d[6,:]

psi2 = 2*np.array(angles)
qr = -Q*np.cos(psi2) - U*np.sin(psi2)
ur =  Q*np.sin(psi2) - U*np.cos(psi2)

######## NO FUNCA  (qr == d[4,:])

opts = dict(color='k', headlength=0, pivot='middle',
            scale=3, linewidth=1, units='xy', width=1, 
            headwidth=1, headaxislength=0)

plot(x, y, marker='o', markersize=0.5, linestyle='None', color='red')

rayita_x = d[2,:]*np.cos(d[3,:])
rayita_y = d[2,:]*np.sin(d[3,:])
rayita_x = 10000*np.cos(d[3,:])
rayita_y = 10000*np.sin(d[3,:])

#mod = np.sqrt(rayita_x**2 + rayita_y**2)
#rayita_x = rayita_x/np.sqrt(mod)
#rayita_y = rayita_y/np.sqrt(mod)

filt = mod > 0.0025

rayita_x = rayita_x[filt]
rayita_y = rayita_y[filt]
pos_x = d[0,:][filt]
pos_y = d[1,:][filt]

quiver(pos_x, pos_y, rayita_x, rayita_y, **opts)



