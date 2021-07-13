"""
PROJECT: POLARIZATION OF THE CMB BY FOREGROUNDS


"""

import numpy as np
import pandas as pd
from PixelSky import SkyMap
import healpy as hp
from scipy.spatial.transform import Rotation as R
from math import atan2, pi, acos
from scipy.interpolate import interp2d, SmoothBivariateSpline
import itertools
from matplotlib import pyplot as plt
from matplotlib import ticker
from sklearn.neighbors import BallTree


# CENTERS ----------------------------------------------

filename = '../data/2mrs_1175_done.dat'

with open(filename, 'r') as f:
    glx = pd.read_csv(f, skiprows=9, delim_whitespace=True)

# drop out unsed columns

cols_out = ['RAdeg', 'DECdeg', 'k_c', 'h_c', 'j_c', 'k_tc', 'h_tc',
            'j_tc', 'e_k', 'e_h', 'e_j', 'e_kt', 'e_ht', 'e_jt', 'e_bv']
glx = glx.drop(columns=cols_out)

# CMB MAP --------------------------------------------

nside = 2048
filedata = '../data/COM_CMB_IQU-smica_2048_R3.00_full.fits'

Q = hp.read_map(filedata, field=1, h=False, dtype=float)
U = hp.read_map(filedata, field=2, h=False, dtype=float)
 

# (from cmfg.profile2d.load_centers)
phi_healpix = glx['l']*np.pi/180.
theta_healpix = (90. - glx['b'])*np.pi/180.
glx['phi'] = phi_healpix
glx['theta'] = theta_healpix
glx['vec'] = hp.ang2vec(theta_healpix, phi_healpix).tolist()
 

# por ahora usar solo un centro


# compute rotation matrix
phi = float(glx.phi[1])
theta = float(glx.theta[1])
pa = 0.  # por ahora no uso el ángulo de posición
vector = hp.ang2vec(glx.theta[1], glx.phi[1])

rotate_pa = R.from_euler('zyz', [-phi, -theta, pa])

rmax = 0.01
listpixs = hp.query_disc(nside, vector, rmax,
                         inclusive=False, fact=4, nest=False) 


dists = []
thetas = []
qq = []
uu = []

for ipix in listpixs:

    v = hp.pix2vec(nside, ipix)
    w = rotate_pa.apply(v)

    dist = hp.rotator.angdist(w, [0, 0, 1])

    theta = atan2(w[1], w[0])
    if theta < 0:
        theta = theta + 2*pi

    dists.append(dist[0])
    thetas.append(theta)
    qq.append(Q[ipix])
    uu.append(U[ipix])


thetas = np.array(thetas)
dists = np.array(dists)
qq = np.array(qq)
uu = np.array(uu)



x = dists*np.cos(thetas)*180/pi
y = dists*np.sin(thetas)*180/pi

rmax_deg = rmax*180/pi
xr = np.linspace(-rmax_deg, rmax_deg, 40)
yr = np.linspace(-rmax_deg, rmax_deg, 40)


# interpolación {

X=np.column_stack((x, y))
tree = BallTree(X, leaf_size=6)

f1 = SmoothBivariateSpline(x, y, qq)
f2 = interp2d(x, y, qq)

# }

xp = []
yp = []
zp1 = []
zp2 = []
zp3 = []

for ix in X:
    rr = np.sqrt(ix[0]**2 + ix[1]**2)
    if(rr<rmax_deg):
        xp.append(ix[0])
        yp.append(ix[1])

        zz1 = f1(ix[0], ix[1])
        zz2 = f2(ix[0], ix[1])

        n_dist, n_ind = tree.query(ix.reshape(1,-1), k=3)
        dd = np.exp(-n_dist*25)
        dsts = dd.sum()
        zz3 = np.dot(dd, qq[n_ind][0])/dsts

        zp1.append(zz1)
        zp2.append(zz2)
        zp3.append(zz3)


# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot()
sc = ax.scatter(x, y, s=30, c=qq)
cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=40)
cb.set_label('Q Stokes parameter')
ax.set_aspect(0.8)
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
cb.formatter.set_powerlimits((-4, -4))
cb.update_ticks()
plt.tight_layout()
fig.savefig('qq_healpix.png')

# «««
plt.close('all')
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot()

sc = ax.scatter(xp, yp, s=30, c=zp1)

cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=40)
cb.set_label('Q Stokes parameter')
ax.set_aspect(0.8)
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
cb.formatter.set_powerlimits((-4, -4))
cb.update_ticks()
plt.tight_layout()
fig.savefig('qq_cartesian.png') 
        
        

# «««
plt.close('all')
fig = plt.figure(figsize=(9, 9))
ax1 = fig.add_subplot(2,2,1)
sc = ax1.scatter(x, y, s=30, c=qq)

ax1.set_aspect(0.8)
ax1.set_xlabel('x [deg]')
ax1.set_ylabel('y [deg]')
ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

ax2 = fig.add_subplot(2,2,2)
ax2.set_aspect(0.8)
ax2.scatter(xp, yp, s=30, c=zp1)
 
ax3 = fig.add_subplot(2,2,3)
ax3.set_aspect(0.8)
ax3.scatter(xp, yp, s=30, c=zp2) 

ax4 = fig.add_subplot(2,2,4)
ax4.set_aspect(0.8)
ax4.scatter(xp, yp, s=30, c=zp3) 

#cb = plt.colorbar(sc, shrink=0.8, aspect=40)
cb = fig.colorbar(sc, ax=[ax2, ax4], location='right', shrink=0.8, aspect=50)
cb.set_label('Q Stokes parameter')
cb.formatter.set_powerlimits((-4, -4))
cb.update_ticks()

plt.tight_layout()
fig.savefig('qq_2.png') 
                      


# ««««««««««««««««««««««««««««««««««««««««««««     
