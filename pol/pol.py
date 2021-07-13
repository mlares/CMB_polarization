"""
PROJECT: POLARIZATION OF THE CMB BY FOREGROUNDS


"""

import numpy as np
import pandas as pd
import healpy as hp
import itertools
from math import atan2, pi, acos
from matplotlib import pyplot as plt
from matplotlib import ticker

from PixelSky import SkyMap

from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

from astropy import units as u
from astropy.coordinates import SkyCoord


# CENTERS ----------------------------------------------

# Galaxias

# filename = '../data/2mrs_1175_done.dat'
# 
# with open(filename, 'r') as f:
#     centers = pd.read_csv(f, skiprows=9, delim_whitespace=True)
# 
# # drop out unsed columns
# 
# cols_out = ['RAdeg', 'DECdeg', 'k_c', 'h_c', 'j_c', 'k_tc', 'h_tc',
#             'j_tc', 'e_k', 'e_h', 'e_j', 'e_kt', 'e_ht', 'e_jt', 'e_bv']
# centers = centers.drop(columns=cols_out)


# Clusters

filename = '../data/GMBCG_SDSS_DR7_PUB_ASCII.txt'

with open(filename, 'r') as f:
    centers = pd.read_csv(f, skiprows=5)

cols_out = ["objid", "photoz", "photoz_err", "gmr_err","rmi","rmi_err",
            "dered_u","dered_g","dered_r","dered_i","dered_z","u_Err",
            "g_Err","r_Err","i_Err","z_Err","weightOK","name"]
centers = centers.drop(columns=cols_out)

l = []
b = []

for index, row in centers.iterrows():
    c_icrs = SkyCoord(ra=row.ra*u.degree, dec=row.dec*u.degree, frame='icrs')
    l.append(c_icrs.galactic.l.value)
    b.append(c_icrs.galactic.b.value)

centers['l'] = np.array(l)
centers['b'] = np.array(b)




# CMB MAP --------------------------------------------

nside = 2048
filedata = '../data/COM_CMB_IQU-smica_2048_R3.00_full.fits'

T = hp.read_map(filedata, field=0, h=False, dtype=float)
Q = hp.read_map(filedata, field=1, h=False, dtype=float)
U = hp.read_map(filedata, field=2, h=False, dtype=float)
 

# (from cmfg.profile2d.load_centers)
phi_healpix = centers['l']*np.pi/180.
theta_healpix = (90. - centers['b'])*np.pi/180.
centers['phi'] = phi_healpix
centers['theta'] = theta_healpix
centers['vec'] = hp.ang2vec(theta_healpix, phi_healpix).tolist()
 

# por ahora usar solo un centro


# compute rotation matrix
phi = float(centers.phi[1])
theta = float(centers.theta[1])
pa = 0.  # por ahora no uso el ángulo de posición
vector = hp.ang2vec(centers.theta[1], centers.phi[1])

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


neigh = NearestNeighbors(n_neighbors=5, radius=0.1)
X = np.column_stack([x, y])
neigh.fit(X)

rmax_deg = rmax*180/pi
xr = np.linspace(-rmax_deg, rmax_deg, 60)
yr = np.linspace(-rmax_deg, rmax_deg, 60)
G = itertools.product(xr, yr)

xp = []
yp = []
zp3 = []

for ix in G:
    rr = np.sqrt(ix[0]**2 + ix[1]**2)
    if(rr<rmax_deg):
        xp.append(ix[0])
        yp.append(ix[1])

        dist, ind = neigh.kneighbors([[ix[0],ix[1]]], 3, return_distance=True)

        dd = np.exp(-dist*25)
        dsts = dd.sum()
        zz3 = np.dot(dd, qq[ind][0])/dsts

        zp3.append(zz3)


# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()
sc = ax.scatter(x, y, s=40, c=qq,
                cmap='viridis', vmin=-1.e-4, vmax=1.e-4)
cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label('Q Stokes parameter')
ax.set_aspect(0.8)
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
cb.formatter.set_powerlimits((-4, -4))
cb.update_ticks()
plt.tight_layout()
fig.savefig('qq_healpix.png')





# «««
plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()
sc = ax.scatter(xp, yp, s=20, c=zp3,
                cmap='viridis', vmin=-1.e-4, vmax=1.e-4)
cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label('Q Stokes parameter')
ax.set_aspect(0.8)
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.6, 0.6)
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
cb.formatter.set_powerlimits((-4, -4))
cb.update_ticks()
plt.tight_layout()
fig.savefig('qq_cartesian.png')
                               

# ««««««««««««««««««««««««««««««««««««««««««««     
