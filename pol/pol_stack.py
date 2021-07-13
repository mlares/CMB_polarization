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
 
import cmfg
from Parser import Parser
from sys import argv 

# CENTERS ----------------------------------------------

config = Parser('../set/POL02.ini')
X = cmfg.profile2d(config)
X.load_centers()
X.select_subsample_centers()
 

# CMB MAP --------------------------------------------

nside = 2048
filedata = '../../data/COM_CMB_IQU-smica_2048_R3.00_full.fits'

T = hp.read_map(filedata, field=0, h=False, dtype=float)
Q = hp.read_map(filedata, field=1, h=False, dtype=float)
U = hp.read_map(filedata, field=2, h=False, dtype=float)
 
# (from cmfg.profile2d.load_centers)
phi_healpix = X.centers['l']*np.pi/180.
theta_healpix = (90. - X.centers['b'])*np.pi/180.
X.centers['phi'] = phi_healpix
X.centers['theta'] = theta_healpix
X.centers['vec'] = hp.ang2vec(theta_healpix, phi_healpix).tolist()


rmax = config.p.r_stop # rad
rmax_deg = rmax*180/pi
N = 250
xr = np.linspace(-rmax_deg, rmax_deg, N)
yr = np.linspace(-rmax_deg, rmax_deg, N)

idxs = itertools.product(range(N), range(N))
idxs = list(idxs)
G = itertools.product(xr, yr)
G = list(G)

neigh = NearestNeighbors(n_neighbors=3, radius=0.001)

Zt = np.zeros((N,N))
Zq = np.zeros((N,N))
Zu = np.zeros((N,N))
Zqr = np.zeros((N,N))
Zur = np.zeros((N,N))


Nmax = X.centers.shape[0]
Nmax = 1  # PROFILING

Ncen = 0

for icenter in range(Nmax):
    print(icenter)
    Ncen += 1
    # compute rotation matrix
    phi = float(X.centers.iloc[icenter].phi)
    theta = float(X.centers.iloc[icenter].theta)
    pa = float(X.centers.iloc[icenter].pa) 
    vector = hp.ang2vec(theta, phi)
    rotate_pa = R.from_euler('zyz', [-phi, -theta, pa])

    listpixs = hp.query_disc(nside, vector, rmax,
                             inclusive=False, fact=4, nest=False) 

    dists, thetas, tt, qq, uu, qr = [], [], [], [], [], []

    for ipix in listpixs:

        v = hp.pix2vec(nside, ipix)
        w = rotate_pa.apply(v)

        dist = hp.rotator.angdist(w, [0, 0, 1])

        theta = atan2(w[1], w[0])
        if theta < 0:
            theta = theta + 2*pi

        dists.append(dist[0])
        thetas.append(theta)
        tt.append(T[ipix])
        qq.append(Q[ipix])
        uu.append(U[ipix])

    thetas = np.array(thetas)
    dists = np.array(dists)
    tt = np.array(tt)
    qq = np.array(qq)
    uu = np.array(uu)

    # calcular el ángulo psi
    psi2 = 2*thetas
    qr = -qq*np.cos(psi2) - uu*np.sin(psi2)
    ur =  qq*np.sin(psi2) - uu*np.cos(psi2)

    x = dists*np.cos(thetas)*180/pi
    y = dists*np.sin(thetas)*180/pi
    neigh.fit(np.column_stack([x, y]))

    for i, ix in zip(idxs, G):
        rr = np.linalg.norm(ix)
        if(rr<rmax_deg):
            dist, ind = neigh.kneighbors([ix], 3, return_distance=True)
            dd = np.exp(-dist*25)
            dsts = dd.sum()

            val = np.dot(dd, tt[ind][0])/dsts            
            Zt[i[0], i[1]] = Zt[i[0], i[1]] + val
            val = np.dot(dd, qq[ind][0])/dsts
            Zq[i[0], i[1]] = Zq[i[0], i[1]] + val
            val = np.dot(dd, uu[ind][0])/dsts
            Zu[i[0], i[1]] = Zu[i[0], i[1]] + val
            val = np.dot(dd, qr[ind][0])/dsts
            Zqr[i[0], i[1]] = Zqr[i[0], i[1]] + val
            val = np.dot(dd, ur[ind][0])/dsts
            Zur[i[0], i[1]] = Zur[i[0], i[1]] + val


Zt = Zt/Ncen
Zq = Zq/Ncen
Zu = Zu/Ncen
Zqr = Zqr/Ncen
Zur = Zur/Ncen

#with open('../../data/glxs_zetas.pk', 'wb') as f:
#   pickle.dump([Zt, Zq, Zu, Zqr, Zur], f)

P = np.sqrt(Zq**2 + Zu**2)
alpha = np.arctan(Zu, Zq) / 2


# # PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
# plt.close('all')
# fig = plt.figure(figsize=(7, 5))
# ax = fig.add_subplot()
# 
# #sc = ax.imshow(Zt, cmap='RdBu', vmin=-1.e-5, vmax=1.e-5)
# sc = ax.imshow(Zt, cmap='RdBu')
# 
# cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
# cb.set_label('averaged temperature')
# ax.set_xlabel('x [deg]')
# ax.set_ylabel('y [deg]')
# #ax.set_xlim(-rmax, rmax)
# #ax.set_ylim(-rmax, rmax)
# #ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
# #cb.formatter.set_powerlimits((-6, -6))
# #cb.update_ticks()
# plt.tight_layout()
# fig.savefig('Zt.png')
# 
