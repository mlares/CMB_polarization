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

filename = '../data/2mrs_1175_done.dat'

with open(filename, 'r') as f:
    centers = pd.read_csv(f, skiprows=9, delim_whitespace=True)

# drop out unsed columns

cols_out = ['RAdeg', 'DECdeg', 'k_c', 'h_c', 'j_c', 'k_tc', 'h_tc',
            'j_tc', 'e_k', 'e_h', 'e_j', 'e_kt', 'e_ht', 'e_jt', 'e_bv']
centers = centers.drop(columns=cols_out)


# FILTER  ----------------------------------------------

Sa_lbl = ['1', '2']
Sb_lbl = ['3', '4']
Sc_lbl = ['5', '6']
Sd_lbl = ['7', '8']
E_lbl = ['-7', '-6', '-5']
Sno_lbl = ['10', '11', '12', '15', '16', '19', '20', '98']
Gtypes = []

gtypes = ['Sabc']

for s in gtypes:
    opt = s.lower()
    if 'spiral' in opt or 'late' in opt:
        opt = 'abcd'
    noellipt = ('early' not in opt) and ('elliptical' not in opt)
    if ('a' in opt or 'sa' in opt) and noellipt:
        Gtypes = Gtypes + Sa_lbl
    if ('b' in opt or 'sb' in opt) and noellipt:
        Gtypes = Gtypes + Sb_lbl
    if ('c' in opt or 'sc' in opt) and noellipt:
        Gtypes = Gtypes + Sc_lbl
    if ('d' in opt or 'sd' in opt) and noellipt:
        Gtypes = Gtypes + Sd_lbl

for s in gtypes:
    opt = s.lower()
    if 'ellipt' in opt or 'early' in opt:
        opt = 'e'
    if 'e' in opt:
        Gtypes = sum([Gtypes, E_lbl], [])

redshift_min = 0.008
redshift_max = 0.01

ellipt_min = 0.
ellipt_max = 0.5
 
glx_angsize_min = 0.0005
glx_angsize_max = 10.
 

filt1 = []
for t in centers['type']:
    f1 = t[0] in Gtypes and not (t[:2] in Sno_lbl)
    f2 = t[0:2] in Gtypes and not (t[:2] in Sno_lbl)
    f = f1 or f2
    filt1.append(f)

# filter on: redshift ----------------------------
zmin = redshift_min
zmax = redshift_max
filt2 = []
for cz in centers['v']:
    z = cz / 300000
    f = z > zmin and z < zmax
    filt2.append(f)

# filter on: elliptical isophotal orientation -----
boamin = ellipt_min
boamax = ellipt_max
filt3 = []
for boa in centers['b/a']:
    f = boa > boamin and boa < boamax
    filt3.append(f)


glxsize = np.array(centers['r_ext'])
glxsize = 10**glxsize      # !!!!!!!!!!!!!! VERIFICAR
glxsize = glxsize*u.arcsec
glxsize = glxsize.to(u.rad)
centers['glx_size_rad'] = glxsize 


# filter on: galaxy size ----------------
filt4 = []
smin = glx_angsize_min
smax = glx_angsize_max
for glxsize in centers['glx_size_rad']:
    f = glxsize > smin and glxsize < smax
    filt4.append(f)


# filter all ----------------------------
filt = np.logical_and.reduce((filt1, filt2, filt3, filt4))
centers = centers[filt]

centers=centers[filt].reset_index(drop=True)

# END FILTER  ------------------------------------------









# Clusters

# filename = '../data/GMBCG_SDSS_DR7_PUB_ASCII.txt'
# with open(filename, 'r') as f:
#     centers = pd.read_csv(f, skiprows=5)
# cols_out = ["objid", "photoz", "photoz_err", "gmr_err","rmi","rmi_err",
#             "dered_u","dered_g","dered_r","dered_i","dered_z","u_Err",
#             "g_Err","r_Err","i_Err","z_Err","weightOK","name"]
# centers = centers.drop(columns=cols_out)
# l = []
# b = []
# for index, row in centers.iterrows():
#     c_icrs = SkyCoord(ra=row.ra*u.degree, dec=row.dec*u.degree, frame='icrs')
#     l.append(c_icrs.galactic.l.value)
#     b.append(c_icrs.galactic.b.value)
# centers['l'] = np.array(l)
# centers['b'] = np.array(b)


# Clusters (processed)
# 
# filename = '../data/GMBCG_SDSS_DR7_PUB_ASCII.dat'
# with open(filename, 'r') as f:
#     centers = pd.read_csv(f)
 


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


pa = 0.  # por ahora no uso el ángulo de posición
rmax = 0.01  # rad
rmax_deg = rmax*180/pi
N = 60
xr = np.linspace(-rmax_deg, rmax_deg, N)
yr = np.linspace(-rmax_deg, rmax_deg, N)

idxs = itertools.product(range(N), range(N))
idxs = list(idxs)
G = itertools.product(xr, yr)
G = list(G)

neigh = NearestNeighbors(n_neighbors=5, radius=0.1)

Zt = np.zeros((N,N))
Zq = np.zeros((N,N))
Zu = np.zeros((N,N))
Zqr = np.zeros((N,N))
Zur = np.zeros((N,N))


Nmax = centers.shape[0]

Ncen = 0
for icenter in range(Nmax):
    print(icenter)
    Ncen += 1
    # compute rotation matrix
    phi = float(centers.iloc[icenter].phi)
    theta = float(centers.iloc[icenter].theta)
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

    M = tt

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

P = np.sqrt(Zq**2 + Zu**2)
alpha = np.arctan(Zu, Zq) / 2


# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()
sc = ax.scatter(x, y, s=40, c=qq,
                cmap='RdBu', vmin=-1.e-4, vmax=1.e-4)
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

xp, yp = [], []
for g in G:
    xp.append(g[0])
    yp.append(g[1])
zz = Zq.copy()
zz=zz.reshape(N*N)

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

sc = ax.scatter(xp, yp, s=20, c=zz, cmap='RdBu', vmin=-1.e-4, vmax=1.e-4)
                #cmap='RdBu', vmin=-1.e-4, vmax=1.e-4)


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
