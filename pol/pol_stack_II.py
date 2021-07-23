"""
PROJECT: POLARIZATION OF THE CMB BY FOREGROUNDS


"""

import numpy as np
import itertools
from math import atan2, pi, acos
from matplotlib import pyplot as plt
from matplotlib import ticker

import cmfg
from Parser import Parser
from sys import argv 

import healpy as hp
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors 

import pickle

import multiprocessing
from joblib import Parallel, delayed
import cmfg


if len(argv) > 1:
    config = Parser(argv[1])
else:
    config = Parser()   

# CENTERS ----------------------------------------------

# filename = pxs.check_file(sys.argv)
# config = ConfigParser()
# config.read(filename)

#config = Parser('../set/POL03.ini')
X = cmfg.profile2d(config)
X.load_centers()
X.select_subsample_centers()

def f(x, N, rmax, rmax_deg, xr, yr, idxs, G):
    
    center = x[1]
    neigh = NearestNeighbors(n_neighbors=3, radius=0.01)

    #print(center['ID'])

    Zt = np.zeros((N,N))
    Zq = np.zeros((N,N))
    Zu = np.zeros((N,N))
    Zqr = np.zeros((N,N))
    Zur = np.zeros((N,N))

    # compute rotation matrix
    phi = float(center.phi)
    theta = float(center.theta)
    pa = float(center.pa) 
    vector = hp.ang2vec(theta, phi)
    rotate_pa = R.from_euler('zyz', [-phi, -theta, pa])

    s = phi
    print(vector, rmax.value)

    listpixs = hp.query_disc(nside, vector, rmax.value,
                             inclusive=False, fact=4, nest=False) 

    dists, thetas, tt, qq, uu, qr = [], [], [], [], [], []

    print('entering listpixs')
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
    print('quiting listpixs')

    thetas = np.array(thetas)
    dists = np.array(dists)
    tt = np.array(tt)
    qq = np.array(qq)
    uu = np.array(uu)

    print('control 1')
    # calcular el ángulo psi
    psi2 = 2*thetas
    qr = -qq*np.cos(psi2) - uu*np.sin(psi2)
    ur =  qq*np.sin(psi2) - uu*np.cos(psi2)
    print('control 2')

    x = dists*np.cos(thetas)*180/pi
    y = dists*np.sin(thetas)*180/pi
    neigh.fit(np.column_stack([x, y]))

    print('neigh fit')

    for i, ix in zip(idxs, G):
        rr = np.linalg.norm(ix)
        if(rr<rmax_deg.value):
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
 
    res = [Zt, Zq, Zu, Zqr, Zur]
    return res

# CMB MAP --------------------------------------------

nside = 2048
filedata = '../../data/COM_CMB_IQU-smica_2048_R3.00_full.fits'

T = hp.read_map(filedata, field=0, h=False, dtype=float)
Q = hp.read_map(filedata, field=1, h=False, dtype=float)
U = hp.read_map(filedata, field=2, h=False, dtype=float)
 
N = 250
rmax = config.p.r_stop # rad
rmax_deg = rmax*180/pi
xr = np.linspace(-rmax_deg, rmax_deg, N).value
yr = np.linspace(-rmax_deg, rmax_deg, N).value

idxs = itertools.product(range(N), range(N))
idxs = list(idxs)
G = itertools.product(xr, yr)
G = list(G)

Ncen = X.centers.shape[0]

print(Ncen)

run = (delayed(f)(x, N, rmax, rmax_deg, xr, yr, idxs, G) for x in X.centers.head(Ncen).iterrows()) 
results = Parallel(n_jobs=40)(run)

fout = f'{config.filenames.dir_output}{config.filenames.experiment_id}/data_{config.filenames.experiment_id}.pk'
print('escribiendo datos...')
print(fout)
 
with open(fout, 'wb') as arch:
   pickle.dump(results, arch)
 
 

#    #with open('../../data/glxs_zetas.pk', 'wb') as f:
#    #   pickle.dump([Zt, Zq, Zu, Zqr, Zur], f)
#    
#    #P = np.sqrt(Zq**2 + Zu**2)
#    #alpha = np.arctan(Zu, Zq) / 2
