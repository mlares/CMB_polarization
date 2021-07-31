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
import tqdm
import time

start_time = time.time()


# LOAD config

if len(argv) > 1:
    config = Parser(argv[1])
else:
    config = Parser()   

X = cmfg.profile2d(config)
X.load_centers()
X.select_subsample_centers()




# CMB MAP

nside = int(config['cmb']['filedata_cmb_nside'])
filedata = (f'{config.filenames.datadir_cmb}'
            f'{config.filenames.filedata_cmb_mapa}')

T = hp.read_map(filedata, field=0, h=False, dtype=float)
Q = hp.read_map(filedata, field=1, h=False, dtype=float)
U = hp.read_map(filedata, field=2, h=False, dtype=float)
mask = hp.read_map(filedata, field=4, h=False, dtype=float)
 
# main computations

def f(x, N, rmax, rmax_deg, xr, yr, idxs, G):
    """
    Funcion que calcula la contribucion de un centro
    a los mapas de temperatura y polarización.
    """
    
    center = x[1]
    neigh = NearestNeighbors(n_neighbors=3, radius=0.01)

    Zt = np.zeros((N,N))
    Zq = np.zeros((N,N))
    Zu = np.zeros((N,N))
    Zqr = np.zeros((N,N))
    Zur = np.zeros((N,N))
    Msk = np.zeros((N,N), dtype=int)

    # compute rotation matrix
    phi = float(center.phi)
    theta = float(center.theta)
    pa = float(center.pa) 
    vector = hp.ang2vec(theta, phi)
    rotate_pa = R.from_euler('zyz', [-phi, -theta, pa])

    listpixs = hp.query_disc(nside, vector, rmax.value,
                             inclusive=False, fact=4, nest=False) 

    dists, thetas, tt, qq, uu, qr, mm = [], [], [], [], [], [], []

    for ipix in listpixs:
        v = hp.pix2vec(nside, ipix)
        w = rotate_pa.apply(v)
        dist = hp.rotator.angdist(w, [0, 0, 1])
        theta = atan2(w[1], w[0])
        if theta < 0:
            theta = theta + 2*pi

        # check mask
        if mask[ipix]:
            dists.append(dist[0])
            thetas.append(theta)
            tt.append(T[ipix])
            qq.append(Q[ipix])
            uu.append(U[ipix])
            mm.append(mask[ipix])

    thetas = np.array(thetas)
    dists = np.array(dists)
    tt = np.array(tt)
    qq = np.array(qq)
    uu = np.array(uu)
    mm = np.array(mm)

    # calcular el ángulo psi
    psi2 = 2*thetas
    qr = -qq*np.cos(psi2) - uu*np.sin(psi2)
    ur =  qq*np.sin(psi2) - uu*np.cos(psi2)

    x = dists*np.cos(thetas)*180/pi
    y = dists*np.sin(thetas)*180/pi
    neigh.fit(np.column_stack([x, y]))

    for i, ix in zip(idxs, G):
        rr = np.linalg.norm(ix)

        if rr < rmax_deg.value:
            dist, ind = neigh.kneighbors([ix], 3, return_distance=True)
            dd = np.exp(-dist*25)
            dsts = dd.sum()

            if mm[ind].astype(int).sum()==3:

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

                Msk[i[0], i[1]] = Msk[i[0], i[1]] + 1
 
    res = [Zt, Zq, Zu, Zqr, Zur, Msk]
    return res 

# RUN experiment

N = config.p.adaptative_res_nside
rmax = config.p.r_stop # rad
rmax_deg = rmax*180/pi
xr = np.linspace(-rmax_deg, rmax_deg, N).value
yr = np.linspace(-rmax_deg, rmax_deg, N).value

idxs = itertools.product(range(N), range(N))
idxs = list(idxs)
G = itertools.product(xr, yr)
G = list(G)

Ncen = X.centers.shape[0]
run = (delayed(f)(x, N, rmax, rmax_deg, xr, yr, idxs, G) for x in X.centers.head(Ncen).iterrows()) 
results = Parallel(n_jobs=config.p.n_jobs)(run)

# Save results

fout = f'{config.filenames.dir_output}{config.filenames.experiment_id}/data_{config.filenames.experiment_id}.pk'
 
with open(fout, 'wb') as arch:
   pickle.dump(results, arch)

print(f'All done, elapsed time: {(time.time()-start_time)/60: 12.2f} minutes')
