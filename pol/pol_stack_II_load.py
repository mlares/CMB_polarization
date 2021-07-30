import numpy as np
import itertools

from math import pi
import cmfg
from Parser import Parser
from sys import argv 
from random import random

from sklearn.neighbors import NearestNeighbors 
import pickle
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import colors, ticker, rc


# CONFIG

if len(argv) > 1:
    config = Parser(argv[1])
else:
    config = Parser()   
    
X = cmfg.profile2d(config)
X.load_centers()
X.select_subsample_centers()


# LOAD

rmax = config.p.r_stop # rad
rmax_deg = rmax.to(u.deg).value

fout = (f'{config.filenames.dir_output}{config.filenames.experiment_id}'
        f'/data_{config.filenames.experiment_id}.pk')

with open(fout, 'rb') as arch:
   results = pickle.load(arch)

# NORMALIZE

N = results[0][0].shape[0]
Zt = np.zeros((N,N))
Zq = np.zeros((N,N))
Zu = np.zeros((N,N))
Zqr = np.zeros((N,N))
Zur = np.zeros((N,N))
Msk = np.zeros((N,N))
Ncen = 0
for r in results:
    Ncen += 1
    m = r[5].astype(bool) # mask
    Zt[m] =  Zt[m] + r[0][m]
    Zq[m] =  Zq[m] + r[1][m]
    Zu[m] =  Zu[m] + r[2][m]
    Zqr[m] = Zqr[m] + r[3][m]
    Zur[m] = Zur[m] + r[4][m]
    Msk = Msk + m
del results

ff = Msk>0
Zt[ff] =  Zt[ff] / Msk[ff] * 1.e6
Zq[ff] =  Zq[ff] / Msk[ff] * 1.e6
Zu[ff] =  Zu[ff] / Msk[ff] * 1.e6
Zqr[ff] = Zqr[ff] / Msk[ff] * 1.e6 
Zur[ff] = Zur[ff] / Msk[ff] * 1.e6 

P = np.sqrt(Zq**2 + Zu**2)
alpha = np.arctan2(Zu, Zq) / 2

print(Zt.max())
print(Zt.min())
mx = max(-Zt.min(), Zt.max())

# RADIAL PROFILE

N = config.p.adaptative_res_nside
xr = np.linspace(-rmax_deg, rmax_deg, N)
yr = np.linspace(-rmax_deg, rmax_deg, N)

idxs = itertools.product(range(N), range(N))
idxs = np.array(list(idxs))
G = itertools.product(xr, yr)
G = np.array(list(G))

neigh = NearestNeighbors(n_neighbors=6, radius=0.01)
neigh.fit(G)

# -------- 
rr = np.linspace(0.05, rmax_deg, 200)
xpolar, ypolar = [], []
for k, r in enumerate(rr):
    nn = 4 + k    
    tt = np.linspace(0, 2*pi, nn, endpoint=False)
    tt = tt + random()*2*pi
    x = r*np.cos(tt)
    y = r*np.sin(tt)
    xpolar.append(x)
    ypolar.append(y)

val_avg = []
for xp, yp in zip(xpolar, ypolar):
    vals = []
    for xx, yy in zip(xp, yp):
        dist, ind = neigh.kneighbors([[xx,yy]], 3, return_distance=True)
        dd = np.exp(-dist*25)
        dsts = dd.sum()
        zz = Zt[idxs[ind][0][:,0], idxs[ind][0][:,1]]
        vals.append(np.dot(dd, zz)/dsts)
    val_avg.append(np.mean(vals))  


pol_avg = []
for xp, yp in zip(xpolar, ypolar):
    vals = []
    for xx, yy in zip(xp, yp):
        dist, ind = neigh.kneighbors([[xx,yy]], 3, return_distance=True)
        dd = np.exp(-dist*25)
        dsts = dd.sum()
        zz = P[idxs[ind][0][:,0], idxs[ind][0][:,1]]
        vals.append(np.dot(dd, zz)/dsts)
    pol_avg.append(np.mean(vals))  
 
 
alf_avg = []
for xp, yp in zip(xpolar, ypolar):
    vals = []
    for xx, yy in zip(xp, yp):
        dist, ind = neigh.kneighbors([[xx,yy]], 3, return_distance=True)
        dd = np.exp(-dist*25)
        dsts = dd.sum()
        zz = alpha[idxs[ind][0][:,0], idxs[ind][0][:,1]]
        vals.append(np.dot(dd, zz)/dsts)
    alf_avg.append(np.mean(vals))  
 






# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = cmfg.MidpointNormalize(vmin=-mx, vmax=mx, midpoint=0.)

sc = ax.imshow(Zt, cmap='RdBu_r',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg],
               norm=nr)

circle1 = plt.Circle((0, 0), rmax_deg, fc='None', linewidth=6,
                     #color=(0.0196, 0.188, 0.38, 0.5))
                     color='white')
ax.add_patch(circle1)

cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'averaged temperature [$\mu$K]')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#cb.formatter.set_powerlimits((1, 1))
cb.update_ticks()
plt.tight_layout()

pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/Zt_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)


plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = cmfg.MidpointNormalize(vmin=Zq.min(), vmax=Zq.max(), midpoint=0.)

sc = ax.imshow(Zq, cmap='bwr',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg],
               norm=nr)

circle1 = plt.Circle((0, 0), rmax_deg, fc='None', linewidth=6,
                     color='white')
ax.add_patch(circle1)

cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\times\; 10^{-6}\quad$ Q Stokes parameter')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
#ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#cb.formatter.set_powerlimits((-6, -6))
#cb.update_ticks()
plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/Zq_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)   


# ------------


plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = cmfg.MidpointNormalize(vmin=Zq.min(), vmax=Zq.max(), midpoint=0.)

sc = ax.imshow(Zu, cmap='bwr',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg],
               norm=nr)

circle1 = plt.Circle((0, 0), rmax_deg, fc='None', linewidth=6,
                     color='white')
ax.add_patch(circle1)

cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\times\; 10^{-6}\quad$ U Stokes parameter')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
#ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#cb.formatter.set_powerlimits((-6, -6))
#cb.update_ticks()
plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/Zu_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)

# ------------


plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = cmfg.MidpointNormalize(vmin=Zqr.min(), vmax=Zqr.max(), midpoint=0.)

sc = ax.imshow(Zqr, cmap='bwr',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg],
               norm=nr)

circle1 = plt.Circle((0, 0), rmax_deg, fc='None', linewidth=6,
                     color='white')
ax.add_patch(circle1)

cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\times\; 10^{-6}\quad$ Q$_r$ Stokes parameter')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
#ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#cb.formatter.set_powerlimits((-6, -6))
#cb.update_ticks()
plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/Zqr_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)   
 
# ------------


plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = cmfg.MidpointNormalize(vmin=Zq.min(), vmax=Zq.max(), midpoint=0.)

sc = ax.imshow(Zur, cmap='bwr',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg],
               norm=nr)

circle1 = plt.Circle((0, 0), rmax_deg, fc='None', linewidth=6,
                     color='white')
ax.add_patch(circle1)

cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\times\; 10^{-6}\quad$ U$_r$ Stokes parameter')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
#ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#cb.formatter.set_powerlimits((-6, -6))
#cb.update_ticks()
plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/Zur_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)

# ------------


plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

ax.plot(rr, val_avg)
ax.axhline(0, linestyle='--', color='silver')

ax.set_xlabel('radial distance [deg]')
ax.set_ylabel(r'averaged temperature [$\times 10^6\,\mu$K]')
plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/Zt_radial_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)

# ------------
 

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

ax.plot(rr, pol_avg)
ax.axhline(0, linestyle='--', color='silver')

ax.set_xlabel('radial distance [deg]')
ax.set_ylabel(r'averaged polarization flux [$\times 10^6]')
plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/pol_radial_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)

# ------------               
 
plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

ax.plot(rr, alf_avg)
ax.axhline(0, linestyle='--', color='silver')

ax.set_xlabel('radial distance [deg]')
ax.set_ylabel(r'averaged angle')
plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/alf_radial_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)

# ------------               
 

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = cmfg.MidpointNormalize(vmin=Zq.min(), vmax=Zq.max(), midpoint=0.)

sc = ax.imshow(Zur, cmap='bwr',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg],
               norm=nr)

circle1 = plt.Circle((0, 0), rmax_deg, fc='None', linewidth=6,
                     color='white')
ax.add_patch(circle1)

cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\times\; 10^{-6}\quad$ U$_r$ Stokes parameter')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
#ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#cb.formatter.set_powerlimits((-6, -6))
#cb.update_ticks()
plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/Zur_radial_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)



# tails
tx = G[:,0]
ty = G[:,1]

# heads
dx = (P*np.cos(alpha)).reshape(N*N)*10000
dy = (P*np.sin(alpha)).reshape(N*N)*10000
hx = tx + dx
hy = ty + dy

radial = np.sqrt(G[:,0]**2 + G[:,1]**2)
filt = radial < rmax_deg

plt.close('all')
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_subplot(2, 1, 1)
zz = Zq.reshape(N*N)[filt]
ax1.hist(zz, bins=50, density=True)
ax1.set_xlim(-2.5, 2.5)
ax1.set_xlabel('Q')
ax1.set_ylabel(r'dN/dQ')

ax2 = fig.add_subplot(2, 1, 2)
zz = Zu.reshape(N*N)[filt]
ax2.hist(zz, bins=50, density=True)
ax2.set_xlim(-2.5, 2.5)
ax2.set_xlabel('U')
ax2.set_ylabel(r'dN/dU') 

plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/hists_QU_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)

ql = []
ul = []
pl = []
al = []
for i, j in idxs:
    r = np.sqrt(xr[i]**2 + yr[i]**2)
    if r < rmax_deg:
        if abs(Zq[i, j]) > 1.e-6 and abs(Zu[i, j]) > 1.e-6:
            ql.append(Zq[i, j])
            ul.append(Zu[i, j])

            P = np.sqrt(Zq[i,j]**2 + Zu[i,j]**2)
            alpha = np.arctan2(Zu[i,j], Zq[i,j]) / 2

            pl.append(P)
            al.append(alpha)

ql = np.array(ql)
ul = np.array(ul)

font = {'family' : 'DejaVu Sans',
        'weight' : 'medium',
        'size'   : 14}
rc('font', **font)

plt.close('all')
fig = plt.figure(figsize=(7, 5))

ax = fig.add_subplot()
ax.plot(ql, ul-ql, marker='o', markersize=12, color=(0, 0.7, 1, 0.01), linestyle='None')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(color='silver')
ax.set_xlabel(r'Q  [$\times 10^6 \, \mu$K]', fontsize=16)
ax.set_ylabel(r'U - Q  [$\times 10^6 \, \mu$K]', fontsize=16)

plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/QU_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)


# ---------

N = config.p.adaptative_res_nside
rmax = config.p.r_stop # rad
rmax_deg = rmax*180/pi
xr = np.linspace(-rmax_deg, rmax_deg, N).value
yr = np.linspace(-rmax_deg, rmax_deg, N).value

idxs = itertools.product(range(N), range(N))
idxs = list(idxs)
G = itertools.product(xr, yr)
G = list(G)   

gg = np.array(G)
pos_x = gg[:,0]
pos_y = gg[:,1]

rx = rayita_x.reshape(1, N*N)
ry = rayita_y.reshape(1, N*N)

# SAMPLE
samp = np.random.choice([True] + 25*[False], N*N)
pos_x = pos_x[samp]
pos_y = pos_y[samp]
rx =    rx[0][samp]
ry =    ry[0][samp]


plt.close('all')
fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot() 

opts = dict(color='k', headlength=0, pivot='middle',
            scale=3, linewidth=.5, units='xy', width=.5, 
            headwidth=1, headaxislength=0)

ax.plot(x, y, marker='o', markersize=0.5, linestyle='None', color='red')

rayita_x = P*np.cos(alpha)*10000
rayita_y = P*np.sin(alpha)*10000
plt.quiver(pos_x, pos_y, rx, ry, **opts)       

plt.tight_layout()
pltname = (f'{config.filenames.dir_plots}{config.filenames.experiment_id}'
           f'/quiver_{config.filenames.experiment_id}.pdf')
fig.savefig(pltname)
