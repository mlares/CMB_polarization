import pickle
import numpy as np

import cmfg
from Parser import Parser
from math import pi
from astropy import units as u
import itertools
from sklearn.neighbors import NearestNeighbors 
from matplotlib import pyplot as plt
from random import random
from matplotlib import colors, ticker, rc


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



with open('../out/POL03/data_POL03.pk', 'rb') as arch:
   results = pickle.load(arch)

N = results[0][0].shape[0]
Zt = np.zeros((N,N))
Zq = np.zeros((N,N))
Zu = np.zeros((N,N))
Zqr = np.zeros((N,N))
Zur = np.zeros((N,N))
Ncen = 0
for r in results:
    Ncen += 1
    Zt =  Zt + r[0]
    Zq =  Zq + r[1]
    Zu =  Zu + r[2]
    Zqr = Zqr + r[3]
    Zur = Zur + r[4]
del results


Zt =  Zt / Ncen * 1.e6
Zq =  Zq / Ncen * 1.e6
Zu =  Zu / Ncen * 1.e6
Zqr = Zqr / Ncen* 1.e6 
Zur = Zur / Ncen* 1.e6 

P = np.sqrt(Zq**2 + Zu**2)
alpha = np.arctan2(Zu, Zq) / 2

Pr = np.sqrt(Zqr**2 + Zur**2)
alphar = np.arctan2(Zur, Zqr) / 2

# ADDITIONAL DATA »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

config = Parser('../set/POL03.ini')
X = cmfg.profile2d(config)
X.load_centers()
X.select_subsample_centers()
                           
rmax = config.p.r_stop # rad
rmax_deg = rmax.to(u.deg).value

print('rmax_deg ------> ', rmax_deg)




# COMPUTE RADIAL PROFILE

N = 120
xr = np.linspace(-rmax_deg, rmax_deg, N)
yr = np.linspace(-rmax_deg, rmax_deg, N)

idxs = itertools.product(range(N), range(N))
idxs = np.array(list(idxs))
G = itertools.product(xr, yr)
G = np.array(list(G))

neigh = NearestNeighbors(n_neighbors=6, radius=0.01)
neigh.fit(G)

# -------- 
rr = np.linspace(0.02, 2.8, 100)
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


# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = MidpointNormalize(vmin=-15, vmax=15, midpoint=0.)

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
fig.savefig('Zt_POL03.png')   





# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = MidpointNormalize(vmin=Zq.min(), vmax=Zq.max(), midpoint=0.)

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
fig.savefig('Zq_POL03.png')   

# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = MidpointNormalize(vmin=Zq.min(), vmax=Zq.max(), midpoint=0.)

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
fig.savefig('Zu_POL03.png')   
                             


# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = MidpointNormalize(vmin=Zqr.min(), vmax=Zqr.max(), midpoint=0.)

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
fig.savefig('Zqr_POL03.png')   
                           

 
# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = MidpointNormalize(vmin=Zq.min(), vmax=Zq.max(), midpoint=0.)

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
fig.savefig('Zur_POL03.png')  



plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

ax.plot(rr, val_avg)
ax.axhline(0, linestyle='--', color='silver')

ax.set_xlabel('radial distance [deg]')
ax.set_ylabel(r'averaged temperature [$\times 10^6\,\mu$K]')
plt.tight_layout()
fig.savefig('Zt_POL03_radial.png')  


# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»




plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

nr = MidpointNormalize(vmin=Zq.min(), vmax=Zq.max(), midpoint=0.)

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
fig.savefig('Zur_b_POL03.png')  



# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
# P y angulo -----------------------------

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()
sc = ax.imshow(P, cmap='pink_r',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg])
cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\times\; 10^{-6}\quad$ P')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
plt.tight_layout()
fig.savefig('P_POL03.png')  



# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()
sc = ax.imshow(alpha, cmap='bwr',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg])
cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\alpha$ [rad]')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
plt.tight_layout()
fig.savefig('alpha_POL03.png')  


# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()

ax.hist(alpha)

ax.set_xlabel('alpha [rad]')
ax.set_ylabel('dN/d(alpha)')
plt.tight_layout()
fig.savefig('alpha_hist_POL03.png')  




# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
# P y angulo -----------------------------

plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()
sc = ax.imshow(Pr, cmap='pink_r',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg])
cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\times\; 10^{-6}\quad$ P')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
plt.tight_layout()
fig.savefig('P_r_POL03.png')  



# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
plt.close('all')
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot()
sc = ax.imshow(alphar, cmap='bwr',
               extent=[-rmax_deg, rmax_deg, -rmax_deg, rmax_deg])
cb = plt.colorbar(sc, ax=ax, shrink=0.8, aspect=60)
cb.set_label(r'$\alpha$ [rad]')
ax.set_xlabel('x [deg]')
ax.set_ylabel('y [deg]')
plt.tight_layout()
fig.savefig('alpha_r_POL03.png')  







# tails
tx = G[:,0]
ty = G[:,1]

# heads
dx = (P*np.cos(alpha)).reshape(N*N)*10000
dy = (P*np.sin(alpha)).reshape(N*N)*10000
hx = tx + dx
hy = ty + dy

filt = dx > 1.e-4

for i in range(N*N):
    if filt[i]:
        print(tx[i], hx[i], ty[i], hy[i])        
        

# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

plt.close('all')
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_subplot(2, 1, 1)
zz = Zq.reshape(N*N)
ax1.hist(zz, bins=50, density=True)
ax1.set_xlim(-1.5, 1.5)
ax1.set_xlabel('Q')
ax1.set_ylabel(r'dN/dQ')


ax2 = fig.add_subplot(2, 1, 2)
zz = Zu.reshape(N*N)
ax2.hist(zz, bins=50, density=True)
ax2.set_xlim(-1.5, 1.5)
ax2.set_xlabel('U')
ax2.set_ylabel(r'dN/dU') 

plt.tight_layout()
fig.savefig('hists_POL03_radial.png')  



# PLOTS »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»

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

            P_tmp = np.sqrt(Zq[i,j]**2 + Zu[i,j]**2)
            alpha_tmp = np.arctan2(Zu[i,j], Zq[i,j]) / 2

            pl.append(P_tmp)
            al.append(alpha_tmp)

ql = np.array(ql)
ul = np.array(ul)


font = {'family' : 'normal',
        'weight' : 'medium',
        'size'   : 14}
rc('font', **font)


plt.close('all')
fig = plt.figure(figsize=(7, 5))

ax = fig.add_subplot()
ax.plot(ql, ul-ql, marker='o', markersize=12, color=(0, 0.7, 1, 0.01), linestyle='None')

ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.25, 0.25)
ax.grid(color='silver')
ax.set_xlabel(r'Q  [$\times 10^6 \, \mu$K]', fontsize=16)
ax.set_ylabel(r'U - Q  [$\times 10^6 \, \mu$K]', fontsize=16)

plt.tight_layout()
fig.savefig('qu_POL03.png')


