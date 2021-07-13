import pickle
import numpy as np
from scipy.stats import f as F
from scipy import stats

from matplotlib import pyplot as plt

# READ DATA
ang = [0, 15, 30, 45, 60, 75]

stokes_param = 'S'

L = []
for a in ang:
    S = stokes_param + 'pol_' + str(a) + 'deg.pickle'
    filename = '/'.join(['data', S])
    with open(filename, "rb") as input_file:
        e = pickle.load(input_file)
        L.append(e)


# PLOT DISTRIBUTION
b = np.linspace(0, 120, 200)

fig = plt.figure()
ax = fig.add_subplot()

lab = 'control samples'
for x in L[1:]:
    H = np.histogram(x, density=True, bins=b)
    ax.plot(H[1][1:], H[0], color='thistle',label=lab, linewidth=3)
    lab=None

x = L[0]
H = np.histogram(x, density=True, bins=b)
ax.plot(H[1][1:], H[0], color='teal', label='galaxies', linewidth=1)
ax.set_xlabel(stokes_param + ' stokes')
ax.set_ylabel('1/N dN/d' + stokes_param)
ax.legend()

fig.savefig('plot.png')
plt.close()


# PLOT QQ
#-----------------------------------------------

fig = plt.figure()
ax = fig.add_subplot()
interpolation = 'nearest'
y = np.array(L[0])
y = y[abs(y)<140]

ax.plot([0, 140], [0, 140], color='grey', linestyle='--')

for xx in L[1:]:
    x = np.array(xx)
    x = xx[abs(xx)<140]
    n_quantiles = min(len(y), len(x))
    quantiles = np.linspace(start=0, stop=1, num=int(n_quantiles))
    qy = np.quantile(y, quantiles, interpolation=interpolation)
    qx = np.quantile(x, quantiles, interpolation=interpolation)
    ax.plot(qy, qx)

ax.set_title('QQ plot for S=\sqrt{U^2+Q^2} stokes')
ax.set_xlabel('selected sample quantiles')
ax.set_ylabel('control sample quantiles')
fig.savefig('plot_qqplank.png') 
plt.close()

#-------------------------------------

x = L[1]
y = L[0]
df1 = len(x) - 1
df2 = len(y) - 1

fig = plt.figure()
ax = fig.add_subplot()

y = np.array(L[0])
vary = np.var(y)

t = np.linspace(0.925, 1.1, 100)
yt = F.pdf(t, len(y), len(x))
ax.plot(t, yt, label='F distribution')

lab = 'sample / control sample'
for xx in L[1:]:
    x = np.array(xx)
    r = vary / np.var(x)
    s=stats.f_oneway(x, y)
    ax.plot([r, r], [0, 40], c='tomato', label=lab)
    lab=None

lab = 'pairs of control samples'
for i in range(1,6):
    for j in range(i+1,6):
        x1 = np.array(L[i])
        x2 = np.array(L[j])
        r = np.var(x1) / np.var(x2)
        ax.plot([r, r], [0, 20], c='grey', label=lab)
        r = np.var(x2) / np.var(x1)
        ax.plot([r, r], [0, 20], c='grey')
        lab=None

ax.set_title(f'Variance ratio F-test (Stokes: {stokes_param})')
ax.set_xlabel('Variance ratio')
ax.set_ylabel('theoretical distribution')
ax.legend()
fig.savefig('F.png') 
plt.close()
 





