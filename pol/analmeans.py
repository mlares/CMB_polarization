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

x = [*L[1], *L[2], *L[3], *L[4], *L[5]]
x = np.array(x)
y = L[0]

Nx = len(x)
Ny = len(y)

statistic = ( x.mean() - y.mean() ) / np.sqrt( x.var()/Nx + y.var()/Ny)


from scipy import stats

df = Nx + Ny - 1

p_value = 1 - stats.t.cdf(statistic, df=df)