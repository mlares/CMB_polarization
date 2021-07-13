import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import healpy as hp

cmap_colors = get_cmap('binary', 256)(np.linspace(0, 1, 256))
cmap_colors[..., 3] = 0.4  # Make colormap partially transparent
cmap = ListedColormap(cmap_colors)

m = hp.read_map('wmap_band_iqumap_r9_9yr_K_v5.fits', (0, 1, 2), verbose=False)
I, Q, U = hp.smoothing(m, np.deg2rad(5))
lic = hp.line_integral_convolution(Q, U)

lic = hp.smoothing(lic, np.deg2rad(0.5))

hp.mollview(np.log(1 + np.sqrt(Q**2 + U**2) * 100), cmap='inferno', cbar=False)
hp.mollview(lic, cmap=cmap, cbar=False, reuse_axes=True, title='WMAP K')

plt.savefig('wmapk.png', dpi=150)
