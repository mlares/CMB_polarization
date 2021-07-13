class SkyMap():
    '''class SkyMap: utils to work with healpix pixelized maps. 

    Methods
    -------
        load: loads a CMB map
    '''

    import healpy as hp

    def __init__(self, nside=256, ordering='ring', frame='equatorial'):

        import healpy as hp

        self.nside = nside
        self.ordering = 'ring'
        self.frame = 'equatorial'
        self.npixs = hp.nside2npix(self.nside)

        # fac =
        # Dsel: sample of galaxies
        # self, skymap, nside,fac,rprof,Dsel,vec,hp_data_sel,hp_mask):

    def __len__(self):
        return self.npixs

    def __repr__(self):
        return 'Sky Map with {!s} pixels in {!s} order'.format(
            self.npixs, self.ordering)

    def __str__(self):
        return 'Sky Map with {!s} pixels in {!s} order'.format(
            self.npixs, self.ordering)

    def load(self, filename, *args, **kwargs):
        '''
        Reads the CMB map

        Args:
            filename (str): the file name of the map to be read

        Raises:

        Returns:
            readmap: a healpix map, class ?

        '''

        import healpy as hp

        d = hp.read_map(filename, h=True, dtype=float, **kwargs)
        self.data = d[0]
        self.header = d[1]

        return(True)

    def apply_mask(self, mask):

        import healpy as hp

        m = self[0].copy()
        k = mask[0].copy()
        m[k < 0.5] = hp.UNSEEN
        masked_map = hp.ma(m)
        return(masked_map)


class PixelTools:

    def spread_pixels(self, Nside_low, Nside_high, ID, order='nest'):
        """
        returns a list of pixel IDs in the Nside_high resolution
        from a pixel ID in the Nside_low resolution.
        """
        from math import log
        import numpy as np
        import healpy as hp

        if order != 'nest' and order != 'ring':
            raise KeyError('ERROR: check order in spread_pixels')

        if Nside_low == Nside_high:
            if isinstance(ID, list):
                return ID
            else:
                return [ID]

        if Nside_low > Nside_high:
            raise KeyError('ERROR using spread_pixels')

        Llow = int(log(Nside_low, 2))
        Lhigh = int(log(Nside_high, 2))

        if isinstance(ID, list) or isinstance(ID, np.ndarray):
            pixids = []
            if order == 'ring':
                for ipix in ID:
                    j = hp.ring2nest(Nside_low, ipix)
                    pixids.append(j)
            else:
                pixids = ID

            pix_IDs = []
            for id in pixids:
                b = bin(id)
                DN = Lhigh-Llow
                a = [bin(i)[2:].zfill(2*DN) for i in range(4**DN)]
                for i in a:
                    x = (b[2:].zfill(Llow) + i)
                    pix_IDs.append(int(x, 2))
        elif isinstance(ID, int) or isinstance(ID, np.int64):

            if order == 'ring':
                pixids = hp.ring2nest(Nside_low, ID)
            else:
                pixids = ID

            b = bin(pixids)
            DN = Lhigh-Llow
            a = [bin(i)[2:].zfill(2*DN) for i in range(4**DN)]
            pix_IDs = []
            for i in a:
                x = (b[2:].zfill(Llow) + i)
                pix_IDs.append(int(x, 2))
        else:
            print('wtf')
            pix_IDs = 0

        if order == 'ring':
            for i in range(len(pix_IDs)):
                pix_IDs[i] = hp.nest2ring(Nside_high, pix_IDs[i])

        return(pix_IDs)

    def downgrade_pixels(self, Nside_low, Nside_high, ID):
        """
        Function to be used for testing purposes
        It accomplish the same task than spread_pixels,
        but with a less efficient, brute force method.
        """
        import healpy as hp

        Npix_high = hp.nside2npix(Nside_high)
        listp = []
        a = []
        for i in range(Npix_high):
            v = hp.pix2vec(Nside_high, i, nest=True)
            k = hp.vec2pix(Nside_low, v[0], v[1], v[2], nest=True)
            if (k == ID):
                listp.append(i)
                a.append(bin(i))
        return([listp, a])
# from random import random
# N = 1000
# dists = [random()*0.029 for _ in range(N)]
# thetas= [random()*3.14 for _ in range(N)]
# temps = [random() for _ in range(N)]
# brad = ap.breaks_rad.to(u.rad).value
# bang = ap.breaks_ang.value
# bins2d = [brad, bang]
# H1 = np.histogram2d(dists, thetas, bins=bins2d, weights=temps,
#                     density=False)[0]
# dists = [random()*0.029 for _ in range(N)]*u.rad
# thetas= [random()*3.14 for _ in range(N)]*u.rad
# temps = [random() for _ in range(N)]
# brad = ap.breaks_rad.to(u.rad)
# bang = ap.breaks_ang
# bins2d = [brad, bang]
# H2 = np.histogram2d(dists, thetas, bins=bins2d, weights=temps,
#                     density=False)[0]
#   from astropy import coordinates as coo
#   from astropy.modeling import rotations as R
#   from astropy import units as u
#   phi = float(center.phi)*u.deg
#   theta = float(center.theta)*u.deg
#   pa = float(center.pa)*u.deg
#   r = R.from_euler('zxz', [phi.value, theta.value, pa.value], degrees=True)
#   v = coo.spherical_to_cartesian(1., theta.to(u.rad), phi.to(u.rad))
#   r.apply(v)
#   #########
#   alpha = np.random.rand()*360.*u.deg
#   delta = math.acos(np.random.random()*2.-1)*u.rad
#   delta = delta.to(u.deg) - 90.*u.deg
#   phi = 90.*u.deg
#   v = coo.spherical_to_cartesian(1., alpha, delta)
#   #Note that the input angles should be in latitude/longitude or
#   #elevation/azimuthal form.  I.e., the origin is along the equator
#   #rather than at the north pole.
#   lon = alpha
#   lat = delta
#   phi = phi
#   r = R.EulerAngleRotation(lon, lat, phi, 'zyz')
#   #Rotates one coordinate system into another (fixed) coordinate system.
#   #All coordinate systems are right-handed. The sign of the angles is
#   #determined by the right-hand rule.
#   ############
#   v = [6, 5, 20.]
#   v = v / np.sqrt(np.dot(v,v))
#   d, lat, lon = coo.cartesian_to_spherical(v[0], v[1], v[2])
#   phi = 0.*u.rad
#   r = R.EulerAngleRotation(lon, lat, phi, 'zyz')
#   v_new = r.spherical2cartesian(lon, lat)
#   r = R.from_euler('zyz', [lon.value, lat, 0.], degrees=False)
#   r.apply(v)
#   """
#   Source code for astropy.modeling.rotations
#   # Licensed under a 3-clause BSD style license - see LICENSE.rst
#   https://docs.astropy.org/en/stable/_modules/astropy/modeling/rotations.html
#   Implements rotations, including spherical rotations
#   as defined in WCS Paper II
#   [1]_
#   `RotateNative2Celestial` and `RotateCelestial2Native`
#   follow the convention in
#   WCS Paper II to rotate to/from a native sphere and the celestial sphere.
#   The implementation uses `EulerAngleRotation`. The model parameters are
#   three angles: the longitude (``lon``) and latitude (``lat``) of the
#   fiducial point in the celestial system (``CRVAL`` keywords in FITS),
#   and the longitude of the celestial pole in the native system
#   (``lon_pole``). The Euler angles are ``lon+90``, ``90-lat`` and
#   ``-(lon_pole-90)``.  """
#   #  # esto anda:
#   #  v = [0., 1., 0.]
#   #  d, lat, lon = coo.cartesian_to_spherical(v[0], v[1], v[2])
#   #  lat = lat.value + np.pi/2.
#   #  r = R.from_euler('zyz', [lon.value, lat, 0.], degrees=False)
#   #  r.apply(v)
#   #  v = [0., 1., 1.]
#   #  v = v / sqrt(np.dot(v,v))
#   #  d, lat, lon = coo.cartesian_to_spherical(v[0], v[1], v[2])
#   #  lat = lat.value + np.pi/2.
#   #  r = R.from_euler('zyz', [lon.value, lat, 0.], degrees=False)
#   #  r.apply(v)
