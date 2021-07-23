from PixelSky import SkyMap
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial.transform import Rotation as R

import pandas as pd
import healpy as hp
import numpy as np
from math import atan2, pi, acos
from tqdm import tqdm
from random import random, sample
from astropy import units as u
import pickle
from sys import exit

pipeline_check = 0

def deplete_profile(A, B, C, r):
    """
    Profile to dilute the number of pixels in rings.
    """
    import numpy as np
    dilute = 1 - A*np.exp(-B*np.exp(-C*r))
    return dilute

def unwrap_run(arg, **kwarg):
    """Wrap the serial function for parallel run.

    This function just call the serialized version, but allows to run
    it concurrently.
    """
    return profile2d.run_batch(*arg, **kwarg)


class profile2d:
    '''
    profile2d (class): compute angular correlations in CMB maps.

    Methods
    -------
    load_centers : load centers from file using config.
    load_tracers : load tracers from file using config.
    initialize_counters : initialize counters
    run : computes the radial profile for a sample.
    run_single : computes the radial profile for a single center.
    run_batch : serial computation of the radial profile for a sample.
    run_batch_II : computes in parallel the radial profile for a sample.
    '''

    def __init__(self, config):
        """Initialize instance of class profile2d.

        This class is prepared to work with a list of centers
        and a pixelized skymap (healpix).  Both datasets must
        be loaded with the methods load_centers, load_tracers.
        """
        self.config = config
        self.centers = None
        self.map = None
        self.mask = None
        global pipeline_check
        pipeline_check = 0

    def load_centers(self):
        """load centers from a galaxy catalogue.

        The galaxy catalogue must contain a row with the column names,
        which also must include:
        - RAdeg: right ascention
        - DECdeg: declination
        - type : galaxy type, following ZCAT convention
          See http://tdc-www.harvard.edu/2mrs/2mrs_readme.html, Sec. F
        - r_ext : galaxy size, in arcsec (from xsc:r_ext)
          https://old.ipac.caltech.edu/2mass/releases/allsky/doc/sec2_3a.html
        """
        self.check_centers()
        conf = self.config.filenames

        # read Galaxy catalog
        glx_catalog = conf.datadir_glx + conf.filedata_glx
        glx = pd.read_csv(glx_catalog, delim_whitespace=True, header=9)

        # Control sample (random centers)
        if self.config.p.control_sample:
            print('( ! ) Randomizing center positions')
            N = glx.shape[0]
            r_l = [random()*360. for _ in range(N)]
            r_b = [random()*2.-1. for _ in range(N)]
            r_b = [acos(r) for r in r_b]
            r_b = [90. - r*180./pi for r in r_b]
            glx['l'] = r_l
            glx['b'] = r_b

        # healpix coordinates
        phi_healpix = glx['l']*np.pi/180.
        theta_healpix = (90. - glx['b'])*np.pi/180.
        glx['phi'] = phi_healpix
        glx['theta'] = theta_healpix
        glx['vec'] = hp.ang2vec(theta_healpix, phi_healpix).tolist()

        # glx angular size [u.rad]
        glxsize = np.array(glx['r_ext'])
        glxsize = 10**glxsize      # !!!!!!!!!!!!!! VERIFICAR
        glxsize = glxsize*u.arcsec
        glxsize = glxsize.to(u.rad)
        glx['glx_size_rad'] = glxsize.value

        # glx physical size [u.kpc]
        self.cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
        z = glx['v']/300000.
        z = z.where(glx['v'] > 1.e-3, 1.e-3)
        glx['v'] = z
        d_A = self.cosmo.angular_diameter_distance(z=glx['v'])
        r_kpc = (glxsize * d_A).to(u.kpc, u.dimensionless_angles())
        r_kpc = r_kpc.value
        glx['glx_size_kpc'] = r_kpc

        # glx position angle
        glx['pa'] = glx['pa']*pi/180.
        if self.config.p.control_angles:
            N = glx.shape[0]
            glx['pa'] = [random()*2*pi for _ in range(N)]

        global pipeline_check
        pipeline_check += 1

        self.centers = glx

    def check_centers(self):
        """Check if the centers file is admisible.

        This method verifies that the dataframe contains the
        neccesary columns.
        """
        conf = self.config.filenames
        # read Galaxy catalog
        glx_catalog = conf.datadir_glx + conf.filedata_glx
        glx = pd.read_csv(glx_catalog, delim_whitespace=True, header=9)
        k = glx.keys()

        f = True
        f = f and 'pa' in k
        f = f and 'RAdeg' in k
        f = f and 'DECdeg' in k

        global pipeline_check
        pipeline_check += 10

        if not f:
            print('Error in loading galaxy data')
            exit()

    def load_tracers(self):
        """load tracers from Healpix map of the CMBR.

        In this case tracers are the pixels used to compute the radial
        temperature profile.
        """
        global pipeline_check
        pipeline_check += 100

        conf = self.config.filenames
        if self.config.p.verbose:
            print(conf.datadir_cmb + conf.filedata_cmb_mapa)

        nside = int(self.config['cmb']['filedata_cmb_nside'])
        npixs = hp.nside2npix(nside)

        mapa = SkyMap(nside)
        filedata = conf.datadir_cmb + conf.filedata_cmb_mapa
        column = int(self.config['cmb']['filedata_field_mapa'])
        mapa.load(filedata, field=(column), verbose=self.config.p.verbose)

        if self.config.p.control_ranmap:
            mapa.data = np.random.normal(loc=1.e-6, scale=1.e-8, size=npixs)

        self.map = mapa

        mask = SkyMap(nside)
        filedata = conf.datadir_cmb + conf.filedata_cmb_mask
        column = int(self.config['cmb']['filedata_field_mask'])
        mask.load(filedata, field=(column), verbose=self.config.p.verbose)

        msk = [True]*npixs
        for ipix, pix in enumerate(mask.data):
            if pix < .1:
                msk[ipix] = False

        self.mask = msk

    def initialize_counters(self, center=None):
        """Initialize counters for temperature map aroung centers.

        Also sets normalization factor if neccesary.
        This is kept separate for organization of the code.
        """
        Nb_r = self.config.p.r_n_bins
        Nb_a = self.config.p.theta_n_bins

        # breaks for angular distance, in radians
        rmin = self.config.p.r_start
        rmax = self.config.p.r_stop
        if not self.config.p.norm_to:
            rmin = rmin.to(u.rad).value
            rmax = rmax.to(u.rad).value
        radii = np.linspace(rmin, rmax, Nb_r+1)

        if center is None:
            norm_factor = 1.
        else:
            norm_factor = 1.
            if self.config.p.norm_to == 'PHYSICAL':
                z = center['v']
                d_A = self.cosmo.angular_diameter_distance(z=z)
                # pass from rad to kpc and divide by glx. size in kpc
                norm_factor = d_A.to(u.kpc, u.dimensionless_angles())
                norm_factor = norm_factor.value / center['glx_size_kpc']
                rmax = rmax*center['glx_size_rad']

            if self.config.p.norm_to == 'ANGULAR':
                # divide by glx. size in rad
                norm_factor = 1. / center['glx_size_rad']
                rmin = rmin*center['glx_size_rad']
                rmax = rmax*center['glx_size_rad']

        # breaks for angle, in radians
        amin = self.config.p.theta_start
        amax = self.config.p.theta_stop

        amin = amin.to(u.rad).value
        amax = amax.to(u.rad).value

        # initialize 2d histogram
        angles = np.linspace(amin, amax, Nb_a+1)
        bins2d = [radii, angles]
        Ht = np.zeros([Nb_r, Nb_a])
        Kt = np.zeros([Nb_r, Nb_a])

        return bins2d, rmax, Ht, Kt, norm_factor

    def select_subsample_centers(self):
        """Make a selection of centers.

        The selection is made by filtering on galxy type and distance
        to the galaxy (redshift). This makes use of the configuration
        paramenters from the .ini file.
        """

        # filter on: galaxy type ----------------
        gtypes = self.config.p.galaxy_types

        Sa_lbl = ['1', '2']
        Sb_lbl = ['3', '4']
        Sc_lbl = ['5', '6']
        Sd_lbl = ['7', '8']
        E_lbl = ['-7', '-6', '-5']
        Sno_lbl = ['10', '11', '12', '15', '16', '19', '20', '98']
        Gtypes = []
        for s in gtypes:
            opt = s.lower()
            if 'spiral' in opt or 'late' in opt:
                opt = 'abcd'
            noellipt = ('early' not in opt) and ('elliptical' not in opt)
            if ('a' in opt or 'sa' in opt) and noellipt:
                Gtypes = sum([Gtypes, Sa_lbl], [])
            if ('b' in opt or 'sb' in opt) and noellipt:
                Gtypes = sum([Gtypes, Sb_lbl], [])
            if ('c' in opt or 'sc' in opt) and noellipt:
                Gtypes = sum([Gtypes, Sc_lbl], [])
            if ('d' in opt or 'sd' in opt) and noellipt:
                Gtypes = sum([Gtypes, Sd_lbl], [])

        for s in gtypes:
            opt = s.lower()
            if 'ellipt' in opt or 'early' in opt:
                opt = 'e'
            if 'e' in opt:
                Gtypes = sum([Gtypes, E_lbl], [])

        filt1 = []
        for t in self.centers['type']:
            f1 = t[0] in Gtypes and not (t[:2] in Sno_lbl)
            f2 = t[0:2] in Gtypes and not (t[:2] in Sno_lbl)
            f = f1 or f2
            filt1.append(f)

        # filter on: redshift ----------------------------
        zmin = self.config.p.redshift_min
        zmax = self.config.p.redshift_max
        filt2 = []
        for z in self.centers['v']:
            f = z > zmin and z < zmax
            filt2.append(f)

        # filter on: elliptical isophotal orientation -----
        boamin = self.config.p.ellipt_min
        boamax = self.config.p.ellipt_max
        filt3 = []
        for boa in self.centers['b/a']:
            f = boa > boamin and boa < boamax
            filt3.append(f)

        # filter on: galaxy size ----------------
        filt4 = []
        if self.config.p.glx_angsize_unit is not None:
            smin = self.config.p.glx_angsize_min
            smax = self.config.p.glx_angsize_max
            for glxsize in self.centers['glx_size_rad']:
                f = glxsize > smin and glxsize < smax
                filt4.append(f)

        if self.config.p.glx_physize_unit is not None:
             smin = self.config.p.glx_physize_min
             smax = self.config.p.glx_physize_max
             for glxsize in self.centers['glx_size_kpc']:
                 f = glxsize > smin and glxsize < smax
                 filt4.append(f)

        # filter all ----------------------------
        filt = []
        for i in range(len(filt1)):
            filt.append(filt1[i] and filt2[i] and filt3[i] and filt4[i])

        #filt1 = np.array(filt1)
        #filt2 = np.array(filt2)
        #filt3 = np.array(filt3)
        #filt4 = np.array(filt4)
        #filt = np.logical_and([filt1, filt2, filt3, filt4])
        #filt = np.logical_and.reduce((filt1, filt2, filt3, filt4))
        # print(self.centers.shape)
        # print(len(filt))

        filt = np.array(filt)
        self.centers = self.centers[filt]

        # limit the number of centers
        if self.config.p.max_centers > 0:
            self.centers = self.centers[:self.config.p.max_centers]

        return None

    def run_single(self, center):
        """Compute the temperature in CMB data around a center.

        This works as a wrapper for the functions
        run_single_montecarlo and run_single_repix.

        Parameters
        ----------
        center : list or array
            List of centers

        Returns
        -------
        H : array like
            Cross product of temperatures
        K : array like
            Counts of pairs contributing to each bin
        """
        if self.config.p.optimize == 'repix':
            # Optimize using low resolution pixels
            Ht, Kt = self.run_single_repix(center)
        elif self.config.p.optimize == 'manual':
            # Optimize using manual dilution
            Ht, Kt = self.run_single_montecarlo(center)
        else:
            # run without optimization
            skymap = self.map
            res = self.initialize_counters(center[1])
            bins2d, rmax, Ht, Kt, norm_factor = res

            # compute rotation matrix
            phi = float(center[1].phi)
            theta = float(center[1].theta)
            pa = float(center[1].pa)

            vector = hp.ang2vec(center[1].theta, center[1].phi)
            if self.config.p.disk_align:
                rotate_pa = R.from_euler('zyz', [-phi, -theta, pa])
            else:
                rotate_pa = R.from_euler('zy', [-phi, -theta])

            listpixs = hp.query_disc(skymap.nside, vector, rmax,
                                     inclusive=False, fact=4, nest=False)
            dists = []
            thetas = []
            temps = []

            for ipix in listpixs:

                if not self.mask[ipix]:
                    continue

                v = hp.pix2vec(skymap.nside, ipix)
                w = rotate_pa.apply(v)

                """Angular distance
                each center is in position [0,0,1] wrt the new system.
                Normalization is made if required from configuration file"""
                dist = hp.rotator.angdist(w, [0, 0, 1])
                dist = dist * norm_factor

                """Position angle
                the angle wrt the galaxy disk (given by the position angle
                results from the projection of the new axes X (i.e. w[0])
                and Y (i.e. w[1])"""
                theta = atan2(w[1], w[0])
                if theta < 0:
                    theta = theta + 2*pi

                temps.append(skymap.data[ipix])
                dists.append(dist[0])
                thetas.append(theta)

            H = np.histogram2d(dists, thetas, bins=bins2d,
                               weights=temps, density=False)
            K = np.histogram2d(dists, thetas, bins=bins2d, density=False)

            Ht = Ht + H[0]
            Kt = Kt + K[0]

        return Ht, Kt

    def run_single_repix(self, center):
        """Compute the temperature in CMB data around a center.

        This version is optimized with two levels of pixelization
        resolution.

        Parameters
        ----------
        center : list or array
            List of centers

        Returns
        -------
        H : array like
            Cross product of temperatures
        K : array like
            Counts of pairs contributing to each bin
        """
        from PixelSky import PixelTools

        skymap = self.map
        bins2d, rmax, Ht, Kt, norm_factor = self.initialize_counters(center[1])

        # compute rotation matrix
        phi = float(center[1].phi)
        theta = float(center[1].theta)
        pa = float(center[1].pa)

        A = self.config.p.dilute_A
        B = self.config.p.dilute_B
        C = self.config.p.dilute_C

        vector = hp.ang2vec(center[1].theta, center[1].phi)
        if self.config.p.disk_align:
            rotate_pa = R.from_euler('zyz', [-phi, -theta, pa])
        else:
            rotate_pa = R.from_euler('zy', [-phi, -theta])

        nside_lowres = self.config.p.adaptative_res_nside
        listp_coarse = hp.query_disc(nside_lowres, vector, rmax,
                                     inclusive=False, fact=4, nest=False)
        px = PixelTools()
        listpixs = []
        for pix_lowres_id in listp_coarse:

            # compute distance to center
            v = hp.pix2vec(nside_lowres, pix_lowres_id)
            w = rotate_pa.apply(v)
            dist = hp.rotator.angdist(w, [0, 0, 1])

            # compute dilution factor
            dilute = deplete_profile(A, B, C, dist)

            # compute the number of pixels in high resolution
            lps_hires = px.spread_pixels(nside_lowres, skymap.nside,
                                         pix_lowres_id, order='ring')

            # dilute pixels for montecarlo estimation (based on pixels)
            Nin = int(len(lps_hires)*dilute)
            lps_sample = sample(lps_hires, Nin)
            for k in lps_sample:
                listpixs.append(k)

        dists = []
        thetas = []
        temps = []

        for ipix in listpixs:

            if not self.mask[ipix]:
                continue

            v = hp.pix2vec(skymap.nside, ipix)
            w = rotate_pa.apply(v)
            dist = hp.rotator.angdist(w, [0, 0, 1])

            dist = dist * norm_factor
            theta = atan2(w[1], w[0])
            if theta < 0:
                theta = theta + 2*pi

            temps.append(skymap.data[ipix])
            dists.append(dist[0])
            thetas.append(theta)

        H = np.histogram2d(dists, thetas, bins=bins2d,
                           weights=temps, density=False)
        K = np.histogram2d(dists, thetas, bins=bins2d, density=False)

        Ht = Ht + H[0]
        Kt = Kt + K[0]

        return Ht, Kt

    def run_single_montecarlo(self, center):
        """Compute the temperature in CMB data around a center.

        This version is optimized with a Monte Carlo estimation
        ot the mean temperatures per radial bin.

        Parameters
        ----------
        center : list or array
            List of centers

        Returns
        -------
        H : array like
            Cross product of temperatures
        K : array like
            Counts of pairs contributing to each bin
        """
        skymap = self.map
        bins2d, rmax, Ht, Kt, norm_factor = self.initialize_counters(center[1])

        # compute rotation matrix
        phi = float(center[1].phi)
        theta = float(center[1].theta)
        pa = float(center[1].pa)

        vector = hp.ang2vec(center[1].theta, center[1].phi)
        if self.config.p.disk_align:
            rotate_pa = R.from_euler('zyz', [-phi, -theta, pa])
        else:
            rotate_pa = R.from_euler('zy', [-phi, -theta])

        # -- annuli for montecarlo estimations of the mean temperature
        p = self.config.p
        imaxs = p.r_avg_cuts
        rmaxs = []
        for i in imaxs:
            r = i / p.r_n_bins * rmax
            rmaxs.append(r)
        factor = p.r_avg_fact
        listp_prev = []
        # --

        for ir, rmx in zip(imaxs, rmaxs):

            listp_now = hp.query_disc(skymap.nside, vector, rmx,
                                      inclusive=False, fact=4, nest=False)

            listpixs = list(set(listp_now) - set(listp_prev))

            Npixs = len(listpixs)
            rf = rmaxs[0] / rmx * factor
            Nin = int(Npixs*rf)

            listpixs = sample(listpixs, Nin)

            dists = []
            thetas = []
            temps = []

            for ipix in listpixs:

                if not self.mask[ipix]:
                    continue

                v = hp.pix2vec(skymap.nside, ipix)
                w = rotate_pa.apply(v)

                dist = hp.rotator.angdist(w, [0, 0, 1])
                dist = dist * norm_factor

                theta = atan2(w[1], w[0])
                if theta < 0:
                    theta = theta + 2*pi

                temps.append(skymap.data[ipix])
                dists.append(dist[0])
                thetas.append(theta)
            factor = p.r_avg_fact

            H = np.histogram2d(dists, thetas, bins=bins2d,
                               weights=temps, density=False)
            K = np.histogram2d(dists, thetas, bins=bins2d, density=False)

            Ht = Ht + H[0]
            Kt = Kt + K[0]

        return Ht, Kt

    def run(self, parallel=None, njobs=1):
        """Compute (stacked) temperature map in CMB data around centers.

        When centers are fixed, it returns the stacked radial profile.

        Parameters
        ----------
        parallel : bool (optional)
            run in parallel?
        njobs : integer (optional)
            number of jobs

        Returns
        -------
        profile : array like
            Array containing the mean temperature per radial bin.
            Angular bins are also returned if required from
            configuration.  The scale of the radial coordinate depends
            on the configuration. All configuration parameters are
            stored in self.config
        """
        if pipeline_check < 111:
            print('Functions load_centers and load_tracers are required')
            exit()

        p = self.config.p
        fn = self.config.filenames

        if isinstance(parallel, bool):
            run_parallel = parallel
        else:
            run_parallel = p.run_parallel

        if self.config.p.verbose:
            print('starting computations...')

        # run on sample data
        if run_parallel:
            res = self.run_batch_II()
        else:
            centers = self.centers
            centers_ids = range(len(centers))
            res = self.run_batch(centers, centers_ids)

        fout = (f"{fn.dir_output}{p.experiment_id}"
                f"/profile_{p.experiment_id}.pk")
        if p.verbose:
            print(fout)
        pickle.dump(res, open(fout, 'wb'))

        # control sample
        if p.control_n_samples > 0:
            theta = self.centers['theta']
            phi = self.centers['phi']
            N = self.centers.shape[0]
            for i in range(p.control_n_samples):
                phi = [random()*2*pi for _ in range(N)]
                cos_theta = [random()*2.-1. for _ in range(N)]
                theta = [acos(r) for r in cos_theta]
                self.centers['theta'] = theta
                self.centers['phi'] = phi

                if run_parallel:
                    res = self.run_batch_II()
                else:
                    centers = self.centers
                    centers_ids = range(len(centers))
                    res = self.run_batch(centers, centers_ids)

                # escribir los randoms
                fout = (f"{fn.dir_output}{p.experiment_id}"
                        f"/control_{p.experiment_id}_{i}.pk")
                if p.verbose:
                    print(fout)
                pickle.dump(res, open(fout, 'wb'))

        H, K = res
        return H, K

    def run_batch(self, centers, index):
        """Compute (stacked) temperature map in CMB data around centers.

        When centers are fixed, it returns the stacked radial profile.

        Parameters
        ----------
        centers : list or array
            List of centers
        index : index number
            a dummy number (required for parallel)

        Returns
        -------
        H : array like
            Cross product of temperatures
        K : array like

        Notes
        -----
        This function admits a pandas dataframe or a row, with the
        type as traversed with df.iterrows()
        When called from unwrap_run, it receives a "row"
        """
        bins2d, rmax, Ht, Kt, norm_factor = self.initialize_counters()

        if isinstance(centers, tuple):
            # a single center
            H, K = self.run_single(centers)
        else:
            # a dataframe
            if self.config.p.showp:
                bf1 = "{desc}: {percentage:.4f}% | "
                bf2 = "{n_fmt}/{total_fmt} ({elapsed}/{remaining})"
                bf = ''.join([bf1, bf2])
                total = centers.shape[0]
                iterator = tqdm(centers.iterrows(),
                                total=total, bar_format=bf)
            else:
                total = centers.shape[0]
                iterator = centers.iterrows()

            H = []
            K = []
            for center in iterator:
                Hi, Ki = self.run_single(center)
                H.append(Hi)
                K.append(Ki)

        return H, K

    def run_batch_II(self):
        """Compute (stacked, parallel) temperature map around centers.

        Paralelization is made on the basis of centers
        """
        from joblib import Parallel, delayed

        njobs = self.config.p.n_jobs

        if self.config.p.verbose:
            vlevel = 11
        else:
            vlevel = 0
        Pll = Parallel(n_jobs=njobs, verbose=vlevel, prefer="processes")
        centers = self.centers
        Ncenters = centers.shape[0]
        ids = np.array(range(Ncenters)) + 1

        cntrs = []
        for c in centers.iterrows():
            cntrs.append(c)

        z = zip([self]*Ncenters, cntrs, ids)

        d_experiment = delayed(unwrap_run)

        results = Pll(d_experiment(i) for i in z)

        # totals:
        # Ht, Kt = np.array(results).sum(axis=0)

        Ht = []
        Kt = []
        for r in results:
            Ht.append(r[0])
            Kt.append(r[1])

        return Ht, Kt

    def testrot(self):
        """Test rotation.

        Loads data and extract a small disc centered on each centers,
        in order to erify if the angles between selected pixels and
        their respective centers are small
        """
        centers = self.centers
        skymap = self.map
        if self.config.p.showp:
            bf1 = "{desc}: {percentage:.4f}% | "
            bf2 = "{n_fmt}/{total_fmt} ({elapsed}/{remaining})"
            bf = ''.join([bf1, bf2])
            total = centers.shape[0]
            iterator = tqdm(centers.itertuples(), total=total, bar_format=bf)
        else:
            iterator = centers.itertuples()

        for center in iterator:

            # compute rotation matrix
            phi = float(center.phi)
            theta = float(center.theta)
            pa = float(center.pa)

            vector = hp.ang2vec(center.theta, center.phi)
            rotate_pa = R.from_euler('zyz', [-phi, -theta, pa])
            listpixs = hp.query_disc(
                        skymap.nside,
                        vector,
                        0.15,
                        inclusive=True,
                        fact=4,
                        nest=False)

            for ipix in listpixs:
                v = hp.pix2vec(skymap.nside, ipix)
                w = rotate_pa.apply(v)
                # w must be ~ [0,0,1]
                print(v, w)

    def show_dilution(self):
        """
        Show the dilution function in pixel optimization.

        The parameters are taken from the settings.
        """
        from matplotlib import pyplot as plt
        import numpy as np

        A = self.config.p.dilute_A
        B = self.config.p.dilute_B
        C = self.config.p.dilute_C

        print(A, B, C)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()

        print(self.centers)

        rmaxs = []
        for center in self.centers.iterrows():
            res = self.initialize_counters(center[1])
            bins2d, rmax, Ht, Kt, norm_factor = res
            rmaxs.append(rmax)

        rmmx = max(rmaxs)
        r = np.linspace(0, rmmx, 500)

        for center in self.centers.iterrows():
            res = self.initialize_counters(center[1])
            bins2d, rmax, Ht, Kt, norm_factor = res
            dilute = deplete_profile(A, B, C, r)
            ax.plot(r, dilute,
                    color='cadetblue', alpha=0.6, linewidth=3)

        for r in rmaxs:
            ax.plot([r, r], [0, deplete_profile(A, B, C, r)],
                    color='slategrey', alpha=0.6, linewidth=1)

        ax.set_xlabel('radial distance, r [rads]')
        ax.set_ylabel(r'depletion factor, $\beta$(r)')
        ax.text(0.5*rmmx, 0.9, r"$\beta$(r) = 1 - A*exp[-B*exp(-C*r)]")
        ax.text(0.5*rmmx, 0.85, f"A = {A}")
        ax.text(0.5*rmmx, 0.8, f"B = {B}")
        ax.text(0.5*rmmx, 0.75, f"C = {C}")
        plt.tight_layout()
        fname = (f"{self.config.filenames.dir_plots}"
                 f"{self.config.p.experiment_id}"
                 f"/depletion_{self.config.p.experiment_id}.png")
        fig.savefig(fname)
        plt.close('all')


def test_rotation(N):
    from math import pi, acos
    from random import random
    alphas = [random()*360 for _ in range(N)]
    deltas = [90 - acos(random()*2 - 1)*180/pi for _ in range(N)]
    for alpha, delta in zip(alphas, deltas):
        colatitude = 90-delta
        theta_healpix = (colatitude)*pi/180
        phi_healpix = alpha*pi/180
        v = hp.ang2vec(theta_healpix, phi_healpix)
        rotate_pa = R.from_euler('zyz', [-alpha, -colatitude, 0], degrees=True)
        w = rotate_pa.apply(v)
        print(f"alpha={alpha}, delta={delta}\n v={v}\nw={w}\n")
        return None
