import numpy as np
from configparser import ConfigParser
from astropy import units as u

DEFAULT_INI = '../set/set_experiment.ini'


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


def choice_yn(string, default_choice=None):
    if string.lower() in 'yesitrue':
        choice = True
    elif string.lower() in 'nofalse':
        choice = False
    else:
        if isinstance(default_choice, bool):
            choice = default_choice
        else:
            raise AttributeError('Check Y/N choice')
    return choice


def is_iterable(obj):
    from collections.abc import Iterable
    return isinstance(obj, Iterable)

# Idea: change named tuple by class:
# class parameters:
# initialize with default parameters.


class Parser(ConfigParser):
    """parser class.

    Manipulation of configuration parameters. This method allows to read a
    configuration file or to set parameters for a Constrained Causally
    Conected Network (C3Net) model.
    """

    def __init__(self, argv=None, *args, **kwargs):
        """Initialize a parser.

        Parameters
        ----------
            None
        Returns
        -------
            None
        Raises
        ------
            Instantiate a Parser object.
        """
        super().__init__()
        self.message = None
        self.check_file(argv)
        self.read_config_file()

        self.load_filenames()
        self.load_config(*args, **kwargs)
        self.check_settings()

    def check_file(self, sys_args=""):
        """Parse paramenters for the simulation from a .ini file.

        Parameters
        ----------
            None

        Raises
        ------
            None

        Returns
        -------
            Updates 'filename' variable in Parser class object.
        """
        from os.path import isfile

        mess = ("Configuration file expected:"
                "\n\t filename or CLI input"
                "\n\t example:  python run_experiment.py"
                f"\n\t {DEFAULT_INI}"
                "\n\t Using default configuration file")
        if isinstance(sys_args, str):
            if isfile(sys_args):
                msg = f"Loading configuration parameters from {sys_args}"
                self.message = msg
                filename = sys_args
            else:
                self.message = "Input argument is not a valid file\
                                Using default configuration file instead"
                filename = DEFAULT_INI

        elif isinstance(sys_args, list):

            if len(sys_args) == 2:
                filename = sys_args[1]

                if isfile(filename):
                    msg = f"Loading configuration parameters from {filename}"
                    self.message = msg
                else:
                    self.message = mess
                    filename = DEFAULT_INI
            else:
                self.message = mess
                filename = DEFAULT_INI

        else:
            self.message = mess
            filename = DEFAULT_INI

        self.filename = filename

    def read_config_file(self):
        """Parse paramenters for the simulation from a .ini file.

        Parameters
        ----------
            None

        Raises
        ------
            None

        Returns
        -------
            None
        """
        self.read(self.filename)

    def load_filenames(self):
        """Make filenames based on info in config file.

        Parameters
        ----------
            None

        Raises
        ------
            None

        Returns
        -------
        self.filenames: named tuple
            Updates the list of filenames and the list of parameters
            in a Parser class object.
        """
        from collections import namedtuple

        experiment_id = self['experiment']['experiment_ID']
        datadir_cmb = self['cmb']['datadir_cmb']
        filedata_cmb_mapa = self['cmb']['filedata_cmb_mapa']
        filedata_cmb_mask = self['cmb']['filedata_cmb_mask']
        datadir_glx = self['glx']['datadir_glx']
        filedata_glx = self['glx']['filedata_glx']
        pickle_name_root = self['out']['pickle_name_root']
        pickle_name_ext = self['out']['pickle_name_ext']
        dir_plots = self['out']['dir_plots']
        plot_name_root = self['out']['plot_name_root']
        plot_format = self['out']['plot_format']
        plot_ftype = self['out']['plot_ftype']

        dir_output = self['out']['dir_output']
        dir_plots = self['out']['dir_plots']
        choice = self['out']['clobber']
        overwrite = choice_yn(choice, default_choice=False)

        names = ['experiment_id',
                 'datadir_cmb',
                 'filedata_cmb_mapa',
                 'filedata_cmb_mask',
                 'datadir_glx',
                 'filedata_glx',
                 'pickle_name_root',
                 'pickle_name_ext',
                 'dir_plots',
                 'plot_name_root',
                 'plot_ftype',
                 'plot_format',
                 'dir_output',
                 'overwrite']

        names = ' '.join(names)
        parset = namedtuple('pars', names)

        res = parset(experiment_id,
                     datadir_cmb,
                     filedata_cmb_mapa,
                     filedata_cmb_mask,
                     datadir_glx,
                     filedata_glx,
                     pickle_name_root,
                     pickle_name_ext,
                     dir_plots,
                     plot_name_root,
                     plot_ftype,
                     plot_format,
                     dir_output,
                     overwrite)
        self.filenames = res

    def load_config(self, keys=None, values=None, *args, **kwargs):
        """Load parameters from config file.

        Parameters
        ----------
        keys: list of strings
            parameters to be reset (mandatory) or set (optional)
        values: list of strings
            Values of the parameters to be reset (mandatory) or set (optional)

        Raises
        ------
            None

        Returns
        -------
        self.config: named tuple
            Updates list of parameters as a named tuple
        """
        from collections import namedtuple
 
        # Override parameter values, if required
        # -------------------------------------------------------------

        # Warning: some parameters admit numbers, other strings, check.
        if isinstance(keys, list):
            # override configuration file with arguments
            if len(keys) != len(values):
                print('Error overriding parameters (using file values)')
            else:
                for k, v in zip(keys, values):
                    for sec in self.sections():
                        has = self.has_option(sec, k)
                        if has:
                            self[sec][k] = v 

        # Obligatory parameters 
        # --------------------------------------------------------

        # List of mandatory parameters:
        secs = ['experiment',
                'run', 'run', 'run', 'run',
                'run', 'run', 'run', 'run']

        pars = ['experiment_id',
                'theta_start', 'theta_stop', 'theta_n_bins', 'theta_units',
                'r_start', 'r_stop', 'r_n_bins', 'r_units'] 

        res = list()
        for sec, par in zip(secs, pars):
            try:
                rpar = self[sec][par]
                res.append(rpar)
            except:
                raise ValueError(f"ERROR: Please check {par} parameter\n"
                                 f"in section {sec} "
                                 f"in the configuration file {self.filename}.")

        # experiment_id
        experiment_id = res[0]

        # theta partition
        theta_start = res[1]
        theta_stop = res[2]
        theta_n_bins = int(res[3])

        theta_units_str = res[4]
        if theta_units_str == 'arcmin':
            theta_units = u.arcmin
        elif theta_units_str == 'arcsec':
            theta_units = u.arcsec
        elif theta_units_str == 'rad':
            theta_units = u.rad
        elif theta_units_str == 'deg':
            theta_units = u.deg
        else:
            theta_units = 1. 

        if is_number(theta_start):
            theta_start = float(theta_start)
            num = 1.
        elif 'pi' in theta_start:
            n = theta_start.replace('pi', '').replace('*', '')
            try:
                num = float(n)
            except Exception:
                num = 1.
            theta_start = num * np.pi
        else:
            print('Error: number not recognized in theta_start')
            exit()
        theta_start = theta_start*theta_units

        if is_number(theta_stop):
            theta_stop = float(theta_stop)
            num = 1.
        elif 'pi' in theta_stop:
            n = theta_stop.replace('pi', '').replace('*', '')
            try:
                float(n)
            except Exception:
                num = 1.
            else:
                num = float(n)
            theta_stop = num * np.pi
        else:
            print('Error: number not recognized in theta_stop')
            exit()
        theta_stop = theta_stop*theta_units

        # r partition (radial distance to center)
        r_start = float(res[5])
        r_stop = float(res[6])
        r_n_bins = int(res[7])

        r_units_str = res[8]
        norm_to = False
        if r_units_str == 'rad':
            r_units = u.rad
        elif r_units_str == 'arcmin':
            r_units = u.arcmin
        elif r_units_str == 'arcsec':
            r_units = u.arcsec
        elif r_units_str == 'parsec':
            r_units = u.parsec
        elif r_units_str == 'kpc':
            r_units = u.kpc
        elif r_units_str in ['physical']:
            norm_to = 'PHYSICAL'
            r_units = 1.*u.dimensionless_unscaled
        elif r_units_str in ['angular']:
            norm_to = 'ANGULAR'
            r_units = 1.*u.dimensionless_unscaled
        elif r_units_str in ['cosine']:
            norm_to = 'COS'
            r_units = 1.*u.dimensionless_unscaled
        else:
            print('Warning: not recognized radial unit or normalization')
            r_units = 1.
        r_start = r_start*r_units
        r_stop = r_stop*r_units

        # Optional parameters parameters
        # ----------------------------------------------------

        try:
            n_jobs = int(self['run']['n_jobs'])
        except KeyError:
            n_jobs = 1.

        try:
            adaptative_resolution = self['run']['adaptative_resolution']
            adaptative_resolution = choice_yn(adaptative_resolution,
                                              default_choice=False)
        except KeyError:
            adaptative_resolution = False

        try:
            max_centers = self['glx']['max_centers']

            if is_number(max_centers):
                max_centers = int(max_centers)
            else:
                yn = choice_yn(max_centers, default_choice=False)
                if not yn:
                    max_centers = -1
        except KeyError:
            max_centers = -1

        try:
            choice = self['glx']['control_sample']
            control_sample = choice_yn(choice, default_choice=False)
        except KeyError:
            control_sample = False

        try:
            choice = self['glx']['control_ranmap']
            control_ranmap = choice_yn(choice, default_choice=False)
        except KeyError:
            control_ranmap = False

        try:
            choice = self['glx']['control_angles']
            control_angles = choice_yn(choice, default_choice=False)
        except KeyError:
            control_angles = False

        try:
            control_n_samples = int(self['glx']['control_n_samples'])
        except KeyError:
            control_n_samples = False

        try:
            choice = self['run']['disk_align']
            disk_align = choice_yn(choice, default_choice=False)
        except KeyError:
            disk_align = False

        try:
            choice = self['run']['run_parallel']
            run_parallel = choice_yn(choice, default_choice=False)
        except KeyError:
            run_parallel = False

        try:
            choice = self['UX']['show_progress']
            showp = choice_yn(choice, default_choice=False)
        except KeyError:
            showp = False

        try:
            choice = self['UX']['verbose']
            verbose = choice_yn(choice, default_choice=True)
        except KeyError:
            verbose = False
        if verbose:
            print('loading parameters...')

        try:
            galaxy_types = self['run']['galaxy_types']
            galaxy_types = galaxy_types.split(' ')
        except KeyError:
            galaxy_types = [] 

        try:
            redshift_min = float(self['run']['redshift_min'])
        except KeyError:
            redshift_min = 0.

        try:
            redshift_max = float(self['run']['redshift_max'])
        except KeyError:
            redshift_max = 1000.
            
        try:
            ellipt_min = float(self['run']['ellipt_min'])
        except KeyError:
            ellipt_min = 0.
            
        try:
            ellipt_max = float(self['run']['ellipt_max'])
        except KeyError:
            ellipt_max = 1.

        try:
            r_avg_cuts = self['run']['r_avg_cuts'].split(' ')
            if is_number(r_avg_cuts[0]):
                r_avg_cuts = [int(i) for i in r_avg_cuts]
                r_avg_cuts = [i if i < r_n_bins else r_n_bins for i in r_avg_cuts]
            else:
                r_avg_cuts = [r_n_bins]
            if r_avg_cuts[-1] < r_n_bins:
                r_avg_cuts.append(r_n_bins)
        except KeyError:
            r_avg_cuts = list()

        try:
            r_avg_fact = float(self['run']['r_avg_fact'])
        except KeyError:
            r_avg_fact = 1.

        if adaptative_resolution:
            optimize = 'repix'
        elif len(r_avg_cuts) > 1:
            optimize = 'manual'
        else:
            optimize = False

        # this is an alternative way to check if parameter is set.
        for key in self['run']:
            if key == 'glx_angsize_min':
                if is_number(self['run']['glx_angsize_min']):
                    glx_angsize_min = float(self['run']['glx_angsize_min'])
                else:
                    glx_angsize_min = 0.
            else:
                glx_angsize_min = 1.e9

        for key in self['run']:
            if key == 'glx_angsize_max':
                if is_number(self['run']['glx_angsize_max']):
                    glx_angsize_max = float(self['run']['glx_angsize_max'])
                else:
                    glx_angsize_max = 1.e9
            else:
                glx_angsize_max = 1.e9

        p_unit = None
        for key in self['run']:
            if key == 'glx_angsize_unit':
                unit = self['run']['glx_angsize_unit'].lower()
                if unit == 'rad':
                    p_unit = u.rad
                elif unit == 'arcmin':
                    p_unit = u.arcmin
                elif unit == 'arcsec':
                    p_unit = u.arcsec
                elif unit.lower() in 'nofalse':
                    p_unit = None
                else:
                    print('glx_angsize_unit not valid in config file'
                          '\nIgnoring filtering on angscal galaxy size.')
                    p_unit = None
        glx_angsize_unit = p_unit

        glx_physize_min = 0.
        for key in self['run']:
            if key == 'glx_physize_min':
                if is_number(self['run']['glx_physize_min']):
                    glx_physize_min = float(self['run']['glx_physize_min'])
                else:
                    glx_physize_min = 0.

        glx_physize_max = 1.e9
        for key in self['run']:
            if key == 'glx_physize_max':
                if is_number(self['run']['glx_physize_max']):
                    glx_physize_max = float(self['run']['glx_physize_max'])
                else:
                    glx_physize_max = 1.e9

        p_unit = None
        for key in self['run']:
            if key == 'glx_physize_unit':
                unit = self['run']['glx_physize_unit'].lower()
                if unit == 'Mpc':
                    p_unit = u.Mpc
                elif unit == 'kpc':
                    p_unit = u.kpc
                elif unit == 'parsec':
                    p_unit = u.parsec
                elif unit.lower() in 'nofalse':
                    p_unit = None
                else:
                    print('glx_physize_unit not valid in config file'
                          '\nIgnoring filtering on physcal galaxy size.')
                    p_unit = None
        glx_physize_unit = p_unit

        # (if no adaptative_res_nside is set, default is no # optimization)
        adaptative_res_nside = int(self['cmb']['filedata_cmb_nside'])
        for key in self['run']:
            if key == 'adaptative_res_nside':
                adaptative_res_nside = int(self['run']['adaptative_res_nside'])

        try:
            choice = self['run']['adaptative_res_dilut']
        except KeyError:
            choice = None
            dilute_A = 0.
            dilute_B = 8
            dilute_C = 15

        if is_iterable(choice):
            dilut_pars = self['run']['adaptative_res_dilut'].split(' ')
            if len(dilut_pars) == 1:
                dilute_A = dilut_pars[0]
            elif len(dilut_pars) == 2:
                dilute_A, dilute_B = dilut_pars
            elif len(dilut_pars) == 3:
                dilute_A, dilute_B, dilute_C = dilut_pars
            else:
                dilute_A = 0.9
                dilute_B = 8
                dilute_C = 15
        elif is_number(choice):
            dilute_A = choice
            dilute_B = 8
            dilute_C = 15
        dilute_A = float(dilute_A)
        dilute_B = float(dilute_B)
        dilute_C = float(dilute_C)

        # Make named tuple
        # --------------------------------------------------------

        names = ['experiment_id',
                 'n_jobs',
                 'control_sample',
                 'control_n_samples',
                 'control_ranmap',
                 'control_angles',
                 'r_start',
                 'r_stop',
                 'r_n_bins',
                 'r_units',
                 'r_avg_cuts',
                 'r_avg_fact',
                 'theta_start',
                 'theta_stop',
                 'theta_n_bins',
                 'theta_units',
                 'norm_to',
                 'adaptative_resolution',
                 'adaptative_res_nside',
                 'dilute_A',
                 'dilute_B',
                 'dilute_C',
                 'disk_align',
                 'galaxy_types',
                 'redshift_min',
                 'redshift_max',
                 'ellipt_min',
                 'ellipt_max',
                 'glx_angsize_min',
                 'glx_angsize_max',
                 'glx_angsize_unit',
                 'glx_physize_min',
                 'glx_physize_max',
                 'glx_physize_unit',
                 'max_centers',
                 'verbose',
                 'run_parallel',
                 'showp',
                 'optimize']

        names = ' '.join(names)
        parset = namedtuple('pars', names)
        res = parset(experiment_id,
                     n_jobs,
                     control_sample,
                     control_n_samples,
                     control_ranmap,
                     control_angles,
                     r_start,
                     r_stop,
                     r_n_bins,
                     r_units,
                     r_avg_cuts,
                     r_avg_fact,
                     theta_start,
                     theta_stop,
                     theta_n_bins,
                     theta_units,
                     norm_to,
                     adaptative_resolution,
                     adaptative_res_nside,
                     dilute_A,
                     dilute_B,
                     dilute_C,
                     disk_align,
                     galaxy_types,
                     redshift_min,
                     redshift_max,
                     ellipt_min,
                     ellipt_max,
                     glx_angsize_min,
                     glx_angsize_max,
                     glx_angsize_unit,
                     glx_physize_min,
                     glx_physize_max,
                     glx_physize_unit,
                     max_centers,
                     verbose,
                     run_parallel,
                     showp,
                     optimize)

        self.p = res

    def check_settings(self):
        """Check if parameters make sense.

        Parameters
        ----------
            None

        Raises
        ------
            None

        Returns
        -------
            Exception if settings have inconsistencies.
        """
        from os import path, makedirs

        if self.p.verbose:
            print(self.message)
            print('Checking settings...')

        # Check or create output directory for the current experiment
        if path.isdir(self.filenames.dir_output):
            dir_exp = (f"{self.filenames.dir_output}/"
                       f"{self.p.experiment_id}")
            try:
                makedirs(dir_exp)
                if self.p.verbose:
                    print("Directory ", dir_exp,  " Created ")
            except FileExistsError:
                pass
                # directory already exists
        else:
            msg = f"Directory {self.filenames.dir_output} does not exist!"
            raise NotADirectoryError(msg)

        # plots directory
        if not path.isdir(self.filenames.dir_plots):
            print(f"Directory {self.filenames.dir_plots} does not exist")
            try:
                makedirs(self.filenames.dir_plots)
                if self.p.verbose:
                    print("Directory ", self.filenames.dir_plots,  " Created ")
            except FileExistsError:
                # directory already exists
                pass

        # plots directory for this experiment
        dir_plt = (f"{self.filenames.dir_plots}/"
                   f"{self.filenames.experiment_id}")

        if not path.isdir(dir_plt):
            print(f"Directory {dir_plt} does not exist")

            try:
                makedirs(dir_plt)
                if self.p.verbose:
                    print("Directory ", dir_plt,  " Created ")
            except FileExistsError:
                # directory already exists
                pass

        if self.filename != DEFAULT_INI:
            rootfn = self.filename.split('.ini')[0].split('/')[-1]
            experiment_id = self['experiment']['experiment_ID']
            if rootfn != experiment_id:
                r = input(f"Warning: the ID for the experiment: "
                          f"{experiment_id}\n         "
                          f"is different from the filename: {rootfn},\n"
                          f"proceed anyway? ")
                ans = choice_yn(r, default_choice=False)
            else:
                ans = True
            if not ans:
                print("Please fix filename or experiment index and try again.")
                exit()


# class ExperimentSuites(Parser):
# 
#     def GenDefaultConfigFile():
# 
#     def suite_generate():
# 
#     def suite():
# 
# 
# # experiment, parsing and math
# import cmfg
# from Parser import Parser
# from sys import argv
# import numpy as np
# import pickle
# import math as m
# import pandas as pd
# 
# ind_min = 12
# ind_max = 22
# 
# inifile = '../set/config_TRNT_' + str(ind_min).zfill(3) + '.ini'
# config = Parser(inifile)
# 
# cnames = config.p._fields
# dfa = pd.DataFrame(columns=cnames)
# 
# for i, ids in enumerate(range(ind_min, ind_max+1)):
#     inifile = '../set/config_TRNT_' + str(ids).zfill(3) + '.ini'
#     print(inifile)
#     config = Parser(inifile)
#     dfa.loc[i] = config.p
# 
# sacar = ['n_jobs', 'verbose', 'run_parallel', 'showp', 'overwrite',
#          'dir_output', 'dir_plots']
# df2 = df = dfa.drop(columns=sacar)
#  
# df2.to_excel('../set/settings_TRNT.xlsx') 
# 
# 
# 
# 
# 
# 
# 
# 
#  
# from openpyxl import load_workbook
# from collections import namedtuple
# import jinja2
# from os.path import join as pathjoin
# 
# wb = load_workbook('SYSH_suite.xlsx')
# print(wb.sheetnames)
# 
# first_column = 'A'
# last_column = 'O'
# 
# first_row = 2
# last_row = 26
# 
# col_range = f"{first_column}:{last_column}"
# 
# suite = wb['SAYSHELL']
# s=suite[col_range]
#  
# for row in suite.iter_rows(min_row=1, max_row=1):
#     names = []
#     nms = ''
#     for cell in row:
#         names.append(cell.value)
#         nms = nms + str(cell.value) + ' '
#     print(names)
#     nms = nms.strip()
# 
# parset = namedtuple('pars', nms)
# 
# 
# source_dir = './'
# template_file = 'SYSH_config_template.ini'
# templateLoader = jinja2.FileSystemLoader(searchpath=source_dir)
# 
# latex_jinja_env = jinja2.Environment(
#     block_start_string=r"\BLOCK{",
#     block_end_string='}',
#     variable_start_string=r'\VAR{',
#     variable_end_string='}',
#     comment_start_string=r'\#{',
#     comment_end_string='}',
#     line_statement_prefix='%%',
#     line_comment_prefix='%#',
#     trim_blocks=True,
#     autoescape=False,
#     loader=templateLoader
# )
# template_intro = latex_jinja_env.get_template(template_file)
# 
# 
# k = 0
# for row in suite.iter_rows(min_row=first_row, max_row=last_row):
#     r = []
#     k += 1
#     for cell in row:
#         r.append(str(cell.value).strip())
# 
#     t = tuple(r)
#     p = parset(*t)
# 
#     ininame = 'config_SYSH_' + str(k).zfill(3) + '.ini'
# 
#     filename = pathjoin(source_dir, ininame)
#     target = open(filename, 'w')
#     target.write(template_intro.render(p=p))
#     target.close()
# 
# """
#     TO DO:
# 
#     Make a template ini file
#     Make a program that read a partial ini file and generates a complete one
# 
# """                            
