"""
PROJECT: POLARIZATION OF THE CMB BY FOREGROUNDS


"""
import numpy as np
import itertools
from math import atan2, pi, acos

import cmfg
from Parser import Parser
from sys import argv 

import pickle
import cmfg

if len(argv) > 1:
    config = Parser(argv[1])
else:
    config = Parser()   

fout = f'{config.filenames.dir_output}{config.filenames.experiment_id}/data_{config.filenames.experiment_id}.pk'
print('escribiendo datos...')
print(fout)
