
from Parser import Parser
from sys import argv 
#import PixelSky as pxs
import cmfg
#from configparser import ConfigParser

if len(argv) > 1:
    config = Parser(argv[1])
else:
    config = Parser()


#filename = pxs.check_file(argv)
#config = ConfigParser()
#config.read(filename)
           
#config = Parser('../set/POL03.ini')
#X = cmfg.profile2d(config)
#X.load_centers()
#X.select_subsample_centers()
                              
