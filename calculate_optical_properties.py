#!/usr/bin/env python3
import sys,os,subprocess,shutil,datetime
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from contextlib import contextmanager
from optical_element_io import cd

# takes optical_element object (oe) as argument
def calc_properties_mirror(oe,nterms=50):
    '''
    untested function for calculating optical properties of electrostatic
    mirrors with MIRROR. 

    Parameters:
        oe : OpticalElement object 
            optical element for which to calculate optical properties.

    Optional parameters:
        nterms : int
            number of terms to use in hermite series to approximate field.
            default 50
    '''
    oe.fitname = oe.basename_noext+'.fit'
    with cd(oe.dirname):
        outputmode = subprocess.PIPE if oe.verbose else None
        try:
            output = subprocess.run(['herm1.exe',oe.potname,oe.fitname,nterms,n,n],stdout=outputmode,timeout=oe.timeout).stdout
            print(output.decode('utf-8')) if oe.verbose else 0
        except TimeoutExpired:
            print('Optical properties calculation failed. Rerunning.')
            calc_properties_mirror(oe,nterms)


def calc_properties_optics(oe):
    '''
    function for calculating optical properties of electric or magnetic lenses
    with OPTICS ABER5. 

    Parameters:
        oe : OpticalElement object 
            optical element for which to calculate optical properties.
    '''
    with cd(oe.dirname):
        outputmode = subprocess.PIPE if oe.verbose else None
        if(os.path.exists(os.imgcondfilename) != True):
            print('No optical imaging conditions file found. Run OpticalElement.write_opt_img_cond_file() first.')
            raise FileNotFoundError
        try:
            output = subprocess.run(['OPTICS.exe','ABER',oe.imgcondbasename_noext],stdout=outputmode,timeout=oe.timeout).stdout
            print(output.decode('utf-8')) if oe.verbose else 0
        except TimeoutExpired:
            print('Optical properties calculation failed. Rerunning.')
            calc_properties_optics(oe)
        except AttributeError:
            print('OpticalElement.imagecondbasename_noext not set. Run '
                  'OpticalElement.write_opt_img_cond_file() first.') 
            raise AttributeError
