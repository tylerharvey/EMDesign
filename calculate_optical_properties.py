#!/usr/bin/env python3
import sys,os,subprocess,shutil,datetime
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from contextlib import contextmanager
from optical_element_io import cd

# takes optical_element object (oe) as argument
def calc_properties_mirror(oe,nterms=50):
    oe.fitname = oe.basename_noext+'.fit'
    with cd(oe.dirname):
        outputmode = subprocess.PIPE if oe.verbose else None
        output = subprocess.run(['herm1.exe',oe.potname,oe.fitname,nterms,n,n],stdout=outputmode).stdout
        print(output.decode('utf-8')) if oe.verbose else 0


def calc_properties_optics(oe):
    with cd(oe.dirname):
        outputmode = subprocess.PIPE if oe.verbose else None
        output = subprocess.run(['OPTICS.exe','ABER',oe.imgcondbasename_noext],stdout=outputmode).stdout
        print(output.decode('utf-8')) if oe.verbose else 0
