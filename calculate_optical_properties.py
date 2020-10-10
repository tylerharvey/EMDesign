#!/usr/bin/env python3
import sys,os,subprocess,shutil,datetime
from subprocess import TimeoutExpired
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from contextlib import contextmanager
from optical_element_io import cd
import asyncio
import nest_asyncio
nest_asyncio.apply()

async def run_async(command_and_args,i=0,max_attempts=3,timeout=1000,user_input=None,verbose=True):
    '''
    Function for handling simple subprocess executions.

    Parameters:
        command_and_args : list
            list of command and all arguments as strings, e.g. ['ls','-l'].

    Optional parameters:
        timeout : float
            time in seconds to wait for the program called.
            Passed in from oe.timeout.
        user_input : str
            Any user input to send to the first interactive prompt generated by
            the called program. Default None.
    '''

    outputmode = asyncio.subprocess.PIPE if verbose else None
    proc = await asyncio.create_subprocess_exec(*command_and_args,
                                    stdout=outputmode,
                                    stderr=asyncio.subprocess.PIPE,
                                    stdin=asyncio.subprocess.PIPE)

    if(user_input):
        stdin = user_input.encode()
        stdout,stderr = await asyncio.wait_for(proc.communicate(stdin),timeout=timeout)
        if(stdout):
            print(f'{stdout.decode()}')


    try:
        stdout,stderr = await asyncio.wait_for(proc.communicate(),timeout=timeout)
    except asyncio.TimeoutError:
        i+=1
        proc.kill()
        if(i > max_attempts):
            print('Maximum attempts reached.')
            raise asyncio.TimeoutError
        else:
            print(f'Program {command_and_args[0]} timed out. Rerunning.')
            try: 
                await run_async(command_and_args,i+1,timeout=timeout,user_input=user_input)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError

    if(stdout):
        print(f'{stdout.decode()}')
    # MEBS doesn't generally use STDERR

async def run_herm_then_mirror(oe,nterms):
    symstring = 'AN' if oe.mirror else 'NN'
    if(oe.mirror):
        user_input = 'Y\n' if oe.curved_mirror else 'N\n'
    else:
        user_input = None
        
    try:
        await run_async(['herm1.exe',oe.potname,oe.fitname,str(nterms),symstring],timeout=oe.timeout,
                                                         user_input=user_input,verbose=oe.verbose)

        await run_async(['MIRROR.exe',oe.mircondbasename_noext],timeout=oe.timeout,verbose=oe.verbose)
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError

# takes optical_element object (oe) as argument
def calc_properties_mirror(oe,nterms=50):
    '''
    Function for calculating optical properties of electrostatic
    mirrors with MIRROR. Implemented for only a single optical
    element at the moment.

    Parameters:
        oe : OpticalElement object 
            optical element for which to calculate optical properties.

    Optional parameters:
        nterms : int
            number of terms to use in hermite series to approximate field.
            default 50
        mirror : bool
            Indicates whether optical element includes a mirror. Default True.
        curved_mirror : bool
            Indicates whether optical element includes a curved mirror.
            Default False.
    '''
    with cd(oe.dirname):
        try:
            asyncio.run(run_herm_then_mirror(oe,nterms))
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError
        # output = subprocess.run(['herm1.exe',oe.potname,oe.fitname,str(nterms),symstring],stdout=outputmode,timeout=oe.timeout).stdout
        # print(output.decode('utf-8')) if oe.verbose else 0
        # output = subprocess.run(['MIRROR.exe',oe.mircondbasename_noext],stdout=outputmode,timeout=oe.timeout).stdout
        # print(output.decode('utf-8')) if oe.verbose else 0

def calc_properties_optics(oe,i=0,max_attempts=3):
    '''
    function for calculating optical properties of electric or magnetic lenses
    with OPTICS ABER5. 

    Parameters:
        oe : OpticalElement object 
            optical element for which to calculate optical properties.
    '''
    with cd(oe.dirname):
        outputmode = subprocess.PIPE if oe.verbose else None
        if(os.path.exists(oe.imgcondfilename) != True):
            print('No optical imaging conditions file found. Run OpticalElement.write_opt_img_cond_file() first.')
            raise FileNotFoundError
        try:
            output = subprocess.run(['OPTICS.exe','ABER',oe.imgcondbasename_noext],stdout=outputmode,timeout=oe.timeout).stdout
            print(output.decode('utf-8')) if oe.verbose else 0
        except TimeoutExpired:
            i+=1
            print('Optical properties calculation failed. Rerunning.')
            if(i > max_attempts):
                raise TimeoutExpired
            else:
                calc_properties_optics(oe,i)
        except AttributeError:
            print('OpticalElement.imagecondbasename_noext not set. Run '
                  'OpticalElement.write_opt_img_cond_file() first.') 
            raise AttributeError
