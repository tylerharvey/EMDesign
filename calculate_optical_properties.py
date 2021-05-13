#!/usr/bin/env python3
import os, subprocess
from subprocess import TimeoutExpired
import numpy as np
import matplotlib.pyplot as plt
from misc_library import Logger, cd
import asyncio
import nest_asyncio
nest_asyncio.apply()

async def run_async(command_and_args,i=0,max_attempts=3,timeout=1000,user_input=None):
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

    outputmode = asyncio.subprocess.PIPE 
    proc = await asyncio.create_subprocess_exec(*command_and_args,
                                    stdout=outputmode,
                                    stderr=asyncio.subprocess.PIPE,
                                    stdin=asyncio.subprocess.PIPE)

    Mlog = Logger('MEBS')
    if(user_input):
        stdin = user_input.encode()
        stdout,stderr = await asyncio.wait_for(proc.communicate(stdin),timeout=timeout)
        if(stdout):
            Mlog.log.debug(f'{stdout.decode()}')


    try:
        stdout,stderr = await asyncio.wait_for(proc.communicate(),timeout=timeout)
        if(stdout):
            Mlog.log.debug(f'{stdout.decode()}')
        # MEBS doesn't generally use STDERR
    except asyncio.TimeoutError:
        i+=1
        proc.kill()
        olog = Logger('output')
        if(i > max_attempts):
            olog.log.critical('Maximum attempts reached.')
            raise asyncio.TimeoutError
        else:
            olog.log.error(f'Program {command_and_args[0]} timed out. Rerunning.')
            try: 
                await run_async(command_and_args,i+1,timeout=timeout,user_input=user_input)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError


async def run_herm_then_mirror(oe,col,nterms):
    symstring = 'AN' if oe.mirror else 'NN'
    if(oe.mirror):
        user_input = 'Y\n' if oe.curved_mirror else 'N\n'
    else:
        user_input = None
        
    try:
        await run_async(['herm1.exe',oe.potname,oe.fitname,str(nterms),symstring],timeout=oe.timeout,
                                                         user_input=user_input)

        await run_async(['MIRROR.exe',col.mircondbasename_noext],timeout=col.timeout)
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError

async def run_herm_then_mirror_multi(col,nterms):
    for oe in col.oe_list:
        symstring = 'AN' if oe.mirror else 'NN'
        if(oe.mirror):
            user_input = 'Y\n' if oe.curved_mirror else 'N\n'
        else:
            user_input = None
            
        try:
            await run_async(['herm1.exe',oe.potname,oe.fitname,str(nterms),symstring],timeout=oe.timeout,
                                                                user_input=user_input)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError

    try:
        await run_async(['MIRROR.exe',col.mircondbasename_noext],timeout=col.timeout)
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError

# takes optical_element object (oe) as argument
def calc_properties_mirror(oe,col,nterms=50):
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
            asyncio.run(run_herm_then_mirror(oe,col,nterms))
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError

def calc_properties_mirror_multi(col,nterms=50):
    '''
    Function for calculating optical properties of optical columns
    MIRROR. Implemented for a multi-element column. Also compatible
    with single-element columns.

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
    with cd(col.dirname):
        try:
            asyncio.run(run_herm_then_mirror_multi(col,nterms))
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError

def calc_properties_optics(col,i=0,max_attempts=3):
    '''
    function for calculating optical properties of electric or magnetic lenses
    with OPTICS ABER5. 

    Parameters:
        col : OpticalColumn object 
            optical column for which to calculate optical properties.
    '''
    olog = Logger('output')
    Mlog = Logger('MEBS')
    with cd(col.dirname):
        outputmode = subprocess.PIPE 
        if(os.path.exists(col.imgcondfilename) != True):
            olog.log.critical('No optical imaging conditions file found. '
                           'Run OpticalElement.write_opt_img_cond_file() first.')
            raise FileNotFoundError
        try:
            output = subprocess.run(['OPTICS.exe','ABER',col.imgcondbasename_noext],stdout=outputmode,timeout=col.timeout).stdout
            Mlog.log.debug(output.decode('utf-8'))
        except TimeoutExpired:
            i+=1
            olog.log.error('Optical properties calculation failed. Rerunning.')
            if(i > max_attempts):
                raise TimeoutExpired
            else:
                calc_properties_optics(col,i)
        except AttributeError:
            olog.log.critical('OpticalElement.imagecondbasename_noext not set. Run '
                  'OpticalElement.write_opt_img_cond_file() first.') 
            raise AttributeError
