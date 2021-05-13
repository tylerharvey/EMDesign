import numpy as np
import os, sys
import logging
from contextlib import contextmanager

# indices is an ordered numpy array of MEBS indices
# like oe.r_indices
# index is a MEBS index
def np_index(indices, index):
    ''' 
    Returns numpy index from a MEBS index based on an ordered list of indices.

    Parameters:
        indices : array
            ordered list of MEBS indices, e.g. oe.r_indices
        index : int
            MEBS index.
    '''
    return len(indices[indices < index])

# indices is an ordered numpy array of MEBS indices
# like oe.r_indices
# index is a MEBS index
def last_np_index(indices, index):
    l = indices[indices <= index]
    return len(l)-1 # if len(l) > 0 else 0 ## shouldn't be necessary now

# takes a list of np indices np_list = [(1,0),(2,5),...]
# and casts to a format that can be used for numpy indexing
# i.e. array[index_array_from_list(index_list)]
def index_array_from_list(index_list):
    if(len(index_list) == 0 or not any(index_list)): #index_list == ([],[])):
        return [],[]
    tmp_array = np.array(index_list)
    return (tmp_array[:,0],tmp_array[:,1])

def check_len(string, colwidth):
    if(len(string.strip()) >= colwidth):
        raise ValueError(f'Error: zero space between columns. Value: {string} with length {len(string)}, while column width is {self.colwidth}. Increase column width and rerun.')
    else:
        return string

# this is not intuitive, and best illustrated with an example: 
# if you have two numerical strings of length 10 and colwidth is 10,
# string.split() does not split, so item will be length 20
# the check works, but not for the reason you might think
def check_len_multi(string, colwidth):
    for item in string.split():
        if(len(item) >= colwidth):
          raise ValueError('Error: zero space between columns. Increase column width and rerun.')
    return string


# safe chdir, found on stackexchange; do not understand it but
# when used as "with cd():" it returns to the previous directory
# no matter what
@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def setup_logger(name, logfile=None, level=logging.INFO):
    if(logfile):
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler(sys.stdout)
    if(level==logging.DEBUG):
        formatter = logging.Formatter('%(module)s:%(funcName)s:%(lineno)d: %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

class Logger:
    def __init__(self,logname,i=-1):
        self.log = logging.getLogger(logname)
        if(i >= 0):
            self.log.info(f'Iteration {i}')

def choose_logger(logfile=None):
    if(logfile):
        logfile_noext,ext = os.path.splitext(logfile) 
        MEBSlogfile = logfile_noext+"_MEBS"+ext
        outputlogfile = logfile_noext+"_output"+ext
        internallogfile = logfile_noext+"_internal"+ext
    else:
        MEBSlogfile = None
        outputlogfile = None
        internallogfile = None

    MEBS = setup_logger('MEBS',MEBSlogfile)
    output = setup_logger('output',outputlogfile)
    internal = setup_logger('internal',internallogfile,level=logging.DEBUG)

# def Mlog(text,i=-1):
#     logger = logging.getLogger('MEBS')
#     if(i >= 0):
#         logger.info(f'Iteration {i}')
#     logger.info(text)
# 
# def olog(text,i=-1):
#     logger = logging.getLogger('output')
#     if(i >= 0):
#         logger.info(f'Iteration {i}')
#     logger.info(text)
# 
# def ilog(text='',vtp=[],i=-1):
#     '''
#     Internal logging.
# 
#     Optional parameters:
#         text : string
#             Any message to print. Default ''.
#         vtp : list
#             A list of variables to print (vtp) in debug format.
#             Default [].
#     '''
#     logger = logging.getLogger('internal')
#     if(i >= 0):
#         logger.info(f'Iteration {i}')
#     if(text and len(vtp) > 0):
#         text += '\n'
#     for variable in vtp:
#         text += f'{variable=} '
#     logger.info(text)

    # def close(self)
    #     self.MEBS.close()
    #     self.output.close()
    #     self.internal.close()
    # def Mprint(self,text,i=-1):
    #     if(i >= 0):
    #         self.MEBS.write(f'Iteration {i}\n')
    #     self.MEBS.write(text+'\n')

    # def oprint(self,text,i=-1):
    #     if(i >= 0):
    #         self.output.write(f'Iteration {i}\n')
    #     self.output.write(text+'\n')

    # def iprint(self,variables,i=-1):
    #     if(i >= 0):
    #         self.iprint.write(f'Iteration {i}\n')
    #     text = ''
    #     for variable in variables:
    #         text += f'{variable=} '
    #     self.iprint.write(text+'\n')

    # def close(self)
    #     self.MEBS.close()
    #     self.output.close()
    #     self.internal.close()


