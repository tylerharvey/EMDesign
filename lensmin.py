#!/usr/bin/env python
'''
Run as 
$ lensmin.py input_file [log_file]
if log_file is omitted, output is sent to STDOUT
'''
import numpy as np
import matplotlib.pyplot as plt
from optical_element_io import *
from column_io import OpticalColumn
from calculate_optical_properties import calc_properties_optics
from automation import optimize_many_shapes
from misc_library import choose_logger, strtobool
import sys

input_file = open(sys.argv[1],'r')
if(len(sys.argv) > 2):
    choose_logger(sys.argv[2])
else:
    choose_logger(None)

lines = input_file.readlines()
lines = [line.rstrip().split(': ')[1] for line in lines if len(line.split(': '))==2] # remove carriage return and variable text
seed_file = lines[0]
new_filename = lines[1]
opt_img_cond_filename = lines[2]
simplex_filename = lines[3]
z_max = None if lines[4] == 'None' else float(lines[4])
r_min = float(lines[5])
simplex_scale = float(lines[6])
obj_pos=float(lines[7])
img_pos=float(lines[8])
energy=float(lines[9])
enforce_smoothness=strtobool(lines[10])
target_c3=float(lines[11])
obj = StrongMagLens(seed_file)
obj.write(new_filename)
col = OpticalColumn(obj)
col.write_opt_img_cond_file(opt_img_cond_filename,obj_pos=obj_pos,img_pos=img_pos,energy=energy)
obj.calc_field()
calc_properties_optics(col)
col.read_optical_properties()

initial_simplex = None if simplex_filename == 'None' else np.load(simplex_filename)
optimize_many_shapes(obj,col,obj.mag_mat_z_indices+obj.coil_z_indices,obj.mag_mat_r_indices+obj.coil_r_indices,z_min=None,z_max=z_max,r_min=r_min,r_max=None,target_c3=target_c3,enforce_smoothness=enforce_smoothness,simplex_scale=simplex_scale,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':initial_simplex,'return_all':True}) # ,'maxfev':100000 #,method='Nelder-Mead',manual_bounds=True,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None})

