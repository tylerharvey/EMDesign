#!/usr/bin/env python
'''
Run as 
$ lensmin.py input_file > log_file
'''
import numpy as np
import matplotlib.pyplot as plt
from optical_element_io import *
from column_io import OpticalColumn
from calculate_optical_properties import calc_properties_optics
from automation import optimize_single_current, optimize_image_plane, optimize_many_shapes
from importlib import reload
import sys

input_file = open(sys.argv[1],'r')
lines = input_file.readlines()
lines = [line.rstrip() for line in lines]
seed_file = lines[0]
new_filename = lines[1]
opt_img_cond_filename = lines[2]
simplex_filename = lines[3]
z_max = float(lines[4])
r_min = float(lines[5])
obj = StrongMagLens(seed_file)
obj.write(new_filename)
col = OpticalColumn(obj)
col.write_opt_img_cond_file(opt_img_cond_filename)
obj.calc_field()
calc_properties_optics(obj,col)
col.read_optical_properties()

initial_simplex = None if simplex_filename == 'None' else np.load(simplex_filename)
optimize_many_shapes(obj,col,obj.mag_mat_z_indices+obj.coil_z_indices,obj.mag_mat_r_indices+obj.coil_r_indices,z_min=-150,z_max=z_max,r_min=r_min,r_max=150,simplex_scale=10,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':initial_simplex,'return_all':True}) # ,'maxfev':100000 #,method='Nelder-Mead',manual_bounds=True,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None})

print(col.c3)