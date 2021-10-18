'''
Run as 
$ mirmin.py input_file [log_file]
if log_file is omitted, output is sent to STDOUT
'''
import numpy as np
import matplotlib.pyplot as plt
from optical_element_io import *
from column_io import OpticalColumn
from calculate_optical_properties import calc_properties_mirror
from automation import optimize_many_shapes
from misc_library import choose_logger
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
mir_img_cond_filename = lines[2]
simplex_filename = lines[3]
z_min = float(lines[4])
z_max = float(lines[5])
r_min = float(lines[6])
voltages = np.fromstring(lines[7],dtype=float,sep=',')
flags = lines[8].split(',')
curved = bool(lines[9])
end_z_indices_list = None if lines[10] == 'None' else [[1]]
end_r_indices_list = None if lines[10] == 'None' else np.fromstring(lines[10],dtype=int,sep=',')[np.newaxis,:]
automate_curvature=bool(lines[11])
simplex_scale=float(lines[12])
source_pos=float(lines[13])
img_pos=float(lines[14])
energy=float(lines[15])
enforce_smoothness=bool(lines[16])
mir = ElecLens(seed_file)
mir.mirror_type(mirror=True,curved_mirror=curved)
mir.write(new_filename)
col = OpticalColumn(mir)
col.write_mir_img_cond_file(mir_img_cond_filename, turning_point=5, energy=energy,
                            source_pos=source_pos, img_pos=img_pos,
                            potentials=MirPotentials(mir,voltages,flags))
# saved for later use
mir.initial_V = col.potentials.voltages
mir.V = voltages
mir.calc_field()
calc_properties_mirror(mir,col)
col.read_mir_optical_properties(raytrace=True)
# col.plot_rays()

initial_simplex = None if simplex_filename == 'None' else np.load(simplex_filename)
optimize_many_shapes(mir,col,mir.electrode_z_indices,mir.electrode_r_indices,z_curv_z_indices_list=None,z_curv_r_indices_list=None,end_z_indices_list=end_z_indices_list,end_r_indices_list=end_r_indices_list,automate_present_curvature=automate_curvature,enforce_smoothness=enforce_smoothness,z_min=z_min,z_max=z_max,r_min=r_min,r_max=None,simplex_scale=simplex_scale,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':initial_simplex,'return_all':True}) # ,'maxfev':100000 #,method='Nelder-Mead',manual_bounds=True,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None})
# optimize_many_shapes(mir,col,mir.dielectric_z_indices+mir.electrode_z_indices,mir.dielectric_r_indices+mir.electrode_r_indices,z_curv_z_indices_list=None,z_curv_r_indices_list=None,end_z_indices_list=end_z_indices_list,end_r_indices_list=end_r_indices_list,automate_present_curvature=automate_curvature,enforce_smoothness=enforce_smoothness,z_min=z_min,z_max=z_max,r_min=r_min,r_max=None,simplex_scale=simplex_scale,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':initial_simplex,'return_all':True}) # ,'maxfev':100000 #,method='Nelder-Mead',manual_bounds=True,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None})
