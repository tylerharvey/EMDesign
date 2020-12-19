import numpy as np
import matplotlib.pyplot as plt
from optical_element_io import *
from column_io import OpticalColumn
from calculate_optical_properties import calc_properties_mirror
from automation import optimize_image_plane, optimize_many_shapes
from importlib import reload
import sys


input_file = open(sys.argv[1],'r')
lines = input_file.readlines()
lines = [line.rstrip().split(': ')[1] for line in lines if len(line.split(': '))==2] # remove carriage return and variable text
seed_file = lines[0]
new_filename = lines[1]
mir_img_cond_filename = lines[2]
simplex_filename = lines[3]
z_min = float(lines[4])
z_max = float(lines[5])
voltages = np.fromstring(lines[6],dtype=float,sep=',')
print(lines[7])
flags = lines[7].split(',')
curved = bool(lines[8])
end_r_indices = None if lines[9] == 'None' else np.fromstring(lines[9],dtype=int,sep=',')
automate_curvature=bool(lines[10])
simplex_scale=float(lines[11])
obj = ElecLens(seed_file,verbose=True)
obj.mirror_type(mirror=True,curved_mirror=curved)
obj.write(new_filename)
col = OpticalColumn(obj)
col.write_mir_img_cond_file(mir_img_cond_filename,
                            turning_point=5,
                            potentials=ElecLens.MirPotentials(obj,voltages,flags))
obj.calc_field()
calc_properties_mirror(obj,col)
col.read_mir_optical_properties(raytrace=True)
# col.plot_rays()

initial_simplex = None if simplex_filename == 'None' else np.load(simplex_filename)
optimize_many_shapes(obj,col,obj.dielectric_z_indices+obj.electrode_z_indices,obj.dielectric_r_indices+obj.electrode_r_indices,z_curv_z_indices_list=None,z_curv_r_indices_list=None,end_z_indices_list=[[1]],end_r_indices_list=[end_r_indices],automate_present_curvature=automate_curvature,z_min=z_min,z_max=z_max,r_min=0,r_max=None,simplex_scale=simplex_scale,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':initial_simplex,'return_all':True}) # ,'maxfev':100000 #,method='Nelder-Mead',manual_bounds=True,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None})

print(col.c3)
