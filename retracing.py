'''
Run as 
$ retracing.py input_file [log_file]
if log_file is omitted, output is sent to STDOUT
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from optical_element_io import *
from column_io import OpticalColumn
from automation import optimize_planes_for_retracing, optimize_voltages_for_retracing, optimize_broadly_for_retracing
from misc_library import choose_logger, strtobool

input_file = open(sys.argv[1],'r')
if(len(sys.argv) > 2):
    choose_logger(sys.argv[2])
else:
    choose_logger(None)

lines = input_file.readlines()
# remove carriage return and variable text
lines = [line.rstrip().split(': ')[1] for line in lines if len(line.split(': '))==2] 
seed_file = lines[0]
new_filename = lines[1]
mir_img_cond_filename = lines[2]
simplex_filename = lines[3]
voltages = np.fromstring(lines[4],dtype=float,sep=',')
flags = lines[5].split(',')
curved = strtobool(lines[6])
end_z_indices_list = None if lines[7] == 'None' else [[1]]
end_r_indices_list = None if lines[7] == 'None' else np.fromstring(lines[7],dtype=int,sep=',')[np.newaxis,:]
z_curv_r_indices_list = None if lines[8] == 'None' else np.fromstring(lines[8],dtype=int,sep=',')
z_curv_z_indices_list = None if lines[8] == 'None' else [1]*z_curv_r_indices_list.shape[0]
edit_nonend_electrodes = strtobool(lines[9])
simplex_scale=float(lines[10])
voltage_logscale=float(lines[11])
img_pos=float(lines[12])
energy=float(lines[13])
maxfev=float(lines[14])
enforce_smoothness=strtobool(lines[15])
max_r_to_edit =  None if lines[16] == 'None' else float(lines[16])
optimize_end_voltage = strtobool(lines[17])
i=18
other_oe_list = []
while(lines[i] != 'End'): # input file now terminated by End: End
    other_oe_list.append(StrongMagLens(lines[i]))
    curr_oe = other_oe_list[-1]
    i+=1
    curr_oe.lens_pos = float(lines[i])
    i+=1
    curr_oe.lens_excitation = float(lines[i])
    i+=1

mir = ElecLens(seed_file)
mir.mirror_type(mirror=True,curved_mirror=curved)
mir.V = voltages
mir.write(new_filename)
mir.lens_pos = 0
if(other_oe_list):
    oe_list = [mir]+other_oe_list
    col = OpticalColumn(oe_list=oe_list)
else:
    col = OpticalColumn(mir)
col.mircondfilename  = mir_img_cond_filename
z_indices_list = mir.electrode_z_indices if edit_nonend_electrodes else None
r_indices_list = mir.electrode_r_indices if edit_nonend_electrodes else None

initial_simplex = None if simplex_filename == 'None' else np.load(simplex_filename)

# optimize_planes_for_retracing(col,bounds=(65,200),img_pos=90,
#                              potentials=MirPotentials(mir,[-500,6905.87,294749,200000],['f','v1','v2','f']))
# 
# 
# initial_simplex = np.array([[0,0,65],[300000,300000,65],[300000,30000,200],[0,0,200]])
# optimize_voltages_for_retracing(col,potentials=MirPotentials(mir,[-500,6905.87,294749,200000],['f','v1','v2','f']),img_pos=90) #,options={'initial_simplex':initial_simplex}) #bounds=[(-10000,190000),(-10000,300000),(65,200)])


optimize_broadly_for_retracing(mir,col,z_indices_list=z_indices_list,r_indices_list=r_indices_list,potentials=MirPotentials(mir,voltages,flags),img_pos=img_pos,end_z_indices_list=end_z_indices_list,end_r_indices_list=end_r_indices_list,z_curv_z_indices_list=z_curv_z_indices_list,z_curv_r_indices_list=z_curv_r_indices_list,simplex_scale=simplex_scale,voltage_logscale=voltage_logscale,options={'adaptive':True,'fatol':0.00001,'disp':True,'return_all':True,'maxfev':maxfev,'initial_simplex':initial_simplex},enforce_smoothness=enforce_smoothness,energy=energy,max_r_to_edit=max_r_to_edit,optimize_end_voltage=optimize_end_voltage)


