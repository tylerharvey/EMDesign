'''
Run as 
$ retracing.py input_file [log_file]
if log_file is omitted, output is sent to STDOUT
'''
import numpy as np
import matplotlib.pyplot as plt
from optical_element_io import *
from column_io import OpticalColumn
from automation import optimize_planes_for_retracing, optimize_voltages_for_retracing, optimize_broadly_for_retracing
from misc_library import choose_logger

input_file = open(sys.argv[1],'r')
if(len(sys.argv) > 2):
    choose_logger(sys.argv[2])
else:
    choose_logger(None)

lines = input_file.readlines()
print(lines)
# remove carriage return and variable text
lines = [line.rstrip().split(': ')[1] for line in lines if len(line.split(': '))==2] 
seed_file = lines[0]
new_filename = lines[1]
mir_img_cond_filename = lines[2]
simplex_filename = lines[3]
voltages = np.fromstring(lines[4],dtype=float,sep=',')
flags = lines[5].split(',')
curved = bool(lines[6])
end_z_indices_list = None if lines[7] == 'None' else [[1]]
end_r_indices_list = None if lines[7] == 'None' else np.fromstring(lines[7],dtype=int,sep=',')[np.newaxis,:]
z_curv_r_indices_list = None if lines[8] == 'None' else np.fromstring(lines[8],dtype=int,sep=',')
z_curv_z_indices_list = None if lines[8] == 'None' else [1]*z_curv_r_indices_list.shape[0]
simplex_scale=float(lines[9])
voltage_logscale=float(lines[10])
img_pos=float(lines[11])
energy=float(lines[12])
maxfev=float(lines[13])

mir = ElecLens(seed_file)
mir.mirror_type(mirror=True,curved_mirror=curved)
mir.write(new_filename)
col = OpticalColumn(mir)
col.mircondfilename  = mir_img_cond_filename

initial_simplex = None if simplex_filename == 'None' else np.load(simplex_filename)

# optimize_planes_for_retracing(col,bounds=(65,200),img_pos=90,
#                              potentials=MirPotentials(mir,[-500,6905.87,294749,200000],['f','v1','v2','f']))
# 
# 
# initial_simplex = np.array([[0,0,65],[300000,300000,65],[300000,30000,200],[0,0,200]])
# optimize_voltages_for_retracing(col,potentials=MirPotentials(mir,[-500,6905.87,294749,200000],['f','v1','v2','f']),img_pos=90) #,options={'initial_simplex':initial_simplex}) #bounds=[(-10000,190000),(-10000,300000),(65,200)])


optimize_broadly_for_retracing(mir,col,potentials=MirPotentials(mir,voltages,flags),img_pos=img_pos,end_z_indices_list=end_z_indices_list,end_r_indices_list=end_r_indices_list,z_curv_z_indices_list=z_curv_z_indices_list,z_curv_r_indices_list=z_curv_r_indices_list,simplex_scale=simplex_scale,voltage_logscale=voltage_logscale,options={'adaptive':True,'fatol':0.00001,'disp':True,'return_all':True,'maxfev':maxfev,'initial_simplex':initial_simplex},energy=energy)

