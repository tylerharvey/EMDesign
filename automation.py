#!/usr/bin/env python3
import sys,os,subprocess,shutil,datetime
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from contextlib import contextmanager
from optical_element_io import cd
from calculate_optical_properties import calc_properties_optics
from scipy.optimize import minimize

def calculate_norm_c3(oe):
    oe.write(oe.filename)
    oe.calc_field()
    calc_properties_optics(oe)
    oe.read_optical_properties()
    # print(f"curr: {oe.coil_curr[0]}, f: {oe.f}, C3: {oe.c3}, ratio: {oe.c3/oe.f}")
    print(f"f: {oe.f}, C3: {oe.c3}, ratio: {oe.c3/oe.f}")
    # oe.plot_field()
    return np.abs(oe.c3)

def change_current_and_calculate(current,oe):
    oe.coil_curr = current
    return calculate_norm_c3(oe)

# for a single coil such that oe.coil_curr = [current]
def optimize_single_current(oe):
    oe.verbose = False # limits the noise
    result = minimize(change_current_and_calculate,oe.coil_curr,args=(oe),method='Nelder-Mead')
    oe.coil_curr = result.x
    oe.write(oe.filename)
    print('Optimization complete')

def optimize_single_mag_mat_shape(oe,n_quad,z_min=None,z_max=None,r_min=None,r_max=None):
    oe.verbose = False
    edge_points = oe.retrieve_edge_points(oe.mag_mat_z_indices[n_quad],oe.mag_mat_r_indices[n_quad],
                                          return_ind_array=True)
    initial_shape = np.concatenate((oe.z[edge_points],oe.r[edge_points]))
    n = len(oe.z[edge_points])
    bounds = [(z_min,z_max)]*n+[(r_min,r_max)]*n
    result = minimize(change_shape_and_calculate,initial_shape,args=(oe,edge_points),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1})
    change_shape_and_calculate(result.x,oe,edge_points)
    # oe.write(oe.filename)
    print('Optimization complete')

def optimize_mag_mat_shape(oe,z_min=None,z_max=None,r_min=None,r_max=None):
    oe.verbose = False
    edge_points = oe.retrieve_edge_points(oe.mag_mat_z_indices,oe.mag_mat_r_indices,return_ind_array=True)
    initial_shape = np.concatenate((oe.z[edge_points],oe.r[edge_points]))
    n = len(oe.z[edge_points])
    bounds = [(z_min,z_max)]*n+[(r_min,r_max)]*n
    result = minimize(change_shape_and_calculate,initial_shape,args=(oe,edge_points),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1})
    change_shape_and_calculate(result.x,oe,edge_points)
    # oe.write(oe.filename)
    print('Optimization complete')

def optimize_image_plane(oe,image_plane=6):
    oe.verbose = False
    initial_plane = [image_plane] # mm
    bounds = (0,100)
    result = minimize(change_imgplane_and_calculate,initial_plane,args=(oe),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1})
    change_imgplane_and_calculate(result.x,oe)
    print('Optimization complete')

def change_imgplane_and_calculate(imgplane,oe):
    oe.write_opt_img_cond_file(oe.imgcondfilename,img_pos=imgplane)
    calc_properties_optics(oe)
    oe.read_optical_properties()
    print(f"f: {oe.f}, C3: {oe.c3}, ratio: {oe.c3/oe.f}")
    # oe.plot_field()
    return np.abs(oe.c3)

def change_shape_and_calculate(shape,oe,edge_points):
    z_shape,r_shape = np.split(shape,2)
    oe.z[edge_points] = z_shape
    oe.r[edge_points] = r_shape
    return calculate_norm_c3(oe)

# def is_quad_self_intersecting(shape):
#     z_shape,r_shape = np.split(shape,2)
#     for 

