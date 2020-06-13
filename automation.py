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
    try: 
        oe.read_optical_properties()
    except UnboundLocalError: # if optics fails, return garbage
        return 10
    print(f"f: {oe.f}, C3: {oe.c3}, ratio: {oe.c3/oe.f}")
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

def quad_and_other_edge_points(oe,n_quad):
    quad_edge_points = oe.retrieve_single_quad_edge_points(oe.mag_mat_z_indices[n_quad],oe.mag_mat_r_indices[n_quad])
    all_mag_mat_edge_points = oe.retrieve_edge_points(oe.mag_mat_z_indices,oe.mag_mat_r_indices)
    coil_edge_points = oe.retrieve_edge_points(oe.coil_z_indices,oe.coil_r_indices)
    other_mag_mat_edge_points = [point for point in all_mag_mat_edge_points if point not in quad_edge_points]
    other_edge_points = coil_edge_points + other_mag_mat_edge_points
    tmp_other = np.array(other_edge_points)
    other_edge_points = (tmp_other[:,0],tmp_other[:,1])
    tmp_quad = np.array(quad_edge_points)
    quad_edge_points = (tmp_quad[:,0],tmp_quad[:,1])
    return quad_edge_points,other_edge_points

    
def optimize_single_mag_mat_shape(oe,n_quad,z_min=None,z_max=None,r_min=None,r_max=None,maxiter=100):
    oe.verbose = False
    edge_points,other_edge_points = quad_and_other_edge_points(oe,n_quad)
    initial_shape = np.concatenate((oe.z[edge_points],oe.r[edge_points]))
    other_edge_shape = (oe.z[other_edge_points],oe.r[other_edge_points])
    n = len(oe.z[edge_points])
    bounds = [(z_min,z_max)]*n+[(r_min,r_max)]*n
    result = minimize(change_shape_and_calculate,initial_shape,args=(oe,edge_points,other_edge_shape),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1,'maxiter':maxiter,'disp':True})
    print('Optimization complete with success flag {}'.format(result.success))
    print(result.message)
    change_shape_and_calculate(result.x,oe,edge_points,other_edge_shape)

def optimize_many_shapes(oe,z_indices_list,r_indices_list,z_min=None,z_max=None,r_min=None,r_max=None):
    oe.verbose=False
    initial_shape = np.array([],dtype=float)
    n_quads = len(z_indices_list)
    n_edge_pts = []
    n_mirrored_edge_pts = []
    edge_points_list = []
    mirrored_edge_points_list = []
    for i in range(n_quads):
        edge_points = oe.retrieve_single_quad_edge_points(z_indices_list[i],r_indicies_list[i],return_ind_array=True)
        mirrored_edge_points,edge_points = find_mirrored_edge_points(edge_points)
        n_edge_i = len(oe.z[edge_points])
        edge_points_list.append(edge_points)
        if(mirrored_edge_points != None):
            intital_shape = np.concatenate((initial_shape,oe.z[edge_points],oe.r[edge_points],oe.r[mirrored_edge_points]))
            n_mirrored_edge_i = len(oe.r[mirrored_edge_points])
            n_edge_pts.append(n_edge_i)
            n_mirrored_edge_pts.append(n_mirrored_edge_i)
            bounds += n_edge_i*[(z_min,z_max)]+n_edge_i*[(r_min,r_max)]+n_mirrored_edge_i*[(r_min,r_max)]
        else:
            intital_shape = np.concatenate((initial_shape,oe.z[edge_points],oe.r[edge_points]))
            n_edge_pts.append(n_edge_i)
            n_mirrored_edge_pts.append(0)
            bounds += n_edge_i*[(z_min,z_max)]+n_edge_i*[(r_min,r_max)]
            mirrored_edge_points_list.append(mirrored_edge_points)
    result = minimize(change_n_quads_and_calculate,initial_shape,args=(oe,edge_points_list,mirrored_edge_points_list),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1,'disp':True})
    print('Optimization complete with success flag {}'.format(result.success))
    print(result.message)
    change_n_quads_and_calculate(result.x,oe,edge_points_list,mirrored_edge_points_list)


def find_mirrored_edge_points(oe,edge_points):
    if 0 in oe.z[edge_points]:
        ind = np.nonzero(oe.z[edge_points] == 0)
        mirrored_edge_points = (edge_points[0][ind],edge_points[1][ind])
        edge_points = (np.delete(edge_points[0],ind),np.delete(edge_points[1],ind))
        return mirrored_edge_points,edge_points
    else:
        return None,edge_points

def optimize_image_plane(oe,min_dist=3,image_plane=6):
    oe.verbose = False
    initial_plane = [image_plane] # mm
    bounds = [(min_dist,100)]
    result = minimize(change_imgplane_and_calculate,initial_plane,args=(oe),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1})
    change_imgplane_and_calculate(result.x,oe)
    print('Optimization complete')

def change_imgplane_and_calculate(imgplane,oe):
    oe.write_opt_img_cond_file(oe.imgcondfilename,img_pos=imgplane[0])
    calc_properties_optics(oe)
    try: 
        oe.read_optical_properties()
    except UnboundLocalError: # if optics fails, return garbage
        return 10
    print(f"f: {oe.f}, C3: {oe.c3}")
    return np.abs(oe.c3)

def change_shape_and_calculate(shape,oe,edge_points,other_edges):
    z_shape,r_shape = np.split(shape,2)
    other_edges_z,other_edges_r = other_edges
    # hacky binary constraints
    if(does_quad_intersect_anything(z_shape,r_shape,other_edges_z,other_edges_r)):
        return 10000
    else: 
        pass
    oe.z[edge_points] = z_shape
    oe.r[edge_points] = r_shape
    return calculate_norm_c3(oe)

def change_n_quads_and_calculate(shape,oe,edge_points_list,mirrored_edge_points_list):
    pass
    # if(do_edges_intersect_anything):
    # for i in range(len(edge_points_list)):
    #     oe.z



class point(object):
    def __init__(self,z,r):
        self.z = z
        self.r = r

    def print(self):
        print(f"z: {self.z}, r: {self.r}")
        
def is_quad_self_intersecting(z_shape,r_shape):
    # z_shape,r_shape = np.split(shape,2)
    n_pts = len(z_shape)
    for i in range(n_pts): # iterate segment-by-segment
        p1 = point(z_shape[i-1],r_shape[i-1])
        p2 = point(z_shape[i],r_shape[i])
        # skip immediate next segment that has a shared vertex 
        # shared vertex = no intersection
        # then check all following segments
        # also skip last segment when i-1 = -1
        end_of_range = min(n_pts,n_pts+i-1) 
        for j in range(i+2,end_of_range):
            q1 = point(z_shape[j-1],r_shape[j-1])
            q2 = point(z_shape[j],r_shape[j])
            if(do_segments_intersect(p1,p2,q1,q2) == True):
                return True
            else:
                pass
    return False

def does_quad_intersect_anything(z_shape,r_shape,other_edges_z,other_edges_r):
    n_pts = len(z_shape)
    for i in range(n_pts): # iterate segment-by-segment
        p1 = point(z_shape[i-1],r_shape[i-1])
        p2 = point(z_shape[i],r_shape[i])
        # skip immediate next segment that has a shared vertex 
        # shared vertex = no intersection
        # then check all following segments
        # also skip last segment when i-1 = -1
        end_of_range = min(n_pts,n_pts+i-1) 
        for j in range(i+2,end_of_range):
            q1 = point(z_shape[j-1],r_shape[j-1])
            q2 = point(z_shape[j],r_shape[j])
            if(do_segments_intersect(p1,p2,q1,q2) == True):
                return True
            else:
                pass
        # now chec points on all other quads
        for j in range(len(other_edges_z)):
            q1 = point(other_edges_z[j-1],other_edges_r[j-1])
            q2 = point(other_edges_z[j],other_edges_r[j])
            if(do_segments_intersect(p1,p2,q1,q2) == True):
                return True
            else:
                pass
    return False

# intersection if q1 and q2 are on opposite sides of p1-p2
# and if p1 and p2 are on opposite sides of q1-q2
# based on www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def do_segments_intersect(p1,p2,q1,q2):
    return (cross_product_sign(p1,p2,q1) != cross_product_sign(p1,p2,q2) and cross_product_sign(q1,q2,p1) != cross_product_sign(q1,q2,p2))

# cross product of p1->p2 and p1->p3
def cross_product_sign(p1,p2,p3):
    return np.sign((p2.z-p1.z)*(p3.r-p1.r) - (p3.z-p1.z)*(p2.r-p1.r))

# outdated function
# def optimize_mag_mat_shape(oe,z_min=None,z_max=None,r_min=None,r_max=None):
#     oe.verbose = False
#     edge_points = oe.retrieve_edge_points(oe.mag_mat_z_indices,oe.mag_mat_r_indices,return_ind_array=True)
#     initial_shape = np.concatenate((oe.z[edge_points],oe.r[edge_points]))
#     n = len(oe.z[edge_points])
#     bounds = [(z_min,z_max)]*n+[(r_min,r_max)]*n
#     result = minimize(change_shape_and_calculate,initial_shape,args=(oe,edge_points),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1,'disp':True})
#     print('Optimization complete with success flag {}'.format(result.success))
#     print(result.message)
#     change_shape_and_calculate(result.x,oe,edge_points)
#     # oe.write(oe.filename)

# leftover
    # mag_mat_edge_points = oe.retrieve_edge_points(oe.mag_mat_z_indices,oe.mag_mat_r_indices,return_ind_array=True)
    # coil_edge_points = oe.retrieve_edge_points(oe.coil_z_indices,oe.coil_r_indices,return_ind_array=True)
    # mag_mat_mirror_edge_points,mag_mat_edge_points = find_mirror_edge_points(mag_mat_edge_points)
    # coil_mirror_edge_points,coil_edge_points = find_mirror_edge_points(coil_edge_points)
    # initial_shape = np.concatenate((oe.z[mag_mat_edge_points],oe.r[mag_mat_edge_points],oe.r[mag_mat_mirror_edge_points],oe.z[coil_edge_points],oe.r[coil_edge_points],oe.r[mag_mat_coil_edge_points))
    # n_mag_mat = len(oe.z[mag_mat_edge_points])
    # n_mag_mat_mirror = len(oe.r[mag_mat_mirror_edge_points])
    # n_coil = len(oe.z[coil_edge_points])
    # n_coil_mirror = len(oe.r[coil_mirror_edge_points])
    # # these could be made more individual
    # bounds = [(z_min,z_max)]*n_mag_mat+[(r_min,r_max)]*(n_mag_mat+n_mag_mat_mirror)+[(z_min,z_max)]*n_coil+[(r_min,r_max)]*(n_coil+n_coil_mirror)
    # result = minimize(change_whole_shape_and_calculate,initial_shape,args=(oe,edge_points),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1,'disp':True})
    # print('Optimization complete with success flag {}'.format(result.success))
    # print(result.message)
    # change_shape_and_calculate(result.x,oe,edge_points)
    # # oe.write(oe.filename)
