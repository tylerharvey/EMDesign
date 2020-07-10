#!/usr/bin/env python3
'''
User methods:
    optimize_many_shapes
    OptimizeShapes (built as a class for obscure reasons)
'''
import sys,os,subprocess,shutil,datetime
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from contextlib import contextmanager
from optical_element_io import cd,index_array_from_list,Point
from calculate_optical_properties import calc_properties_optics
from scipy.optimize import minimize
from skopt import gbrt_minimize, gp_minimize, dummy_minimize, forest_minimize

def calculate_c3(oe):
    '''
    Workhorse function for automation. Writes optical element file, then 
    calculates field, then calculates optical properties, and then reads them.

    Parameters:
        oe : OpticalElement object
            optical element on which to calculate spherical aberration.
    '''
    
    oe.write(oe.filename)
    oe.calc_field()
    calc_properties_optics(oe)
    try: 
        oe.read_optical_properties()
    except UnboundLocalError: # if optics fails, return garbage
        return 100
    print(f"f: {oe.f}, C3: {oe.c3}")
    return np.abs(oe.c3)

def change_current_and_calculate(current,oe):
    oe.coil_curr = current
    return calculate_c3(oe)

# for a single coil such that oe.coil_curr = [current]
def optimize_single_current(oe):
    '''
    In principle, finds current of one coil necessary to minimize spherical
    aberration. In practice, has multiple problems:
    -MEBS throws popups when not running with a defined image plane
    -higher current will probably always be better

    Not currently useful but could be made so with changes.
    '''
    oe.verbose = False # limits the noise
    result = minimize(change_current_and_calculate,oe.coil_curr,args=(oe),method='Nelder-Mead')
    oe.coil_curr = result.x
    oe.write(oe.filename)
    print('Optimization complete')

class OptimizeShapes:
    '''
    This class is necessary to use skopt minimizers as they don't allow 
    additional arguments to be passed to the objective function like scipy.
    Also, skopt uses a list and cannot use a numpy array for the parameters
    to optimize.

    Copy of optimize_many_shapes with slight tweaks.
    '''
    minimize_switch = {'gbrt': gbrt_minimize,'gp': gp_minimize, 'forest': forest_minimize, 'dummy': dummy_minimize}

    def __init__(self,oe,z_indices_list,r_indices_list,other_z_indices_list=[],other_r_indices_list=[],z_min=None,z_max=None,r_min=None,r_max=None,method='gbrt',c3=None,n_random_starts=10,n_calls=100):
        '''
        Parameters:
            oe: OpticalElement object
                optical element to optimize
            r_indices_list : list
            z_indices_list : list
                list of lists of two MEBS r indices and two MEBS z indices that 
                defines quads to optimize
            other_z_indices_list : list
            other_r_indices_list : list
                list of lists of indices for all other quads in optical element.
                used to avoid intersecting lines.
                default empty list
            z_min : float
            z_max : float
            r_min : float
            r_max : float
                bounds
                default None
            method : str
                name of skopt method to use ('gbrt','forest','dummy','gp')
                default 'gbrt'
        '''
        oe.verbose=False
        self.oe = oe
        self.quads,all_edge_points_list,all_mirrored_edge_points_list = define_edges(oe,z_indices_list,r_indices_list)
        self.other_quads,_,_ = define_edges(oe,other_z_indices_list,other_r_indices_list,remove_duplicates_and_mirrored=False)
        self.n_edge_pts = len(all_edge_points_list)
        n_mirrored_edge_pts = len(all_mirrored_edge_points_list)
        edge_points = index_array_from_list(all_edge_points_list)
        mirrored_edge_points = index_array_from_list(all_mirrored_edge_points_list)
        initial_shape = np.concatenate((oe.z[edge_points],oe.r[edge_points],oe.r[mirrored_edge_points])).tolist()
        bounds = self.n_edge_pts*[(z_min,z_max)]+(self.n_edge_pts+n_mirrored_edge_pts)*[(r_min,r_max)]
        result = self.minimize_switch.get(method,dummy_minimize)(self.change_n_quads_and_calculate,x0=initial_shape,dimensions=bounds,n_random_starts=n_random_starts,n_calls=n_calls,y0=c3)
        print('Optimization complete.')
        self.change_n_quads_and_calculate(result.x)

    def change_n_quads_and_calculate(self,shape):
        return change_n_quads_and_calculate(np.array(shape),self.oe,self.quads,self.other_quads,self.n_edge_pts)
        
    
def optimize_many_shapes(oe,z_indices_list,r_indices_list,other_z_indices_list=[],other_r_indices_list=[],z_min=None,z_max=None,r_min=None,r_max=None,method='Nelder-Mead',manual_bounds=True,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None,'return_all':True},simplex_scale=5):
    '''
    Automated optimization of the shape of one or more quads with 
    scipy.optimize.minimize.

    Parameters:
        oe: OpticalElement object
            optical element to optimize
        r_indices_list : list
        z_indices_list : list
            list of lists of two MEBS r indices and two MEBS z indices that 
            defines quads to optimize
        other_z_indices_list : list
        other_r_indices_list : list
            list of lists of indices for all other quads in optical element.
            used to avoid intersecting lines.
            default empty list
        z_min : float
        z_max : float
        r_min : float
        r_max : float
            bounds
            default None
        manual_bounds : boolean
            determines whether bounds will be enforced manually in objective
            function. set to False for methods like TNC that include bounds.
            set to True for Nelder-Mead, Powell, etc.
            intersections are always manually blocked.
            default True
        method : str
            name of method to use
            default 'Nelder-Mead'
        options : dict
            options for the specific solver.
            for TNC, a good set is {'eps':0.5,'stepmx':5,'minfev':1,'disp':True}
            default {'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,
                     'initial_simplex':None} for Nelder-Mead
        simplex_scale : float
            size (in mm) for normal distribution of simplex points around
            initial shape. only used with Nelder-Mead. A larger scale results
            in a longer search that is more likely to find a qualitatively
            different shape.
            default 5
    '''
    oe.verbose=False
    quads,all_edge_points_list,all_mirrored_edge_points_list = define_edges(oe,z_indices_list,r_indices_list)
    other_quads,_,_ = define_edges(oe,other_z_indices_list,other_r_indices_list,remove_duplicates_and_mirrored=False)
    n_edge_pts = len(all_edge_points_list)
    n_mirrored_edge_pts = len(all_mirrored_edge_points_list)
    edge_points = index_array_from_list(all_edge_points_list)
    mirrored_edge_points = index_array_from_list(all_mirrored_edge_points_list)
    initial_shape = np.concatenate((oe.z[edge_points],oe.r[edge_points],oe.r[mirrored_edge_points]))
    bounds = n_edge_pts*[(z_min,z_max)]+(n_edge_pts+n_mirrored_edge_pts)*[(r_min,r_max)]
    if(method=='Nelder-Mead' and options.get('initial_simplex') == None):
        print('Generating initial simplex.')
        options['initial_simplex'] = generate_initial_simplex(initial_shape,oe,quads,other_quads,n_edge_pts,enforce_bounds=True,bounds=np.array(bounds),scale=simplex_scale)
        print('Finished initial simplex generation.')
    if(manual_bounds):
        result = minimize(change_n_quads_and_calculate,initial_shape,args=(oe,quads,other_quads,n_edge_pts,True,np.array(bounds)),method=method,options=options)
    else:
        result = minimize(change_n_quads_and_calculate,initial_shape,args=(oe,quads,other_quads,n_edge_pts),bounds=bounds,method=method,options=options)
    print('Optimization complete with success flag {}'.format(result.success))
    print(result.message)
    change_n_quads_and_calculate(result.x,oe,quads,other_quads,n_edge_pts)
    if(method=='Nelder-Mead' and options.get('return_all') == True):
        np.save(oe.filename_noext+'_all_solns',result['allvecs'])

def optimize_image_plane(oe,min_dist=3,image_plane=6):
    '''
    In principle, could be used to find the optimal image plane to minimize
    spherical aberration. In practice, that is always zero distance from the
    object plane, so needs to be rewritten to be useful.
    '''
    oe.verbose = False
    initial_plane = [image_plane] # mm
    bounds = [(min_dist,100)]
    result = minimize(change_imgplane_and_calculate,initial_plane,args=(oe),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1})
    change_imgplane_and_calculate(result.x,oe)
    print('Optimization complete')

class Quad:
    '''
    Class to carry around quad information for optimization.
    '''

    def __init__(self,oe,z_indices,r_indices,separate_mirrored=True):
        self.edge_points_list = oe.retrieve_single_quad_edge_points(z_indices,r_indices)
        self.original_edge_points_list = self.edge_points_list.copy()
        if(separate_mirrored):
            self.mirrored_edge_points_list,self.edge_points_list = find_mirrored_edge_points(oe,self.edge_points_list)
        else:
            self.mirrored_edge_points_list = [],[]

    def delete_overlaps(self,edge_points_list,prior_edge_points_list):
        edge_points_list = [point for point in edge_points_list if point not in prior_edge_points_list]

    def count(self):
        self.n_edge_pts = len(self.edge_points_list)
        self.n_mirrored_edge_pts = len(self.mirrored_edge_points_list)

    def make_index_arrays(self):
        self.edge_points = index_array_from_list(self.edge_points_list)
        self.original_edge_points = index_array_from_list(self.original_edge_points_list)
        self.mirrored_edge_points = index_array_from_list(self.mirrored_edge_points_list)
        # if((len(self.mirrored_edge_points_list) > 0) and (self.mirrored_edge_points_list != ([],[])) and (self.mirrored_edge_points_list != [],[])):
        #     self.mirrored_edge_points = index_array_from_list(self.mirrored_edge_points_list)
        # else:
        #     self.mirrored_edge_points = [],[]


# takes a list of quads definedb by two z indices and two r indices
# and makes a list of Quad objects
# remove_duplicates_and_mirrored is set to true when the resulting quads 
# include points that will be changed by optimization
# it is set to false for other quads outside the optimization
# that are only defined here for checking intersections
def define_edges(oe,z_indices_list,r_indices_list,remove_duplicates_and_mirrored=True):
    n_quads = len(z_indices_list)
    quads = []
    all_edge_points_list = []
    all_mirrored_edge_points_list = []
    for i in range(n_quads):
        quads.append(Quad(oe,z_indices_list[i],r_indices_list[i],separate_mirrored=remove_duplicates_and_mirrored))
        if(remove_duplicates_and_mirrored):
            quads[-1].delete_overlaps(quads[-1].edge_points_list,all_edge_points_list)
            quads[-1].delete_overlaps(quads[-1].mirrored_edge_points_list,all_mirrored_edge_points_list)
        quads[-1].count()
        all_edge_points_list += quads[-1].edge_points_list
        all_mirrored_edge_points_list += quads[-1].mirrored_edge_points_list
        quads[-1].make_index_arrays()
    return quads,all_edge_points_list,all_mirrored_edge_points_list

def generate_initial_simplex(initial_shape,oe,quads,other_quads,n_edge_pts,enforce_bounds=True,bounds=None,scale=5):
    rng = np.random.default_rng()
    N = len(initial_shape)
    simplex = np.zeros((N+1,N),dtype=float)
    for i in range(N+1):
        simplex[i] = rng.normal(initial_shape,scale,N)
        # keep trying until simplex point is valid
        # inefficient but simple
        while(change_n_quads_and_check(simplex[i],oe,quads,other_quads,n_edge_pts,enforce_bounds,bounds)):
            simplex[i] = rng.normal(initial_shape,scale,N)
    # save result
    np.save(os.path.join(oe.dirname,'initial_simplex_for_'+oe.basename_noext),simplex)
    # return shape to initial shape
    change_n_quads_and_check(initial_shape,oe,quads,other_quads,n_edge_pts)
    return simplex

def change_n_quads_and_check(shape,oe,quads,other_quads,n_edge_pts,enforce_bounds=False,bounds=None):
    z_shapes,r_shapes,mirrored_r_shapes = np.split(shape,[n_edge_pts,2*n_edge_pts])
    for quad in quads:
        oe.z[quad.edge_points],z_shapes = np.split(z_shapes,[quad.n_edge_pts])
        oe.r[quad.edge_points],r_shapes = np.split(r_shapes,[quad.n_edge_pts])
        oe.r[quad.mirrored_edge_points],mirrored_r_shapes = np.split(mirrored_r_shapes,[quad.n_mirrored_edge_pts])
    if(enforce_bounds):
        if((bounds[:,0] > shape).any() or (bounds[:,1] < shape).any()):
            return True
    if(does_fine_mesh_intersect_coarse(oe)):
        return True
    # if(do_quads_intersect_anything(oe,quads,other_quads)):
    #     return True
    return False

def find_mirrored_edge_points(oe,edge_points_list):
    mirrored_edge_points_list = [point for point in edge_points_list if oe.z[point] == 0]
    edge_points_list = [point for point in edge_points_list if oe.z[point] != 0]
    return mirrored_edge_points_list,edge_points_list

def change_imgplane_and_calculate(imgplane,oe):
    oe.write_opt_img_cond_file(oe.imgcondfilename,img_pos=imgplane[0])
    calc_properties_optics(oe)
    try: 
        oe.read_optical_properties()
    except UnboundLocalError: # if optics fails, return garbage
        return 100
    print(f"f: {oe.f}, C3: {oe.c3}")
    return np.abs(oe.c3)

def change_n_quads_and_calculate(shape,oe,quads,other_quads,n_edge_pts,enforce_bounds=False,bounds=None):
    # z_shapes,r_shapes,mirrored_r_shapes = np.split(shape,[n_edge_pts,2*n_edge_pts])
    # for quad in quads:
    #     oe.z[quad.edge_points],z_shapes = np.split(z_shapes,[quad.n_edge_pts])
    #     oe.r[quad.edge_points],r_shapes = np.split(r_shapes,[quad.n_edge_pts])
    #     oe.r[quad.mirrored_edge_points],mirrored_r_shapes = np.split(mirrored_r_shapes,[quad.n_mirrored_edge_pts])
    # if(enforce_bounds):
    #     if((bounds[:,0] > shape).any() or (bounds[:,1] < shape).any()):
    #         return 10000
    # if(do_quads_intersect_anything(oe,quads,other_quads)):
    #     return 10000
    if(change_n_quads_and_check(shape,oe,quads,other_quads,n_edge_pts,enforce_bounds=enforce_bounds,bounds=bounds)):
        return 10000
    return calculate_c3(oe)

def do_quads_intersect_anything(oe,quads,other_quads):
    for i,quad in enumerate(quads):
        # if(is_quad_self_intersecting(oe.z[quad.original_edge_points],oe.r[quad.original_edge_points])):
        #     return True
        # for other_quad in quads[i+1:]:
        for other_quad in quads[i:]: # also checks self-intersection
            if(does_quad_intersect_other_quad(oe.z[quad.original_edge_points],oe.r[quad.original_edge_points],oe.z[other_quad.original_edge_points],oe.r[other_quad.original_edge_points])):
                return True
        for other_quad in other_quads:
            if(does_quad_intersect_other_quad(oe.z[quad.original_edge_points],oe.r[quad.original_edge_points],oe.z[other_quad.original_edge_points],oe.r[other_quad.original_edge_points])):
                return True
    # now check self-intersections of other quads with each other
    for i,quad in enumerate(other_quads):
        for other_quad in other_quads[i:]:
            if(does_quad_intersect_other_quad(oe.z[quad.original_edge_points],oe.r[quad.original_edge_points],oe.z[other_quad.original_edge_points],oe.r[other_quad.original_edge_points])):
                return True
    return False

def does_coarse_mesh_intersect(oe):
    return intersections_in_segment_list(oe.define_coarse_mesh_segments())

def does_fine_mesh_intersect(oe):
    return intersections_in_segment_list(oe.define_fine_mesh_segments())

def intersections_in_segment_list(segments):
    for i,segment in enumerate(segments):
        for other_segment in segments[i+1:]:
            if(do_segments_intersect(*segment,*other_segment)):
                return True
    return False

# also checks coarse-coarse
def does_fine_mesh_intersect_coarse(oe):
    fine_segments = oe.define_fine_mesh_segments()
    return intersections_between_two_segment_lists(fine_segments,oe.define_coarse_mesh_segments())

def intersections_between_two_segment_lists(segments,other_segments):
    for segment in segments:
        for other_segment in other_segments:
            if(do_segments_intersect(*segment,*other_segment)):
                return True
    return False

# generalized function for checking for intersecting line segments 
# between two quads
# can also be used to check self-intersection
def does_quad_intersect_other_quad(z_shape,r_shape,other_quad_z,other_quad_r):
    n_pts = len(z_shape)
    for i in range(n_pts): # iterate segment-by-segment
        p1 = Point(z_shape[i-1],r_shape[i-1])
        p2 = Point(z_shape[i],r_shape[i])
        # now check points on all other quads
        for j in range(len(other_quad_z)):
            q1 = Point(other_quad_z[j-1],other_quad_r[j-1])
            q2 = Point(other_quad_z[j],other_quad_r[j])
            # exclude cases where segments share at least one point
            # these are counted as intersecting by do_segments_intersect
            # but that intersection is fine
            ## if(p1 == q1 or p1 == q2 or p2 == q2 or p2 == q1):
            ##     continue
            # now handled by do_segments_intersect
            if(do_segments_intersect(p1,p2,q1,q2)):
                return True
    return False

# intersection if q1 and q2 are on opposite sides of p1-p2
# and if p1 and p2 are on opposite sides of q1-q2
# based on www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def do_segments_intersect(p1,p2,q1,q2):
    cpa = cross_product_sign(p1,p2,q1)
    cpb = cross_product_sign(p1,p2,q2)
    cpc = cross_product_sign(q1,q2,p1)
    cpd = cross_product_sign(q1,q2,p2)
    # counting collinear points as non-intersecting
    if(cpa == 0 or cpb == 0 or cpc == 0 or cpd == 0):
        return False 
    else:
        return (cpa != cpb and cpc != cpd)
    ## return (cross_product_sign(p1,p2,q1) != cross_product_sign(p1,p2,q2) and cross_product_sign(q1,q2,p1) != cross_product_sign(q1,q2,p2))

# cross product of p1->p2 and p1->p3
def cross_product_sign(p1,p2,p3):
    return np.sign((p2.z-p1.z)*(p3.r-p1.r) - (p3.z-p1.z)*(p2.r-p1.r))

#######
# outdated functions
# redundant not not robust against all intersections
def does_quad_intersect_anything(z_shape,r_shape,other_edges_z,other_edges_r):
    n_pts = len(z_shape)
    for i in range(n_pts): # iterate segment-by-segment
        p1 = Point(z_shape[i-1],r_shape[i-1])
        p2 = Point(z_shape[i],r_shape[i])
        # skip immediate next segment that has a shared vertex 
        # shared vertex = no intersection
        # then check all following segments
        # also skip last segment when i-1 = -1
        end_of_range = min(n_pts,n_pts+i-1) 
        for j in range(i+2,end_of_range):
            q1 = Point(z_shape[j-1],r_shape[j-1])
            q2 = Point(z_shape[j],r_shape[j])
            if(do_segments_intersect(p1,p2,q1,q2) == True):
                return True
        # now chec points on all other quads
        for j in range(len(other_edges_z)):
            q1 = Point(other_edges_z[j-1],other_edges_r[j-1])
            q2 = Point(other_edges_z[j],other_edges_r[j])
            if(do_segments_intersect(p1,p2,q1,q2) == True):
                return True
    return False

# works, now redundant
def is_quad_self_intersecting(z_shape,r_shape):
    # z_shape,r_shape = np.split(shape,2)
    n_pts = len(z_shape)
    for i in range(n_pts): # iterate segment-by-segment
        p1 = Point(z_shape[i-1],r_shape[i-1])
        p2 = Point(z_shape[i],r_shape[i])
        # skip immediate next segment that has a shared vertex 
        # shared vertex = no intersection
        # then check all following segments
        # also skip last segment when i-1 = -1
        end_of_range = min(n_pts,n_pts+i-1) 
        for j in range(i+2,end_of_range):
            q1 = Point(z_shape[j-1],r_shape[j-1])
            q2 = Point(z_shape[j],r_shape[j])
            if(do_segments_intersect(p1,p2,q1,q2) == True):
                return True
    return False

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

# redundant
# calls does_quad_intersect_anything, which is not robust to all intersections
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

# calls does_quad_intersect_anything, which is not robust to all intersections
def change_shape_and_calculate(shape,oe,edge_points,other_edges):
    z_shape,r_shape = np.split(np.array(shape),2) # added np.array for skopt
    other_edges_z,other_edges_r = other_edges
    # hacky binary constraints
    if(does_quad_intersect_anything(z_shape,r_shape,other_edges_z,other_edges_r)):
        return 10000
    oe.z[edge_points] = z_shape
    oe.r[edge_points] = r_shape
    return calculate_c3(oe)

# calls does_quad_intersect_anything, which is not robust to all intersections
class OptimizeSingleMagMatShape:
    def __init__(self,oe,n_quad,z_min=None,z_max=None,r_min=None,r_max=None,c3=None,n_random_starts=100,n_calls=200):
        self.oe = oe
        oe.verbose = False
        self.edge_points,other_edge_points = quad_and_other_edge_points(oe,n_quad)
        initial_shape = np.concatenate((oe.z[self.edge_points],oe.r[self.edge_points])).tolist()
        self.other_edge_shape = (oe.z[other_edge_points],oe.r[other_edge_points])
        n = len(oe.z[self.edge_points])
        bounds = [(z_min,z_max)]*n+[(r_min,r_max)]*n
        result = self.minimize(initial_shape,bounds,c3,n_random_starts,n_calls)
        # print('Optimization complete with success flag {}'.format(result.success))
        # print(result.message)
        print('Optimization complete.')
        self.change_shape_and_calculate(result.x)

    
    def minimize(self,initial_shape,bounds,c3,n_random_starts,n_calls):
        return gbrt_minimize(self.change_shape_and_calculate,dimensions=bounds,x0=initial_shape,y0=c3,n_random_starts=n_random_starts,n_calls=n_calls)
    
    def change_shape_and_calculate(self,shape):
        z_shape,r_shape = np.split(np.array(shape),2)
        other_edges_z,other_edges_r = self.other_edge_shape
        # hacky binary constraints
        if(does_quad_intersect_anything(z_shape,r_shape,other_edges_z,other_edges_r)):
            return 10000
        self.oe.z[self.edge_points] = z_shape
        self.oe.r[self.edge_points] = r_shape
        return calculate_c3(self.oe)

# list version is better
def find_mirrored_edge_points_array(oe,edge_points):
    if 0 in oe.z[edge_points]:
        ind = np.nonzero(oe.z[edge_points] == 0)
        mirrored_edge_points = (edge_points[0][ind],edge_points[1][ind])
        edge_points = (np.delete(edge_points[0],ind),np.delete(edge_points[1],ind))
        return mirrored_edge_points,edge_points
    else:
        return None,edge_points

# broken by later changes to change_shape_and_calculate
# (no intersection checks)
def optimize_mag_mat_shape(oe,z_min=None,z_max=None,r_min=None,r_max=None):
    oe.verbose = False
    edge_points = oe.retrieve_edge_points(oe.mag_mat_z_indices,oe.mag_mat_r_indices,return_ind_array=True)
    initial_shape = np.concatenate((oe.z[edge_points],oe.r[edge_points]))
    n = len(oe.z[edge_points])
    bounds = [(z_min,z_max)]*n+[(r_min,r_max)]*n
    result = minimize(change_shape_and_calculate,initial_shape,args=(oe,edge_points),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1,'disp':True})
    print('Optimization complete with success flag {}'.format(result.success))
    print(result.message)
    change_shape_and_calculate(result.x,oe,edge_points)
    # oe.write(oe.filename)


