#!/usr/bin/env python3
'''
User methods:
    optimize_many_shapes
    OptimizeShapes (built as a class for obscure reasons)
'''
import sys,os,subprocess,shutil,datetime
from subprocess import TimeoutExpired
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from contextlib import contextmanager
from optical_element_io import cd,index_array_from_list,np_index
# from optical_element_io import Point
from calculate_optical_properties import calc_properties_optics, calc_properties_mirror
from scipy.optimize import minimize
from skopt import gbrt_minimize, gp_minimize, dummy_minimize, forest_minimize
import asyncio
# from sympy import *
# from sympy.geometry import *
from shapely.geometry import *

class TimeoutCheck:
    def __init__(self):
        self.timed_out = False

def calculate_c3(oe,col,curr_bound=None,t=None):
    '''
    Workhorse function for automation. Writes optical element file, then 
    calculates field, then calculates optical properties, and then reads them.

    Parameters:
        oe : OpticalElement object
            optical element on which to calculate spherical aberration.
    '''
    
    oe.write(oe.filename)
    oe.calc_field()
    if(col.program == 'optics'):
        try:
            calc_properties_optics(oe,col)
        except TimeoutExpired: # if optics has failed over and over again, bail
            t.timed_out = True
            return 10000 # likely dongle error
    if(col.program == 'mirror'):
        try:
            calc_properties_mirror(oe,col)
        except asyncio.TimeoutError:
            return 10000 # likely reached a shape where the image plane is too far for MIRROR
    try: 
        if(col.program == 'optics'):
            col.read_optical_properties()
        if(col.program == 'mirror'):
            col.read_mir_optical_properties(raytrace=False)
    except UnboundLocalError: # if optics fails, return garbage
        return 100
    print(f"f: {col.f}, C3: {col.c3}")
    if(curr_bound):
        coil_area = 0
        for i in range(len(oe.coil_z_indices)):
            coil_area += oe.determine_quad_area(oe.coil_z_indices[i],oe.coil_r_indices[i])
        if(oe.lens_curr/coil_area > curr_bound):
            return 100
    return np.abs(col.c3)

def change_current_and_calculate(current,oe,col):
    oe.coil_curr = current
    return calculate_c3(oe,col,t=TimeoutCheck())

# for a single coil such that oe.coil_curr = [current]
def optimize_single_current(oe,col):
    '''
    In principle, finds current of one coil necessary to minimize spherical
    aberration. In practice, has multiple problems:
    -MEBS throws popups when not running with a defined image plane
    -higher current will probably always be better

    Not currently useful but could be made so with changes.
    '''
    result = minimize(change_current_and_calculate,oe.coil_curr,args=(oe,col),method='Nelder-Mead')
    oe.coil_curr = result.x
    oe.write(oe.filename)
    print('Optimization complete')

class OptimizeShapes:
    '''
    This class is necessary to use skopt minimizers as they don't allow 
    additional arguments to be passed to the objective function like scipy.
    Also, skopt uses a list and cannot use a numpy array for the parameters
    to optimize.

    Copy of optimize_many_shapes with slight tweaks. Actively updated for
    consistency but not actively bug-tested and may be broken.
    '''
    minimize_switch = {'gbrt': gbrt_minimize,'gp': gp_minimize, 'forest': forest_minimize, 'dummy': dummy_minimize}

    def __init__(self,oe,col,z_indices_list,r_indices_list,other_z_indices_list=None,other_r_indices_list=None,z_min=None,z_max=None,r_min=None,r_max=None,method='gbrt',c3=None,n_random_starts=10,n_calls=100):
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
                only use now is avoiding breakdown fields in electrodes.
                default None.
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
        if(other_z_indices_list is None):
            other_z_indices_list = []
        if(other_r_indices_list is None):
            other_r_indices_list = []
        self.oe = oe
        self.col = col
        self.quads,all_edge_points_list,all_mirrored_edge_points_list,all_Rboundary_edge_points_list = define_edges(oe,z_indices_list,r_indices_list)
        self.other_quads,_,_,_ = define_edges(oe,other_z_indices_list,other_r_indices_list,remove_duplicates_and_mirrored=False)
        self.n_edge_pts = len(all_edge_points_list)
        n_mirrored_edge_pts = len(all_mirrored_edge_points_list) if any(all_mirrored_edge_points_list) else 0
        n_Rboundary_edge_pts = len(all_Rboundary_edge_points_list) if any(all_Rboundary_edge_points_list) else 0
        edge_points = index_array_from_list(all_edge_points_list)
        mirrored_edge_points = index_array_from_list(all_mirrored_edge_points_list)
        Rboundary_edge_points = index_array_from_list(all_Rboundary_edge_points_list)
        initial_shape = np.concatenate((oe.z[edge_points],oe.z[Rboundary_edge_points],oe.r[edge_points],oe.r[mirrored_edge_points])).tolist()
        bounds = (self.n_edge_pts+n_Rboundary_edge_pts)*[(z_min,z_max)]+(self.n_edge_pts+n_mirrored_edge_pts)*[(r_min,r_max)]
        result = self.minimize_switch.get(method,dummy_minimize)(self.change_n_quads_and_calculate,x0=initial_shape,dimensions=bounds,n_random_starts=n_random_starts,n_calls=n_calls,y0=c3)
        print('Optimization complete.')
        self.change_n_quads_and_calculate(result.x)

    def change_n_quads_and_calculate(self,shape):
        return change_n_quads_and_calculate(np.array(shape),self.oe,self.col,self.quads,self.other_quads,self.n_edge_pts,t=TimeoutCheck())
        
    
def optimize_many_shapes(oe,col,z_indices_list,r_indices_list,other_z_indices_list=None,other_r_indices_list=None,z_curv_z_indices_list=None,z_curv_r_indices_list=None,r_curv_z_indices_list=None,r_curv_r_indices_list=None,end_z_indices_list=None,end_r_indices_list=None,z_min=None,z_max=None,automate_present_curvature=False,r_min=0,r_max=None,method='Nelder-Mead',manual_bounds=True,options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None,'return_all':True},simplex_scale=5,curve_scale=0.05,curr_bound=3,breakdown_field=10e3):
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
            only use now is avoiding breakdown fields in electrodes.
            default None.
        z_curv_z_indices_list : list
        z_curv_r_indices_list : list
        r_curv_z_indices_list : list
        r_curv_r_indices_list : list
            point-by-point list of MEBS indices denoting segments where curvature 
            (oe.z_curv or oe.r_curv) should be included in automation.
            e.g. [1,11,21],[1,35,66] for MEBS points (1,1), (11,35) etc.
            use automate_present_curvature=True for easier operation.
            default None.
        end_z_indices_list : list
        end_r_indices_list : list
            these lists are constructed as [[line1_MEBS_z],[line2_MEBS_z],...]
            and [[line1_MEBS_r1,line1_MEBS_r2,...],[line2_MEBS_r1,line2_MEBS_r2,...],...]
            default None.
        automate_present_curvature : bool
            if set to True, the *_curv_*_indices_list inputs are ignored, and 
            present curvature in oe.z_curv, oe.r_curv is used instead to
            determine which curvature to update in automation.
            default False.
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
        curvature_scale : float
            curvature (in 1/mm) for normal distribution of simplex points
            around initial shape. only used with Nelder Mead. 68% of generated
            curvatures will have a radius of more than 1/curvature_scale.
            default 0.05, so a radius of 20mm.
        curr_bound : float
            bound for maximum current desnity in first magnetic lens. 
            default 3 A/mm^2 current density limit
        breakdown_field : float
            field at which vacuum breakdown is possible, in V/mm.
            Default 10,000 V/mmm.
    '''
    if(other_z_indices_list is None):
        other_z_indices_list = []
    if(other_r_indices_list is None):
        other_r_indices_list = []
    if(z_curv_z_indices_list is None):
        z_curv_z_indices_list = []
    if(z_curv_r_indices_list is None):
        z_curv_r_indices_list = []
    if(r_curv_z_indices_list is None):
        r_curv_z_indices_list = []
    if(r_curv_r_indices_list is None):
        r_curv_r_indices_list = []
    if(end_z_indices_list is None):
        end_z_indices_list = []
    if(end_r_indices_list is None):
        end_r_indices_list = []

    options_mutable = options.copy()
    quads,all_edge_points_list,all_mirrored_edge_points_list,all_Rboundary_edge_points_list = define_edges(oe,z_indices_list,r_indices_list)
    other_quads,_,_,_ = define_edges(oe,other_z_indices_list,other_r_indices_list,remove_duplicates_and_mirrored=False)
    if(automate_present_curvature):
        z_curv_points = np.nonzero(oe.z_curv)
        r_curv_points = np.nonzero(oe.r_curv)
        n_z_curv_pts = len(oe.z_curv[z_curv_points])
        n_r_curv_pts = len(oe.r_curv[r_curv_points])
    else:
        z_curv_points_list = define_curves(oe,z_curv_z_indices_list,z_curv_r_indices_list)
        r_curv_points_list = define_curves(oe,r_curv_z_indices_list,r_curv_r_indices_list)
        n_z_curv_pts = len(z_curv_points_list)
        n_r_curv_pts = len(r_curv_points_list)
        z_curv_points = index_array_from_list(z_curv_points_list)
        r_curv_points = index_array_from_list(r_curv_points_list)
    end_electrode_points_list = define_end(oe,end_z_indices_list,end_r_indices_list)
    n_edge_pts = len(all_edge_points_list)
    n_mirrored_edge_pts = len(all_mirrored_edge_points_list) if any(all_mirrored_edge_points_list) else 0
    n_Rboundary_edge_pts = len(all_Rboundary_edge_points_list) if any(all_Rboundary_edge_points_list) else 0
    n_end_pts = len(end_electrode_points_list) if any(end_electrode_points_list) else 0
    n_curv_pts = n_z_curv_pts+n_r_curv_pts
    edge_points = index_array_from_list(all_edge_points_list)
    mirrored_edge_points = index_array_from_list(all_mirrored_edge_points_list)
    Rboundary_edge_points = index_array_from_list(all_Rboundary_edge_points_list)
    end_points = index_array_from_list(end_electrode_points_list)
    # getting complicated, could be rewritten
    if(n_z_curv_pts):
        inv_z_curv = np.divide(1,oe.z_curv[z_curv_points],where=(oe.z_curv[z_curv_points] != 0))
    else:
        inv_z_curv = oe.z_curv[z_curv_points] # empty array
    if(n_r_curv_pts):
        inv_r_curv = np.divide(1,oe.r_curv[r_curv_points],where=(oe.r_curv[r_curv_points] != 0))
    else:
        inv_r_curv = oe.r_curv[r_curv_points] # empty array
    initial_shape = np.concatenate((oe.z[edge_points],oe.z[Rboundary_edge_points],oe.z[end_points],oe.r[edge_points],oe.r[mirrored_edge_points],inv_z_curv,inv_r_curv)).tolist()
    bounds = np.array((n_edge_pts+n_Rboundary_edge_pts+n_end_pts)*[(z_min,z_max)]+(n_edge_pts+n_mirrored_edge_pts)*[(r_min,r_max)]+n_curv_pts*[(None,None)]) # could add segment length-based bounds, but would be complicated
    edge_pts_splitlist = [n_edge_pts,n_edge_pts+n_Rboundary_edge_pts,n_edge_pts+n_Rboundary_edge_pts+n_end_pts,2*n_edge_pts+n_Rboundary_edge_pts+n_end_pts,2*n_edge_pts+n_Rboundary_edge_pts+n_end_pts+n_mirrored_edge_pts,2*n_edge_pts+n_Rboundary_edge_pts+n_end_pts+n_mirrored_edge_pts+n_z_curv_pts]
    shape_data = ShapeData(quads,other_quads,edge_pts_splitlist,edge_points,Rboundary_edge_points,end_points,mirrored_edge_points,z_curv_points,r_curv_points)
    if(method=='Nelder-Mead' and options.get('initial_simplex') is None):
        print('Generating initial simplex.')
        options_mutable['initial_simplex'] = generate_initial_simplex(initial_shape,oe,shape_data,enforce_bounds=True,bounds=np.array(bounds),breakdown_field=breakdown_field,scale=simplex_scale,n_curve_points=n_curv_pts,curve_scale=curve_scale)
        print('Finished initial simplex generation.')
    if(manual_bounds):
        if(oe.lens_type == 'magnetic'):
            result = minimize(change_n_quads_and_calculate,initial_shape,args=(oe,col,shape_data,TimeoutCheck(),True,np.array(bounds),curr_bound),method=method,options=options_mutable)
        elif(oe.lens_type == 'electrostatic'):
            result = minimize(change_n_quads_and_calculate,initial_shape,args=(oe,col,shape_data,TimeoutCheck(),True,np.array(bounds),None,breakdown_field),method=method,options=options_mutable)
    else:
        result = minimize(change_n_quads_and_calculate,initial_shape,
                          args=(oe,col,shape_data,TimeoutCheck()),
                          bounds=bounds,method=method,options=options_mutable)
    print('Optimization complete with success flag {}'.format(result.success))
    print(result.message)
    change_n_quads_and_calculate(result.x,oe,col,shape_data)
    if(col.program == 'mirror'):
        col.raytrace_from_saved_values()
    if(method=='Nelder-Mead' and options.get('return_all') == True):
        np.save(oe.filename_noext+'_all_solns',result['allvecs'])

def optimize_image_plane(oe,min_dist=3,image_plane=6):
    '''
    In principle, could be used to find the optimal image plane to minimize
    spherical aberration. In practice, that is always zero distance from the
    object plane, so needs to be rewritten to be useful.
    '''
    initial_plane = [image_plane] # mm
    bounds = [(min_dist,100)]
    result = minimize(change_imgplane_and_calculate,initial_plane,args=(oe),bounds=bounds,method='TNC',options={'eps':0.5,'stepmx':5,'minfev':1})
    change_imgplane_and_calculate(result.x,oe)
    print('Optimization complete')

class ShapeData:
    '''
    Class to carry around general shape information for optimization.
    '''
    def __init__(self,quads,other_quads,splitlist,edge_points,Rboundary_edge_points,end_points,mirrored_edge_points,z_curv_points,r_curv_points):
        self.quads = quads
        self.other_quads = other_quads
        self.splitlist = splitlist
        self.edge_points = edge_points
        self.Rboundary_edge_points = Rboundary_edge_points
        self.end_points = end_points
        self.mirrored_edge_points = mirrored_edge_points
        self.z_curv_points = z_curv_points
        self.r_curv_points = r_curv_points

class Quad:
    '''
    Class to carry around quad information for optimization.
    '''

    def __init__(self,oe,z_indices,r_indices,separate_mirrored=True,separate_radial_boundary=False):

        # determine if quad is electrode
        try:  # works if lens is electric
            is_array_in_list = [np.array_equal(z_indices,element) for element in oe.electrode_z_indices]
            if(any(is_array_in_list)):
            # unfortunately verbose but functional version of
            ## z_indices in oe.electrode_z_indices:
                self.electrode = True
                self.electrode_index = is_array_in_list.index(True)
                # unfortunately verbose but functional version of 
                ## self.electrode_index = oe.electrode_z_indices.index(z_indices)
            else:
                self.electrode = False
                self.electrode_index = None
        except (NameError,AttributeError): # oe.electrode_z_indices doesn't exist because lens isn't electric
            self.electrode = False
            self.electrode_index = None

        self.z_indices = z_indices
        self.r_indices = r_indices
        self.edge_points_list = oe.retrieve_single_quad_edge_points(z_indices,r_indices)
        self.original_edge_points_list = self.edge_points_list.copy()
        if(separate_mirrored):
            self.mirrored_edge_points_list,self.edge_points_list = find_mirrored_edge_points(oe,self.edge_points_list)
        else:
            self.mirrored_edge_points_list = [],[]
        if(separate_radial_boundary):
            self.Rboundary_edge_points_list,self.edge_points_list = find_Rboundary_edge_points(oe,self.edge_points_list)
        else:
            self.Rboundary_edge_points_list = [],[]

    def delete_overlaps(self,edge_points_list,prior_edge_points_list):
        edge_points_list = [point for point in edge_points_list if point not in prior_edge_points_list]

    def count(self):
        self.n_edge_pts = len(self.edge_points_list)
        self.n_mirrored_edge_pts = len(self.mirrored_edge_points_list)
        self.n_Rboundary_edge_pts = len(self.Rboundary_edge_points_list)

    def make_index_arrays(self):
        self.edge_points = index_array_from_list(self.edge_points_list)
        self.original_edge_points = index_array_from_list(self.original_edge_points_list)
        self.mirrored_edge_points = index_array_from_list(self.mirrored_edge_points_list)
        self.Rboundary_edge_points = index_array_from_list(self.Rboundary_edge_points_list)


def define_curves(oe,z_indices_list,r_indices_list):
    curves_points_list = []
    for z,r in zip(z_indices_list,r_indices_list):
        curves_points_list.append([np_index(oe.r_indices,r),np_index(oe.z_indices,z)])
    return curves_points_list

def define_end(oe,z_indices_list,r_indices_list):
    end_points_list = []
    for z,r_list in zip(z_indices_list,r_indices_list):
        for r in r_list:
            end_points_list.append([np_index(oe.r_indices,r),np_index(oe.z_indices,z)])
    return end_points_list


# takes a list of quads defined by two z indices and two r indices
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
    all_Rboundary_edge_points_list = []
    for i in range(n_quads):
        quads.append(Quad(oe,z_indices_list[i],r_indices_list[i],separate_mirrored=(remove_duplicates_and_mirrored and oe.freeze_xy_plane),separate_radial_boundary=(remove_duplicates_and_mirrored and oe.freeze_radial_boundary)))
        if(remove_duplicates_and_mirrored):
            quads[-1].delete_overlaps(quads[-1].edge_points_list,all_edge_points_list)
            if(oe.freeze_xy_plane):
                quads[-1].delete_overlaps(quads[-1].mirrored_edge_points_list,all_mirrored_edge_points_list)
            if(oe.freeze_radial_boundary):
                quads[-1].delete_overlaps(quads[-1].Rboundary_edge_points_list,all_Rboundary_edge_points_list)
        quads[-1].count()
        all_edge_points_list += quads[-1].edge_points_list
        all_mirrored_edge_points_list += quads[-1].mirrored_edge_points_list
        all_Rboundary_edge_points_list += quads[-1].Rboundary_edge_points_list
        quads[-1].make_index_arrays()
    return quads,all_edge_points_list,all_mirrored_edge_points_list,all_Rboundary_edge_points_list

def generate_initial_simplex(initial_shape,oe,shape_data,enforce_bounds=True,bounds=None,breakdown_field=None,scale=5,n_curve_points=None,curve_scale=0):
    rng = np.random.default_rng()
    N = len(initial_shape)
    simplex = np.zeros((N+1,N),dtype=float)
    if(n_curve_points):
        n_snc = len(initial_shape[:-n_curve_points]) # snc is short for 'shape, no curves'
        scale_array = np.concatenate([np.ones((n_snc),dtype=float)*scale,np.ones((n_curve_points),dtype=float)*curve_scale])
    else:
        scale_array = scale
    for i in range(N+1):
        simplex[i] = rng.normal(initial_shape,scale_array)
        # keep trying until simplex point is valid
        # inefficient but simple
        while(change_n_quads_and_check(simplex[i],oe,shape_data,enforce_bounds,
                                       bounds,breakdown_field)):
            simplex[i] = rng.normal(initial_shape,scale_array)
        if(oe.verbose):
            print(f'Simplex {i+1} of {N+1} complete.')
    # save result
    np.save(os.path.join(oe.dirname,'initial_simplex_for_'+oe.basename_noext),simplex)
    # return shape to initial shape
    change_n_quads_and_check(initial_shape,oe,shape_data)
    return simplex

def change_n_quads_and_check(shape,oe,shape_data,enforce_bounds=False,
                             bounds=None,breakdown_field=None):
    # z_shapes,Rboundary_z_shapes,end_z_shapes,r_shapes,mirrored_r_shapes,z_curv,r_curv = np.split(shape,shape_data.edge_pts_splitlist)
    oe.z[shape_data.edge_points],oe.z[shape_data.Rboundary_edge_points],oe.z[shape_data.end_points],oe.r[shape_data.edge_points],oe.r[shape_data.mirrored_edge_points],inv_z_curv,inv_r_curv = np.split(shape,shape_data.splitlist)
    if(len(inv_z_curv)):
        oe.z_curv[shape_data.z_curv_points] = np.divide(1.0,inv_z_curv,where=(inv_z_curv != 0))
    if(len(inv_r_curv)):
        oe.r_curv[shape_data.r_curv_points] = np.divide(1.0,inv_r_curv,where=(inv_r_curv != 0))
    # for quad in shape_data.quads:
    #     oe.z[quad.edge_points],z_shapes = np.split(z_shapes,[quad.n_edge_pts])
    #     oe.z[quad.Rboundary_edge_points],Rboundary_z_shapes = np.split(Rboundary_z_shapes,[quad.n_Rboundary_edge_pts])
    #     oe.r[quad.edge_points],r_shapes = np.split(r_shapes,[quad.n_edge_pts])
    #     oe.r[quad.mirrored_edge_points],mirrored_r_shapes = np.split(mirrored_r_shapes,[quad.n_mirrored_edge_pts])
    if(enforce_bounds):
        lb_nn = (bounds[:,0] != None)
        ub_nn = (bounds[:,1] != None)
        if((bounds[:,0][lb_nn] > shape[lb_nn]).any() or (bounds[:,1][ub_nn] < shape[ub_nn]).any()):
            return True
    if(does_coarse_mesh_intersect(oe)):
        return True
    if(does_fine_mesh_intersect_coarse(oe)):
        return True
    if(oe.lens_type == 'electrostatic' and breakdown_field and are_electrodes_too_close(oe,breakdown_field,shape_data.quads,shape_data.other_quads)):
        return True
    return False

def find_mirrored_edge_points(oe,edge_points_list):
    mirrored_edge_points_list = [point for point in edge_points_list if oe.z[point] == 0]
    edge_points_list = [point for point in edge_points_list if oe.z[point] != 0]
    return mirrored_edge_points_list,edge_points_list

def find_Rboundary_edge_points(oe,edge_points_list):
    rmax_np_index = np.argmax(oe.r[:,0])
    Rboundary_edge_points_list = [point for point in edge_points_list if point[0] == rmax_np_index]
    edge_points_list = [point for point in edge_points_list if point[0] != rmax_np_index]
    return Rboundary_edge_points_list,edge_points_list

def change_imgplane_and_calculate(imgplane,oe):
    oe.write_opt_img_cond_file(oe.imgcondfilename,img_pos=imgplane[0])
    calc_properties_optics(oe)
    try: 
        oe.read_optical_properties()
    except UnboundLocalError: # if optics fails, return garbage
        return 100
    print(f"f: {col.f}, C3: {col.c3}")
    return np.abs(col.c3)

def change_n_quads_and_calculate(shape,oe,col,shape_data,t=TimeoutCheck(),enforce_bounds=False,bounds=None,curr_bound=None,breakdown_field=None):
    if(t.timed_out):
        return 10000
    if(change_n_quads_and_check(shape,oe,shape_data,enforce_bounds=enforce_bounds,bounds=bounds,breakdown_field=breakdown_field)):
        return 10000
    return calculate_c3(oe,col,curr_bound,t)

def are_electrodes_too_close(oe,breakdown_field,quads,other_quads):
    for i,quad in enumerate(quads):
        if(quad.electrode):
            for other_quad in quads[i+1:]: 
                if(other_quad.electrode):
                    if(max_field(quad,other_quad,oe) > breakdown_field):
                        return True
            for other_quad in other_quads:
                if(other_quad.electrode):
                    if(max_field(quad,other_quad,oe) > breakdown_field):
                        return True

def max_field(quad,other_quad,oe):
    delta_V = np.abs(oe.V[quad.electrode_index] - oe.V[other_quad.electrode_index])
    return delta_V/min_distance(quad,other_quad,oe)

def min_distance(quad,other_quad,oe):
    return oe.make_polygon(quad.z_indices,quad.r_indices).distance(oe.make_polygon(other_quad.z_indices,other_quad.r_indices))

def does_coarse_mesh_intersect(oe):
    try:
        return intersections_in_segment_list(oe.define_coarse_mesh_segments())
    except ValueError: # curvature too high; not really an intersection, but still not a valid mesh
        return True

def intersections_in_segment_list(segments):
    for i,segment in enumerate(segments):
        # check to make sure segment and other_segment are not none
        if(segment):
            for other_segment in segments[i+1:]:
                if(other_segment and segment.shape.crosses(other_segment.shape)):
                    return True
    return False

# only checks non-coarse fine with coarse
def does_fine_mesh_intersect_coarse(oe):
    try:
        fine_segments = oe.define_fine_mesh_segments()
    except ValueError:
        return True
    # oe.coarse_segments should already be defined
    fine_no_coarse = [segment for segment in fine_segments if segment not in oe.coarse_segments]
    return intersections_between_two_segment_lists(fine_no_coarse,oe.coarse_segments.flatten())

def intersections_between_two_segment_lists(segments,other_segments):
    for segment in segments:
        # check to make sure segment and other_segment are not none
        if(segment):
            for other_segment in other_segments:
                if(other_segment and segment.shape.crosses(other_segment.shape)):
                    intersection_point = segment.shape.intersection(other_segment.shape)
                    if([point for point in segment.shape.boundary if point.almost_equals(intersection_point)] or
                       [point for point in other_segment.shape.boundary if point.almost_equals(intersection_point)]):
                        continue
                    else:
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
    return (cpa != cpb and cpc != cpd)

# cross product of p1->p2 and p1->p3
def cross_product_sign(p1,p2,p3,tol=1e-10):
    cp = (p2.z-p1.z)*(p3.r-p1.r) - (p3.z-p1.z)*(p2.r-p1.r)
    if(abs(cp) < tol): # allow collinearity with some finite tolerance
        return 0
    return np.sign(cp)

