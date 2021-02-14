#!/usr/bin/env python3
'''
Contains only user methods; everything else is in automation_library.py.
User methods:
    optimize_many_shapes
    optimize_broadly_for_retracing
'''
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from scipy.optimize import minimize, minimize_scalar
# from shapely.geometry import *
from automation_library import calculate_c3, \
                               change_image_plane_and_check_retracing, \
                               change_voltages_and_shape_and_check_retracing, \
                               change_voltages_and_check_retracing, \
                               change_imgplane_and_calculate, \
                               change_n_quads_and_calculate, \
                               determine_img_pos_limits, \
                               generate_initial_simplex, \
                               prepare_shapes, \
                               change_n_quads_and_check, \
                               TimeoutCheck

def optimize_planes_for_retracing(col, bounds=(0,200), img_pos=90, **kwargs):
    result = minimize_scalar(change_image_plane_and_check_retracing, args=(col, kwargs),  
                             method='bounded', bounds=bounds)

    print(f"Retracing image plane: {result.x}mm.")
    img_pos = result.x

    col.write_mir_img_cond_file(col.mircondfilename, source_pos=img_pos-col.img_source_offset, 
                                img_pos=img_pos, **kwargs)

    col.write_raytrace_file(col.mircondfilename, source_pos=img_pos-col.img_source_offset,
                            screen_pos=img_pos, minimum_rays=True, **kwargs)


def optimize_broadly_for_retracing(
        oe, col, potentials, img_pos, z_indices_list=None, r_indices_list=None, 
        other_z_indices_list=None, other_r_indices_list=None, z_curv_z_indices_list=None, z_curv_r_indices_list=None, 
        r_curv_z_indices_list=None, r_curv_r_indices_list=None, end_z_indices_list=None, end_r_indices_list=None, 
        z_min=None, z_max=None, r_min=0, r_max=None, breakdown_field=10e3, 
        options={'adaptive':True,'fatol':0.00001,'disp':True,'return_all':True}, 
        simplex_scale=4, curve_scale=0.05, voltage_logscale=2, **kwargs):
    '''
    Automated optimization of any electrode shape, electrode voltages,
    and the image position for ray-retracing. Shape optimization is very
    similar to optimize_many_shapes() and also calls prepare_shapes(), 
    but intended for smaller-scale tweaking after initial optimization with 
    optimize_many_shapes(), so a number of keyword arguments that broaden the 
    usage of optimize_many_shapes are not available here.

    Parameters:
        oe : OpticalElement object
            optical element to optimize
        col : OpticalColumn object
            optical column to optimize
        potentials : OpticalElement.MirPotentials object
            initial guess and constraints on voltages to optimize.
            uses MEBS flags ('f','v1', etc.) to determine which voltages
            should be varied.

    Optional parameters: 
        r_indices_list : list
        z_indices_list : list
            list of lists of two MEBS r indices and two MEBS z indices that 
            defines quads to optimize
            default None.
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
        z_min : float
        z_max : float
        r_min : float
        r_max : float
            bounds
            default None, except r_min = 0.
        breakdown_field : float
            field at which vacuum breakdown is possible, in V/mm.
            Default 10,000 V/mmm.
        options : dict
            options for Nelder-Mead.
            default {'adaptive':True}.
        simplex_scale : float
            size (in mm) for normal distribution of simplex points around
            initial shape. only used with Nelder-Mead. A larger scale results
            in a longer search that is more likely to find a qualitatively
            different shape.
            default 3.
        curvature_scale : float
            curvature (in 1/mm) for normal distribution of simplex points
            around initial shape. only used with Nelder Mead. 68% of generated
            curvatures will have a radius of more than 1/curvature_scale.
            default 0.05, so a radius of 20mm.
        voltage_logscale : float
            scale for log-normal distribution of voltage points around initial
            configuaration. units are np.log(Volts).
            default 2.
        **kwargs 
            used to pass additional kwargs to write_raytrace_file().

    '''
    options_mutable = options.copy()

    initial_shape, bounds, shape_data = prepare_shapes(
        oe, col, z_indices_list, r_indices_list, other_z_indices_list, other_r_indices_list, 
        z_curv_z_indices_list, z_curv_r_indices_list, r_curv_z_indices_list, r_curv_r_indices_list,
        end_z_indices_list, end_r_indices_list, z_min, z_max, r_min, r_max, automate_present_curvature=False)

    potentials.voltages = np.array(potentials.voltages) 
    flag_mask = np.array(potentials.flags) != 'f'
    voltages = potentials.voltages[flag_mask]

    img_pos_bounds = determine_img_pos_limits(oe)

    initial_parameters = initial_shape+voltages.tolist() + [img_pos]
    N = len(initial_parameters)
    
    # generate shape simplex
    if(options.get('initial_simplex') is None):
        options_mutable['initial_simplex'] = generate_initial_simplex(
            initial_shape, oe, shape_data, enforce_bounds=True, bounds=np.array(bounds), 
            breakdown_field=breakdown_field, scale=simplex_scale, curve_scale=curve_scale, adaptive=True, N=N)

     

    rng = np.random.default_rng()
    voltage_simplex = np.zeros((N+1,len(voltages)), dtype=float)
    img_pos_simplex = np.zeros((N+1,1), dtype=float)
    for i in range(N+1):
        voltage_simplex[i] = np.exp(rng.normal(np.log(voltages-voltages.min()+1000), voltage_logscale)) \
                             + voltages.min()-1000
        img_pos_simplex[i,:] = rng.uniform(*img_pos_bounds)

    options_mutable['initial_simplex'][:,shape_data.n_pts:-1] = voltage_simplex
    options_mutable['initial_simplex'][:,-1:] = img_pos_simplex

    result = minimize(change_voltages_and_shape_and_check_retracing, initial_parameters,
                      args=(oe, col, potentials, flag_mask, shape_data, bounds, breakdown_field, kwargs),
                      method='Nelder-Mead', options=options_mutable)
    print(result)

    if(options.get('return_all') == True):
        np.save(oe.filename_noext+'_all_solns', result['allvecs'])

    potentials.voltages[flag_mask] = result.x[shape_data.n_pts:-1]
    img_pos = result.x[-1]
    potentials.voltages = potentials.voltages.tolist()
    col.write_mir_img_cond_file(col.mircondfilename, potentials=potentials,
                                source_pos=img_pos-col.img_source_offset, img_pos=img_pos,
                                **kwargs)
    col.write_raytrace_file(col.mircondfilename, potentials=potentials,
                            source_pos=img_pos-col.img_source_offset, screen_pos=img_pos,
                            minimum_rays=True, **kwargs)
    col.calc_rays()
    if(oe.plot):
        col.plot_rays()

def optimize_voltages_for_retracing(col, potentials, img_pos, bounds=None, options=None, **kwargs):
    potentials.voltages = np.array(potentials.voltages) 
    flag_mask = np.array(potentials.flags) != 'f'
    voltages = potentials.voltages[flag_mask]
    initial_parameters = np.append(voltages, img_pos) # [v_1, ... , v_n, img_pos]
    result = minimize(change_voltages_and_check_retracing, initial_parameters,
                      args=(col, potentials, flag_mask, kwargs),
                      method='Nelder-Mead', bounds=bounds, options=options)
    print(result)
    potentials.voltages[flag_mask] = result.x[:-1]
    img_pos = result.x[-1]
    potentials.voltages = potentials.voltages.tolist()
    col.write_mir_img_cond_file(col.mircondfilename, potentials=potentials, source_pos=img_pos-col.img_source_offset,
                                img_pos=img_pos, **kwargs)
    col.write_raytrace_file(col.mircondfilename, potentials=potentials, source_pos=img_pos-col.img_source_offset,
                            screen_pos=img_pos, minimum_rays=True, **kwargs)
    col.calc_rays()
    if(col.oe.plot):
        col.plot_rays()


def optimize_many_shapes(
        oe, col, z_indices_list, r_indices_list, other_z_indices_list=None, other_r_indices_list=None,
        z_curv_z_indices_list=None, z_curv_r_indices_list=None, r_curv_z_indices_list=None, r_curv_r_indices_list=None, 
        end_z_indices_list=None, end_r_indices_list=None, z_min=None, z_max=None, r_min=0, r_max=None, 
        automate_present_curvature=False, method='Nelder-Mead', manual_bounds=True, 
        options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None,'return_all':True}, 
        simplex_scale=5, curve_scale=0.05, curr_bound=3, breakdown_field=10e3, adaptive_simplex=True):
    '''
    Automated optimization of the shape of one or more quads with 
    scipy.optimize.minimize.

    Parameters:
        oe : OpticalElement object
            optical element to optimize
        col : OpticalColumn object
            optical column to optimize
        r_indices_list : list
        z_indices_list : list
            list of lists of two MEBS r indices and two MEBS z indices that 
            defines quads to optimize

    Optional parameters:
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
            default None, except r_min = 0.
        method : str
            name of method to use
            default 'Nelder-Mead'
        manual_bounds : boolean
            determines whether bounds will be enforced manually in objective
            function. set to False for methods like TNC that include bounds.
            set to True for Nelder-Mead, Powell, etc.
            intersections are always manually blocked.
            default True
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
        adaptive_simplex : bool
            if True, adjusts the simplex scale for each point according to an 
            initial search. if False, purely normal distribution of simplex 
            points.
            default True.
    '''
    options_mutable = options.copy()

    initial_shape, bounds, shape_data = prepare_shapes(
        oe, col, z_indices_list, r_indices_list, other_z_indices_list, other_r_indices_list, 
        z_curv_z_indices_list, z_curv_r_indices_list, r_curv_z_indices_list, r_curv_r_indices_list, 
        end_z_indices_list, end_r_indices_list, z_min, z_max, r_min, r_max, automate_present_curvature)

    if(change_n_quads_and_check(np.array(initial_shape), oe, shape_data, enforce_bounds=True, bounds=bounds, 
            breakdown_field=breakdown_field)):
        raise ValueError('Initial shape intersects or violates bounds.')
    if(method=='Nelder-Mead' and options.get('initial_simplex') is None):
        print('Generating initial simplex.')
        options_mutable['initial_simplex'] = generate_initial_simplex(
            initial_shape, oe, shape_data, enforce_bounds=True, bounds=np.array(bounds), 
            breakdown_field=breakdown_field, scale=simplex_scale, curve_scale=curve_scale, adaptive=adaptive_simplex)
        print('Finished initial simplex generation.')
    if(manual_bounds):
        if(oe.lens_type == 'magnetic'):
            result = minimize(change_n_quads_and_calculate, initial_shape, args=(oe, col, shape_data, TimeoutCheck(), 
                              True, np.array(bounds), curr_bound), method=method, options=options_mutable)
        elif(oe.lens_type == 'electrostatic'):
            result = minimize(change_n_quads_and_calculate, initial_shape, args=(oe, col, shape_data, TimeoutCheck(), 
                              True, np.array(bounds), None, breakdown_field), method=method, options=options_mutable)
    else:
        result = minimize(change_n_quads_and_calculate, initial_shape,
                          args=(oe, col, shape_data, TimeoutCheck()),
                          bounds=bounds, method=method, options=options_mutable)

    print('Optimization complete with success flag {}'.format(result.success))
    print(result.message)
    change_n_quads_and_calculate(result.x, oe, col, shape_data)
    if(col.program == 'mirror'):
        col.raytrace_from_saved_values()
    if(method=='Nelder-Mead' and options.get('return_all') == True):
        np.save(oe.filename_noext+'_all_solns', result['allvecs'])

def optimize_image_plane(oe, min_dist=3, image_plane=6):
    '''
    In principle, could be used to find the optimal image plane to minimize
    spherical aberration. In practice, that is always zero distance from the
    object plane, so needs to be rewritten to be useful.
    '''
    initial_plane = [image_plane] # mm
    bounds = [(min_dist,100)]
    result = minimize(change_imgplane_and_calculate, initial_plane, args=(oe), bounds=bounds, method='TNC', 
                      options={'eps':0.5,'stepmx':5,'minfev':1})
    change_imgplane_and_calculate(result.x, oe)
    print('Optimization complete')

