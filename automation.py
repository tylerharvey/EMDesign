#!/usr/bin/env python3
'''
Contains only user methods; everything else is in automation_library.py.
User methods:
    optimize_many_shapes
    optimize_broadly_for_retracing
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from calculate_optical_properties import calc_properties_optics
from automation_library import change_image_plane_and_check_retracing, \
                               change_voltages_and_shape_and_check_retracing, \
                               change_column_and_calculate_mag, \
                               change_n_quads_and_calculate, \
                               change_n_quads_and_calculate_curr, \
                               determine_lens_pos_bounds, \
                               determine_img_pos_bounds, \
                               generate_initial_simplex, \
                               prepare_shapes, \
                               change_n_quads_and_check, \
                               TimeoutCheck
from misc_library import Logger

def optimize_broadly_for_retracing(
        oe, col, potentials, img_pos, z_indices_list=None, r_indices_list=None, 
        other_z_indices_list=None, other_r_indices_list=None, z_curv_z_indices_list=None, z_curv_r_indices_list=None, 
        r_curv_z_indices_list=None, r_curv_r_indices_list=None, end_z_indices_list=None, end_r_indices_list=None, 
        z_min=None, z_max=None, r_min=0, r_max=None, breakdown_field=10e3, 
        options={'adaptive':True,'fatol':0.00001,'disp':True,'return_all':True}, enforce_smoothness=False,
        simplex_scale=4, curve_scale=0.05, voltage_logscale=2, max_r_to_edit=None, optimize_end_voltage=False,
        end_voltage_bounds=None, **kwargs):
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
        potentials : MirPotentials object
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
        enforce_smoothness : bool
            enforce a maximum angle at corners.
            default False.
        simplex_scale : float
            size (in mm) for normal distribution of simplex points around
            initial shape. only used with Nelder-Mead. A larger scale results
            in a longer search that is more likely to find a qualitatively
            different shape.
            default 3.
        curve_scale : float
            curvature (in 1/mm) for normal distribution of simplex points
            around initial shape. only used with Nelder Mead. 68% of generated
            curvatures will have a radius of more than 1/curve_scale.
            default 0.05, so a radius of 20mm.
        voltage_logscale : float
            scale for log-normal distribution of voltage points around initial
            configuaration. units are np.log(Volts).
            default 2.
        max_r_to_edit : float
            maximum radius to include in optimization.
        optimize_end_voltage : bool
            flag indicating whether end voltage is included in optimization.
            default False.
        end_voltage_bounds : list
            length 2 list of floats to bound the end electrode voltage in optimization.
            default None.
        **kwargs 
            used to pass additional kwargs to write_raytrace_file().

    '''
    if(end_voltage_bounds == None and optimize_end_voltage):
        end_voltage_bounds = [-1000,-50]
    options_mutable = options.copy()

    initial_shape, bounds, shape_data = prepare_shapes(
        oe, z_indices_list, r_indices_list, other_z_indices_list, other_r_indices_list, 
        z_curv_z_indices_list, z_curv_r_indices_list, r_curv_z_indices_list, r_curv_r_indices_list,
        end_z_indices_list, end_r_indices_list, z_min, z_max, r_min, r_max, 
        automate_present_curvature=False, max_r_to_edit=max_r_to_edit)

    potentials.voltages = np.array(potentials.voltages) 
    flag_mask = np.array(potentials.flags) != 'f'
    voltages = potentials.voltages[flag_mask]
    if(optimize_end_voltage):
        end_voltage = potentials.voltages[0]
        if(potentials.voltages[0] != potentials.voltages[1]):
            raise ValueError('End voltages not equal and cannot be optimized by existing routine.')

    img_pos_soft_bounds, img_pos_hard_bounds = determine_img_pos_bounds(oe,col)

    # gather positions of all lenses
    lens_pos_list = [oe.lens_pos for oe in col.oe_list if hasattr(oe,'lens_pos')]
    # exclude mirror, which should be first element in oe_list
    other_lens_pos_list = lens_pos_list[1:]
    initial_parameters = initial_shape + [end_voltage]*optimize_end_voltage + \
                         voltages.tolist() + [img_pos] + other_lens_pos_list
    N = len(initial_parameters)
    shape_data.n_end_voltages = 1 if optimize_end_voltage else 0
    shape_data.n_voltages = len(voltages)
    shape_data.n_img_pos = 1
    shape_data.n_lens_pos = len(other_lens_pos_list)

    
    # generate shape simplex
    if(options.get('initial_simplex') is None):
        options_mutable['initial_simplex'] = generate_initial_simplex(
            initial_shape, oe, shape_data, enforce_bounds=True, bounds=np.array(bounds), 
            breakdown_field=breakdown_field, scale=simplex_scale, curve_scale=curve_scale, 
            enforce_smoothness=enforce_smoothness, adaptive=True, N=N)

    rng = np.random.default_rng()
    voltage_simplex = np.zeros((N+1,shape_data.n_voltages), dtype=float)
    img_pos_simplex = np.zeros((N+1,shape_data.n_img_pos), dtype=float)
    if(optimize_end_voltage):
        end_voltage_simplex = np.zeros((N+1,shape_data.n_end_voltages),dtype=float)
    if(shape_data.n_lens_pos):
        lens_pos_simplex = np.zeros((N+1,shape_data.n_lens_pos),dtype=float)
        lens_pos_bounds_low,lens_pos_bounds_high = determine_lens_pos_bounds(oe,col)
    for i in range(N+1):
        voltage_simplex[i] = np.exp(rng.normal(np.log(voltages-voltages.min()+1000), voltage_logscale)) \
                             + voltages.min()-1000
        img_pos_simplex[i,:] = rng.uniform(*img_pos_hard_bounds)
        if(optimize_end_voltage):
            end_voltage_simplex[i,:] = rng.uniform(*end_voltage_bounds)
        if(shape_data.n_lens_pos):
            lens_pos_simplex[i,:] = rng.uniform(lens_pos_bounds_low,lens_pos_bounds_high)

    index_0 = shape_data.n_pts
    index_1 = index_0 + shape_data.n_end_voltages 
    try:
        options_mutable['initial_simplex'][:,index_0:index_1] = end_voltage_simplex
    except NameError:
        pass
    index_0 = index_1
    index_1 = index_0 + shape_data.n_voltages
    options_mutable['initial_simplex'][:,index_0:index_1] = voltage_simplex
    index_0 = index_1
    index_1 = index_0 + shape_data.n_img_pos
    options_mutable['initial_simplex'][:,index_0:index_1] = img_pos_simplex
    index_0 = index_1
    index_1 = index_0 + shape_data.n_lens_pos
    try:
        options_mutable['initial_simplex'][:,index_0:index_1] = lens_pos_simplex
    except NameError:
        pass

    oe.automated = True
    result = minimize(change_voltages_and_shape_and_check_retracing, initial_parameters,
                      args=(oe, col, potentials, flag_mask, shape_data, bounds, img_pos_soft_bounds, 
                            breakdown_field, enforce_smoothness, optimize_end_voltage, kwargs), 
                      method='Nelder-Mead', options=options_mutable)
    ilog = Logger('internal')
    ilog.log.debug(f'Optimize {result=}')

    if(options.get('return_all') == True):
        np.save(oe.filename_noext+'_all_solns', result['allvecs'])

    change_voltages_and_shape_and_check_retracing(result.x,oe,col,potentials,flag_mask,
                                                  shape_data,bounds, img_pos_soft_bounds,
                                                  None,False,optimize_end_voltage,kwargs)
    index_0 = shape_data.n_pts
    index_1 = index_0 + shape_data.n_end_voltages 
    potentials.voltages[:2] = result.x[index_0:index_1]
    index_0 = index_1
    index_1 = index_0 + shape_data.n_voltages
    potentials.voltages[flag_mask] = result.x[index_0:index_1]
    index_0 = index_1
    index_1 = index_0 + shape_data.n_img_pos
    img_pos = result.x[index_0:index_1]
    index_0 = index_1
    index_1 = index_0 + shape_data.n_lens_pos
    other_lens_pos_list = result.x[index_0:index_1]
    for i,oe in enumerate(col.oe_list[1:]):
        oe.lens_pos = other_lens_pos_list[i]

    potentials.voltages = potentials.voltages.tolist()

    col.write_mir_img_cond_file(col.mircondfilename, potentials=potentials,
                                source_pos=img_pos-col.img_source_offset, img_pos=img_pos,
                                **kwargs)
    # col.write_raytrace_file(col.mircondfilename, potentials=potentials,
    #                         source_pos=img_pos-col.img_source_offset, screen_pos=img_pos,
    #                         minimum_rays=True, **kwargs)
    col.calc_rays()
    if(oe.plot):
        col.plot_rays()

def optimize_many_shapes(
        oe, col, z_indices_list, r_indices_list, other_z_indices_list=None, other_r_indices_list=None,
        z_curv_z_indices_list=None, z_curv_r_indices_list=None, r_curv_z_indices_list=None, r_curv_r_indices_list=None, 
        end_z_indices_list=None, end_r_indices_list=None, z_min=None, z_max=None, r_min=0, r_max=None, c3_target=0,
        automate_present_curvature=False, method='Nelder-Mead', manual_bounds=True, 
        options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None,'return_all':True}, 
        simplex_scale=5, curve_scale=0.05, curr_bound=3, breakdown_field=10e3, enforce_smoothness=False,
        adaptive_simplex=True):
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
        curve_scale : float
            curvature (in 1/mm) for normal distribution of simplex points
            around initial shape. only used with Nelder Mead. 68% of generated
            curvatures will have a radius of more than 1/curve_scale.
            default 0.05, so a radius of 20mm.
        curr_bound : float
            bound for maximum current desnity in first magnetic lens. 
            default 3 A/mm^2 current density limit
        breakdown_field : float
            field at which vacuum breakdown is possible, in V/mm.
            Default 10,000 V/mmm.
        enforce_smoothness : bool
            enforce a maximum angle at corners.
            default False.
        adaptive_simplex : bool
            if True, adjusts the simplex scale for each point according to an 
            initial search. if False, purely normal distribution of simplex 
            points.
            default True.
    '''
    options_mutable = options.copy()

    initial_shape, bounds, shape_data = prepare_shapes(
        oe, z_indices_list, r_indices_list, other_z_indices_list, other_r_indices_list, 
        z_curv_z_indices_list, z_curv_r_indices_list, r_curv_z_indices_list, r_curv_r_indices_list, 
        end_z_indices_list, end_r_indices_list, z_min, z_max, r_min, r_max, automate_present_curvature)

    if(change_n_quads_and_check(np.array(initial_shape), oe, shape_data, enforce_bounds=True, bounds=bounds, 
            breakdown_field=breakdown_field, enforce_smoothness=enforce_smoothness)):
        raise ValueError('Initial shape intersects or violates bounds.')
    oe.automated = True
    olog = Logger('output')
    ilog = Logger('internal')
    if(method=='Nelder-Mead' and options.get('initial_simplex') is None):
        olog.log.info('Generating initial simplex.')
        options_mutable['initial_simplex'] = generate_initial_simplex(
            initial_shape, oe, shape_data, enforce_bounds=True, bounds=np.array(bounds), 
            breakdown_field=breakdown_field, scale=simplex_scale, curve_scale=curve_scale, 
            enforce_smoothness=enforce_smoothness, adaptive=adaptive_simplex)
        olog.log.info('Finished initial simplex generation.')
    if(manual_bounds):
        if(oe.lens_type == 'magnetic'):
            result = minimize(change_n_quads_and_calculate, initial_shape, args=(oe, col, shape_data, TimeoutCheck(), 
                              True, np.array(bounds), curr_bound, None, enforce_smoothness, c3_target), 
                              method=method, options=options_mutable)
        elif(oe.lens_type == 'electrostatic'):
            result = minimize(change_n_quads_and_calculate, initial_shape, args=(oe, col, shape_data, TimeoutCheck(), 
                              True, np.array(bounds), None, breakdown_field, enforce_smoothness, c3_target), 
                              method=method, options=options_mutable)
    else:
        raise NotImplementedError
        # no longer actively maintaining this
        result = minimize(change_n_quads_and_calculate, initial_shape,
                          args=(oe, col, shape_data, TimeoutCheck()),
                          bounds=bounds, method=method, options=options_mutable)

    olog.log.info('Optimization complete with success flag {}'.format(result.success))
    ilog.log.debug(result.message)
    change_n_quads_and_calculate(result.x, oe, col, shape_data)
    if(col.program == 'mirror'):
        col.raytrace_from_saved_values()
    if(method=='Nelder-Mead' and options.get('return_all') == True):
        np.save(oe.filename_noext+'_all_solns', result['allvecs'])

