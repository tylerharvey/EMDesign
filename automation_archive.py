from skopt import gbrt_minimize, gp_minimize, dummy_minimize, forest_minimize

class Point(object):
    def __init__(self,z,r):
        self.z = z
        self.r = r

    def print(self):
        print(f"z: {self.z}, r: {self.r}")

    def __eq__(self,other):
        return self.__dict__ == other.__dict__
        
def optimize_image_plane(oe, min_dist=3, image_plane=6):
    '''
    In principle, could be used to find the optimal image plane to minimize
    spherical aberration. In practice, that is always zero distance from the
    object plane, so needs to be rewritten to be useful.
    '''
    initial_plane = [image_plane] # mm
    bounds = [(min_dist,100)]
    oe.automated = True
    result = minimize(change_imgplane_and_calculate, initial_plane, args=(oe), bounds=bounds, method='TNC', 
                      options={'eps':0.5,'stepmx':5,'minfev':1})
    change_imgplane_and_calculate(result.x, oe)
    print('Optimization complete')

def change_imgplane_and_calculate(imgplane, oe):
    oe.write_opt_img_cond_file(oe.imgcondfilename, img_pos=imgplane[0])
    calc_properties_optics(oe)
    try: 
        oe.read_optical_properties()
    except UnboundLocalError: # if optics fails, return garbage
        return 100
    print(f"f: {col.f}, C3: {col.c3}")
    return np.abs(col.c3)

# always returns true with present definition of fine mesh
# define_exhaustive would make this work
def does_fine_mesh_intersect(oe):
    return intersections_in_segment_list(oe.define_fine_mesh_segments())

# makes the smallest possible fine mesh segments
# not useful except for finding fine-fine intersections
# curvature not implemented
def define_exhaustive_fine_mesh_segments(self):
    segments = []
    r_interpolator = interp2d(self.z_indices,self.r_indices,self.r)
    z_interpolator = interp2d(self.z_indices,self.r_indices,self.z)
    # MEBS uses half-integer steps for the fine mesh
    step = 0.5
    for r_index in np.arange(self.r_indices[0],self.r_indices[-1]+step,step):
        for z_index in np.arange(self.z_indices[0],self.z_indices[-1]+step,step):
            if(r_index < self.r_indices[-1]):
                segments.append((Point(z_interpolator(z_index,r_index),r_interpolator(z_index,r_index)),Point(z_interpolator(z_index,r_index+step),r_interpolator(z_index,r_index+step))))
            if(z_index < self.z_indices[-1]):
                segments.append((Point(z_interpolator(z_index,r_index),r_interpolator(z_index,r_index)),Point(z_interpolator(z_index+step,r_index),r_interpolator(z_index+step,r_index))))
    self.fine_segments = segments
    return segments

#######
# outdated functions
# not presently used because checking all coarse mesh intersections does this
# fully functional right now
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

# fully functional, only called by do_quads_intersect_anything
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
    return calculate_c3(oe,t=TimeoutCheck())

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
        return calculate_c3(self.oe,t=TimeoutCheck())

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

    
def change_current_and_calculate(current,oe,col):
    oe.coil_curr = current
    return calculate_c3(oe,col,t=TimeoutCheck())

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

def optimize_shape_for_current(
        oe, col, z_indices_list, r_indices_list, other_z_indices_list=None, other_r_indices_list=None,
        z_curv_z_indices_list=None, z_curv_r_indices_list=None, r_curv_z_indices_list=None, r_curv_r_indices_list=None, 
        end_z_indices_list=None, end_r_indices_list=None, z_min=None, z_max=None, r_min=0, r_max=None, 
        automate_present_curvature=False, method='Nelder-Mead', manual_bounds=True, 
        options={'disp':True,'xatol':0.01,'fatol':0.001,'adaptive':True,'initial_simplex':None,'return_all':True}, 
        simplex_scale=5, curve_scale=0.05, curr_bound=3, breakdown_field=10e3, adaptive_simplex=True):
    '''
    Not actively maintained. General funcion necessary.

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
        oe, z_indices_list, r_indices_list, other_z_indices_list, other_r_indices_list, 
        z_curv_z_indices_list, z_curv_r_indices_list, r_curv_z_indices_list, r_curv_r_indices_list, 
        end_z_indices_list, end_r_indices_list, z_min, z_max, r_min, r_max, automate_present_curvature)

    if(change_n_quads_and_check(np.array(initial_shape), oe, shape_data, enforce_bounds=True, bounds=bounds, 
            breakdown_field=breakdown_field)):
        raise ValueError('Initial shape intersects or violates bounds.')
    oe.automated = True
    olog = Logger('output')
    ilog = Logger('internal')
    if(method=='Nelder-Mead' and options.get('initial_simplex') is None):
        olog.log.info('Generating initial simplex.')
        options_mutable['initial_simplex'] = generate_initial_simplex(
            initial_shape, oe, shape_data, enforce_bounds=True, bounds=np.array(bounds), 
            breakdown_field=breakdown_field, scale=simplex_scale, curve_scale=curve_scale, adaptive=adaptive_simplex)
        olog.log.info('Finished initial simplex generation.')
    if(manual_bounds):
        result = minimize(change_n_quads_and_calculate_curr, initial_shape, args=(oe, col, shape_data, TimeoutCheck(), 
                          True, np.array(bounds), curr_bound), method=method, options=options_mutable)
    else:
        result = minimize(change_n_quads_and_calculate_curr, initial_shape,
                          args=(oe, col, shape_data, TimeoutCheck()),
                          bounds=bounds, method=method, options=options_mutable)

    olog.log.info('Optimization complete with success flag {}'.format(result.success))
    ilog.log.debug(result.message)
    change_n_quads_and_calculate_curr(result.x, oe, col, shape_data)
    if(col.program == 'mirror'):
        col.raytrace_from_saved_values()
    if(method=='Nelder-Mead' and options.get('return_all') == True):
        np.save(oe.filename_noext+'_all_solns', result['allvecs'])

def optimize_column_for_mag(col,img_pos=50,options=None, lens_pos_min=0, lens_pos_max=2000, 
                            curr_bound=3, img_pos_min=0, img_pos_max=2000, **kwargs):
    '''
    Needs further testing. 

    Automatic optimization of magnification of a set of optical elements.

    Parameters:
        col : OpticalColumn object
            col must be initialized as a multi-element column with at least
            two optical elements. The elements must have values for flags
            oe.pos_adj and oe.f_adj, which determine if element position and 
            element focal length are adjusted, and initial values for 
            oe.lens_pos and oe.lens_excitation.
    
    Optional parameters:
        options : dict
            Passed to minimize(). 
            Default None.
        **kwargs
            kwargs passed to write_mir_img_cond_file and write_raytrace_file.
    '''


    bounds = []
    col_vars = []
    i = 0
    for oe in col.oe_list:
        if(oe.pos_adj == True):
            col_vars.append(oe.lens_pos)
            bounds.append((lens_pos_min,lens_pos_max))
            i += 1
        if(oe.f_adj == True):
            col_vars.append(oe.lens_excitation)
            if(oe.lens_type == 'magnetic'):
                coil_area = 0
                for j in range(len(oe.coil_z_indices)):
                    coil_area += oe.determine_quad_area(oe.coil_z_indices[j], oe.coil_r_indices[j])
                total_curr_bound = coil_area*curr_bound # curr_bound is current density
                bounds.append((-total_curr_bound,total_curr_bound))
            i += 1
        oe.calc_field()


    col_vars.append(img_pos)
    bounds.append((img_pos_min,img_pos_max))

    col.write_opt_img_cond_file(col.imgcondfilename, img_pos=img_pos, **kwargs)
    calc_properties_optics(col)
    col.read_optical_properties()
    olog = Logger('output')
    olog.log.info(f'Initial_magnification: {col.mag}')

    result = minimize(change_column_and_calculate_mag, col_vars, 
                      args=(col,np.array(bounds),kwargs), method='Nelder-Mead', options=options)
    olog.log.info(f'Resulting magnification: {1/change_column_and_calculate_mag(result.x,col,np.array(bounds),kwargs)}')
    col.write_opt_img_cond_file(col.imgcondfilename, img_pos=result.x[-1], **kwargs)
    # col.write_mir_img_cond_file(col.mircondfilename, **kwargs)
    # col.write_raytrace_file(col.mircondfilename, **kwargs)

def optimize_planes_for_retracing(col, bounds=(0,200), img_pos=90, **kwargs):
    result = minimize_scalar(change_image_plane_and_check_retracing, args=(col, kwargs),  
                             method='bounded', bounds=bounds)

    olog = Logger('output')
    olog.log.info(f"Retracing image plane: {result.x}mm.")
    img_pos = result.x

    col.write_mir_img_cond_file(col.mircondfilename, source_pos=img_pos-col.img_source_offset, 
                                img_pos=img_pos, **kwargs)

    col.write_raytrace_file(col.mircondfilename, source_pos=img_pos-col.img_source_offset,
                            screen_pos=img_pos, minimum_rays=True, **kwargs)


def optimize_voltages_for_retracing(col, potentials, img_pos, options=None, **kwargs):
    potentials.voltages = np.array(potentials.voltages) 
    flag_mask = np.array(potentials.flags) != 'f'
    voltages = potentials.voltages[flag_mask]
    initial_parameters = np.append(voltages, img_pos) # [v_1, ... , v_n, img_pos]
    result = minimize(change_voltages_and_check_retracing, initial_parameters,
                      args=(col, potentials, flag_mask, kwargs),
                      method='Nelder-Mead', options=options)
    ilog = Logger('internal')
    ilog.log.debug(f'Optimize {result=}')
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

