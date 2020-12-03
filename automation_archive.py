class Point(object):
    def __init__(self,z,r):
        self.z = z
        self.r = r

    def print(self):
        print(f"z: {self.z}, r: {self.r}")

    def __eq__(self,other):
        return self.__dict__ == other.__dict__
        
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


