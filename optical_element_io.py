#!/usr/bin/env python3
import os, subprocess, shutil
from subprocess import TimeoutExpired
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from scipy.interpolate import interp2d
from shapely.geometry import *
from misc_library import Logger, \
                         cd, index_array_from_list, np_index, last_np_index, \
                         check_len, check_len_multi 

# definitions for comments:
# quad : four-pointed object used to define magnetic materials, coils, electrodes, etc. in MEBS

### note: MEBS binaries are extremely picky about file paths
### in order for this code to successfully call them, the following
### folders must be added to the Windows path:
### < ... >\MEBS\OPTICS\bin\
### < ... >\MEBS\SOFEM\bin\CL\
### < ... >\MEBS\MIRROR\bin\MIRROR\
### < ... >\MEBS\HERM1\bin\CL\
### to edit the Windows path, right click on This PC and then select Properties
### On the left, click Avdanced System Settings, and Environment Variables
### at the bottom. Choose PATH in the User variables list and then click Edit 
### 
### the working directory must also be the one in which the MEBS .dat and .axb
### files are located, so os.chdir is used after read-in satisfy this.

class MEBSSegment:
    
    def __init__(self, point_a, point_b, curvature=0, prev_segment=None, reverse=False):
        if(prev_segment):
            if(reverse):
                self.shape = LineString(prev_segment.shape.coords[::-1]) 
            else: 
                self.shape = Linestring(prev_segment.shape)
            self.arc = prev_segment.arc
        else:
            self.point_a = point_a
            self.point_b = point_b
            if(curvature and curvature != np.inf):
                self.arc = True
                self.radius = np.abs(curvature)
                curv_sign = np.sign(curvature)
                center_angle = np.arccos(0.5*self.point_a.distance(self.point_b)/self.radius)
                if(np.isnan(center_angle)):# no intersection
                    raise ValueError(f'Curvature smaller than half the distance between points. '
                                       'Point a: {self.point_a}, point b: {self.point_b}, distance: '
                                       '{self.point_a.distance(self.point_b)}, radius: {self.radius}.') 
                rel_angle = np.arctan2(self.point_b.y-self.point_a.y,self.point_b.x-self.point_a.x)
                tot_angle = rel_angle-curv_sign*center_angle
                # curvature > 0 means point_A-center vector is at a negative angle w.r.t point_a-point_b vector
                self.center = Point(self.point_a.x+self.radius*np.cos(tot_angle),self.point_a.y
                                    +self.radius*np.sin(tot_angle))
                theta_a = np.arctan2(self.point_a.y-self.center.y,self.point_a.x-self.center.x)
                theta_b = np.arctan2(self.point_b.y-self.center.y,self.point_b.x-self.center.x)
                if(theta_a-theta_b > np.pi):
                    theta_b += 2*np.pi
                elif(theta_b-theta_a > np.pi):
                    theta_b -= 2*np.pi
                theta_range = np.linspace(theta_a,theta_b,101,endpoint=True)
                # theta_a = theta_a + 2*np.pi if theta_a < 0 else theta_a
                # theta_b = theta_b - 2*np.pi if theta_b > 0 else theta_b
                # if(curvature > 0):
                #     theta_range = np.linspace(theta_a,theta_b,100,endpoint=True)
                # else:
                #     theta_range = np.linspace(theta_b,theta_a,100,endpoint=True)
                np_arc = np.zeros((len(theta_range),2),dtype=float)
                np_arc[:,0] = self.center.x+self.radius*np.cos(theta_range)
                np_arc[:,1] = self.center.y+self.radius*np.sin(theta_range)
                self.shape = asLineString(np_arc)
            else:
                self.arc = False
                self.shape = LineString([self.point_a,self.point_b])



class OpticalElement:
    '''
    Class for a wide range of optical elements. 
    Specific elements are implemented as subclasses. 
    
    Attributes:
        title : string
            title printed as the first line of the .dat file.
        output : integer
            integer flag that controls the field data outputted by MEBS.
            0 outputs data only along the z axis.
            1 outputs data at all FE mesh points.
            2 outputs data at all mesh points and inside magnetic circuits.
        axsym : integer
            boolean flag used by MEBS to denote whether the element has
            reflection symmetry of the element through the x-y plane 
            (between positive and negative z), with 1 true and 0 false.
        r_indices : integer ndarray
            length N 1D array of the radial indices of the FE mesh. first index
            is 1. (denoted I in MEBS manual.) skipped values are interpolated 
            in MEBS.
        z_indices : integer ndarray
            length M 1D array of the axial indices of the FE mesh. first index
            is 1.  (denoted J in MEBS manual.) skipped values are interpolated
            in MEBS.
        r : float ndarray
            NxM 2D array of radial coordinates defined on the grid of all 
            explicit (r_indices,z_indices) points (i.e. defined on the coarse
            mesh).
        z : float ndarray
            NxM 2D array of axial coordinates defined on the grid of all
            explicit (r_indices,z_indices) points (i.e. defined on the coarse
            mesh).
        filename : path
            full filename of optical element.
        basename_noext : path
            name without directories or extension
        basename : path
            name without directories
        potname : path
            name of potential file
        filename_noext : path
            filename with directories and no extension
        dirname : path
            directory that contains optical element .dat file.
        mirror : bool
            determines whether HERM1 will be run with 'AN' symmetry. starts as
            False (i.e. 'NN' symmetry) and set by mirror_type().
        curved_mirror : bool
            determines whether HERM1 will be run with curved mirror correction.
            starts as False and set by mirror_type().
        freeze_xy_plane : bool
            determines whether the z coordinate for points initially along z=0
            is allowed to move in optimization. starts as False; set to False
            by mirror_type() if curved_mirror = True. set to True if 
            oe.axsym==1.
        freeze_radial_boundary : bool
            determines whether the r coordinate for points initially along 
            r=r_max is allowed to move in optimization. starts as False. set to
            True by mirror_type() regardless of arguments. may need to be
            manually set to True for some optical elements.

    User methods:
        plot_mesh_coarse
        plot_mesh_fine
        plot_field
        add_curvature (converts optical element .dat file for FOFEM use)

    Hard-coded attributes:
        colwidth : int
            width of columns to use in the optical element file.
            default 12.
        timeout : float
            seconds to wait before killing MEBS programs.
            default 3 minutes.
    '''
    
    colwidth = 12 # width of columns in written .dat files
    sp = " "*colwidth 
    int_fmt = Template("{:${colwidth}d}").substitute(colwidth=colwidth)
    float_fmt = Template("{:${colwidth}.${precision}g}")
    rfloat_fmt = Template("{:>${imgcondcolwidth}.${precision}g}")
    timeout = 3*60 # 3 minutes
    automated = True # uninitialized value
    lens_scale = 1
    potentials = None
    lens_excitation = None
    excitation_flag = None
    

    # so reads .dat files for the second-order solver
    # these are almost exactly the same except they have radii of curvature
    # sections after the mesh that looks like the mesh blocks
    def __init__(self, filename='', so=False, plot=False):
        '''
        Initialize an instance of an optical element.

        Parameters:
            filename : path
                full filename pointing to the optical element .dat file.

        Optional flags:
            so : boolean
                denotes whether the optical element .dat file includes 
                curvature coordinates for compatibility with MEBS SOFEM.
                (OPTICS does not use curvature.) flag will automatically 
                flip to True if curvature coordinates are found.
                default False
            plot : boolean
                determines whether plots are shown. 
                default False for non-blocking operation.
        '''

        self.Mlog = Logger('MEBS')
        self.olog = Logger('output')
        self.so=so
        self.mirror = False
        self.curved_mirror = False
        self.freeze_xy_plane = False
        self.freeze_radial_boundary = False
        self.infile = []
        self.initialize_lists()
        self.plot = plot
        self.automated = False # will be turned on when automation starts
        if(filename):
            self.read(filename)
            shutil.copyfile(filename, filename+'.bak')
            
    def initialize_lists(self):
        pass
    
    def read(self, filename):
        f = open(filename,'r')
        self.filename = filename
        self.basename_noext = os.path.splitext(os.path.basename(filename))[0] # name without directories or extension
        self.basename = os.path.basename(filename) # name without directories
        self.fitname = self.basename_noext+'.fit' # name of potential file
        self.filename_noext = os.path.splitext(filename)[0] # with directories
        self.dirname = os.path.dirname(filename)
        self.infile = list(f) # creates a list with one entry per line of f
        self.read_intro()
        self.olog.log.info(f"Reading file {filename} \nwith title: {self.title}")
        line_num = self.read_mesh()
        if(self.so):
            line_num = self.read_curvature(line_num)
        self.read_other_blocks(line_num)
        f.close()
        f = None

    def read_intro(self):
        self.title = self.infile[0].strip('\n')
        self.output, self.axsym = np.fromstring(self.infile[1],dtype=int,count=2,sep=' ')
        if(self.axsym == 1):
            self.freeze_xy_plane = True
    
    def read_mesh(self):
        self.z_indices = np.fromstring(self.infile[3],dtype=int,sep=' ')
        r_indices = []
        z = []
        r = []
        line_num = 4 # first line of z coordinates
        while(self.infile[line_num].isspace() != True):
            r_indices.append(np.fromstring(self.infile[line_num],dtype=int,count=1,sep=' ')[0])
            z.append(np.fromstring(self.infile[line_num],dtype=float,sep=' ')[1:]) # exclude r index
            line_num+=1
        self.r_indices  = np.array(r_indices)
        self.z = np.array(z)
        line_num+=1 # skip blank line we just found
        z_indices_2 = np.fromstring(self.infile[line_num],dtype=int,sep=' ')
        if(np.array_equal(z_indices_2,self.z_indices) != True):
            raise ValueError("Read error! z indices read in first and second mesh block not identical!")
        line_num+=1 # move to r coordinates
        r_indices_2 = []
        while(self.infile[line_num].isspace() != True):
            r_indices_2.append(np.fromstring(self.infile[line_num],dtype=int,count=1,sep=' ')[0])
            r.append(np.fromstring(self.infile[line_num],dtype=float,sep=' ')[1:]) # exclude r index
            line_num+=1
        self.r = np.array(r)
        if(np.array_equal(r_indices,r_indices_2) != True):
            raise ValueError("Read error! r indices read in first and second mesh block not identical!")
        line_num+=1 # skip blank line we just found
        if(self.so == False and 
                (len(self.z_indices) != 5 and len(np.fromstring(self.infile[line_num],dtype=int,sep=' ')) != 5) or 
                (len(self.z_indices) == 5 and 
                    np.array_equal(np.fromstring(self.infile[line_num],dtype=int,sep=' '),self.z_indices))):
            self.olog.log.info('Warning! This data file seems to have curvature coordinates. Setting so=True.')
            self.so = True
        return line_num # save line number of the start of the next block
    
    def read_curvature(self, line_num):
        ## curvature intro
        # for z = | z_A z_B |
        #         | z_C z_D |
        # and r = | r_A r_B |
        #         | r_C r_D |
        # and z_curv = | zc_A zc_B |
        #              | zc_C zc_D |
        # and r_curv = | rc_A rc_B |
        #              | rc_C rc_D |
        # the value zc_A sets the curvature for the segment A-C
        # where zc_A > 0 puts the origin of the circle connecting
        # points A and C at a lower z value than z_A and z_C
        # so, in normal z-r plots, A-C bows out to the right
        # the value rc_A sets the curvature for the segment A-B
        # where rc_A > 0 puts the origin of the circle connecting
        # points A and B at a lower r value than r_A and r_B
        # so, A-B bows upwards
        # (assumes normal MEBS ordering, where z_B > z_A and r_A > r_C)
        # zc_C, zc_D, rc_B and rc_D do nothing
        z_curv = [] # radii of curvature of the left and right sides of quads
        r_curv = [] # radii of curvature of the top and bottom sides of quads
        line_num+=1 # move past z indices line
        while(self.infile[line_num].isspace() != True):
            z_curv.append(np.fromstring(self.infile[line_num],dtype=float,sep=' ')[1:]) # exclude r index
            line_num+=1
        self.z_curv = np.array(z_curv)
        line_num+=1 # skip blank line we just found
        line_num+=1 # move past z indices line
        while(self.infile[line_num].isspace() != True):
            r_curv.append(np.fromstring(self.infile[line_num],dtype=float,sep=' ')[1:]) # exclude r index
            line_num+=1
        self.r_curv = np.array(r_curv)
        line_num+=1 # skip blank line we just found
        return line_num # save line number of the start of the next block
    
    # generic read function for magnetic materials, coils, electrodes, etc.
    def read_quad(self, line_num, z_indices, r_indices, quad_property, property_dtype=float):
        while(self.infile[line_num].isspace() != True):
            indices = np.fromstring(self.infile[line_num],dtype=int,count=4,sep=' ')
            z_indices.append(indices[:2])
            r_indices.append(indices[2:4])
            quad_property.append(np.fromstring(self.infile[line_num],dtype=property_dtype,count=5,sep=' ')[-1])
            line_num+=1
        return line_num+1 # start of next block
    
    def write(self, outfilename, title=None, coord_precision=6, curr_precision=6, 
              field_precision=6, rel_perm_precision=6, voltage_precision=6):
        if not title:
            title = self.title
        else:
            self.title = title

        self.coord_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=coord_precision)
        self.curr_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=curr_precision)
        self.field_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=field_precision)
        self.rel_perm_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=rel_perm_precision)
        self.voltage_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=voltage_precision)
        
        f = open(outfilename,'w')
        self.filename = outfilename
        self.basename_noext = os.path.splitext(os.path.basename(outfilename))[0]
        self.basename = os.path.basename(outfilename)
        self.potname = self.basename_noext+os.path.splitext(self.potname)[1]
        self.fitname = self.basename_noext+'.fit' # name of potential file
        self.filename_noext = os.path.splitext(outfilename)[0]
        self.dirname = os.path.dirname(outfilename)
        
        self.write_intro(f, title)
        self.write_coordinates(f, self.z)
        self.write_coordinates(f, self.r)
        if(self.so):
           self.write_coordinates(f, self.z_curv)
           self.write_coordinates(f, self.r_curv)
        self.write_other_blocks(f)
        f.close()
        f = None
        
    def write_intro(self, f, title):
        f.write(title+"\n")
        f.write(check_len(self.int_fmt.format(self.output),self.colwidth))
        f.write(check_len(self.int_fmt.format(self.axsym),self.colwidth))
        f.write("\n\n") # one blank line after intro

    def write_coordinates(self, f, coords, rounding=True):
        f.write(self.sp)
        # I use i and j here explicitly for clarity as 
        # this usage is consistent with the MEBS manual I and J
        for j in range(len(self.z_indices)):
            f.write(check_len(self.int_fmt.format(self.z_indices[j]),self.colwidth))
        # equivalent: f.write(self.check_len((int_fmt*len(self.z_indices)).format(*self.z_indices)))
        f.write("\n")
        for i in range(len(self.r_indices)):
            f.write(check_len(self.int_fmt.format(self.r_indices[i],self.colwidth),self.colwidth))
            for j in range(len(self.z_indices)):
                coords_to_print = round(coords[i,j],self.colwidth-5) if rounding else coords[i,j]
                f.write(check_len(self.coord_fmt.format(coords_to_print),self.colwidth))
            f.write("\n")
        f.write("\n")
        
    def write_quad(self, f, z_indices, r_indices, quad_property, property_fmt):
        N = len(z_indices)
        for n in range(N):
            f.write(self.autoformat_multi((self.int_fmt*2),z_indices[n]))
            f.write(self.autoformat_multi((self.int_fmt*2),r_indices[n]))
            f.write(check_len(property_fmt.format(quad_property[n]),self.colwidth))
            f.write("\n")
        f.write("\n")
        
    # these are dummy functions for the main class that are replaced 
    # by datafile-specific versions in the subclasses
    def read_other_blocks(self, line_num): 
        pass

    def write_other_blocks(self, f):
        pass
        
    def plot_mesh_fine(self, quads_on=True, index_labels=True, adj=6, zlim=None, rlim=None):
        '''
        Plots fine mesh.

        Optional parameters:
            quads_on : boolean
                determines whether magnetic materials, coils, electrodes, etc.
                are also plotted.
                default True
            index_labels : boolean
                determines whether mesh indices are labeled.
            adj : float 
                offset of mesh labels from matplotlib axes.
                default 6
            zlim : tuple or list of two floats
                sets matplotlib xlim to zoom in on region of the z axis.
                generally looks horrible with index_labels=True.
            rlim : tuple or list of two floats
                sets matplotlib ylim to zoom in on region of th r axis.
                generally looks horrible with index_labels=True.
        '''
        plt.figure(figsize=(15,15))
        self.add_fine_mesh_to_plot()
        self.add_quads_to_plot() if quads_on else 0
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
        plt.xlim(zlim)
        plt.ylim(rlim)
        plt.gca().set_aspect('equal')
        plt.show()

    def plot_mesh_coarse(self, quads_on=True, index_labels=True, adj=6, zlim=None, rlim=None):
        '''
        Plots mesh.

        Optional parameters:
            quads_on : boolean
                determines whether magnetic materials, coils, electrodes, etc.
                are also plotted.
                default True
            index_labels : boolean
                determines whether mesh indices are labeled.
            adj : float 
                offset of mesh labels from matplotlib axes.
                default 6
            zlim : tuple or list of two floats
                sets matplotlib xlim to zoom in on region of the z axis.
                generally looks horrible with index_labels=True.
            rlim : tuple or list of two floats
                sets matplotlib ylim to zoom in on region of th r axis.
                generally looks horrible with index_labels=True.
        '''
        plt.figure(figsize=(10,10))
        self.add_coarse_mesh_to_plot()
        self.add_mesh_labels_to_plot(index_labels, adj)
        self.add_quads_to_plot() if quads_on else 0
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
        plt.xlim(zlim)
        plt.ylim(rlim)
        # plt.title("Mesh (coarse)")
        plt.gca().set_aspect('equal')
        plt.show()

    def add_fine_mesh_to_plot(self, linewidth=0.1):
        self.add_mesh_segments_to_plot(self.define_fine_mesh_segments(), linewidth)

    def add_coarse_mesh_to_plot(self):
        self.add_mesh_segments_to_plot(self.define_coarse_mesh_segments())

    def add_mesh_segments_to_plot(self, segments, linewidth=1):
        for segment in segments:
            if(segment):
                z,r = segment.shape.xy
                plt.plot(z,r,color='m',linewidth=linewidth)

    def add_mesh_labels_to_plot(self, index_labels=True, adj=6):
        for n in range(self.z.shape[0]):
            if(index_labels):
                # add r index label
                plt.text(self.z[n,:].max()+adj, self.r[n,np.argmax(self.z[n,:])], self.r_indices[n])
        for n in range(self.z.shape[1]):
            if(index_labels):
                # add z index label
                plt.text(self.z[np.argmax(self.r[:,n]),n], self.r[:,n].max()+adj, self.z_indices[n])
        
    def add_quads_to_plot(self):
        pass
    
    def retrieve_segments(self, z_indices, r_indices):
        np_z_indices = np.nonzero((self.z_indices >= z_indices.min())*(self.z_indices <= z_indices.max()))[0]
        np_r_indices = np.nonzero((self.r_indices >= r_indices.min())*(self.r_indices <= r_indices.max()))[0]
        if(len(np_z_indices) == 0 or len(np_r_indices) == 0):
            raise ValueError('Quads that are not defined on the coarse mesh are not implemented.')
        seg_np_indices = []
        seg_np_indices.append((np.repeat(np_r_indices.min(),len(np_z_indices)),np_z_indices))
        seg_np_indices.append((np_r_indices,np.repeat(np_z_indices.max(),len(np_r_indices))))
        # reverse these two so that it's ccw order # actually seems cw?
        seg_np_indices.append((np.repeat(np_r_indices.max(),len(np_z_indices)),np_z_indices[::-1]))
        seg_np_indices.append((np_r_indices[::-1],np.repeat(np_z_indices.min(),len(np_r_indices))))
        return seg_np_indices

    def retrieve_MEBSSegments(self, z_indices, r_indices):
        np_z_indices = np.nonzero((self.z_indices >= z_indices.min())*(self.z_indices <= z_indices.max()))[0]
        np_r_indices = np.nonzero((self.r_indices >= r_indices.min())*(self.r_indices <= r_indices.max()))[0]
        if(len(np_z_indices) == 0 or len(np_r_indices) == 0):
            raise ValueError('Quads that are not defined on the coarse mesh are not implemented.')
        segments = []
        if(not hasattr(self,'coarse_segments')): # REVISIT depending on final usage of this fn 
            self.define_coarse_mesh_segments()
        for z_index in np_z_indices[:-1]:
            segments.append(self.coarse_segments[np_r_indices.min(),z_index,1])
        for r_index in np_r_indices[:-1]:
            segments.append(self.coarse_segments[r_index,np_z_indices.max(),0])
        # reverse these for consistent ordering
        for z_index in np_z_indices[:-1][::-1]:
            segments.append(
               MEBSSegment(None, None, prev_segment=self.coarse_segments[np_r_indices.max(),z_index,1], reverse=True))
        for r_index in np_r_indices[:-1][::-1]:
            segments.append(
               MEBSSegment(None, None, prev_segment=self.coarse_segments[r_index,np_z_indices.min(),0], reverse=True))
        return segments
    
    # make a list of the numpy indices for points on the boundary of a given
    # type of quad. quad type is specificed by the inputs.
    # e.g. z_indices = self.mag_mat_z_indices or self.coil_z_indices for 
    # a magnetic lens.
    # with argument return_ind_array=True, reformats list into index arrays
    # to be called as self.r[unique_points]
    def retrieve_edge_points(self, z_indices, r_indices, return_ind_array=False):
        # points holds tuples of np indices for all points on the boundary
        points = []
        for n in range(len(r_indices)):
            for seg in self.retrieve_segments(z_indices[n], r_indices[n]):
                for m in range(len(seg[0])):
                     points.append((seg[0][m],seg[1][m])) 
        unique_points = list(dict.fromkeys(points)) # removes duplicate entries
        return index_array_from_list(unique_points) if return_ind_array else unique_points

    # same as above, but for a single quad
    def retrieve_single_quad_edge_points(self, quad_z_indices, quad_r_indices, return_ind_array=False):
        points = []
        for seg in self.retrieve_segments(quad_z_indices, quad_r_indices):
            for m in range(len(seg[0])):
                points.append((seg[0][m],seg[1][m]))
        unique_points = list(dict.fromkeys(points)) # removes duplicate entries
        return index_array_from_list(unique_points) if return_ind_array else unique_points

    def determine_quad_area(self, quad_z_indices, quad_r_indices):
        return self.make_polygon(quad_z_indices, quad_r_indices).area

    def make_polygon(self, z_indices, r_indices):
        polygon_coords = []
        for segment in self.retrieve_MEBSSegments(z_indices, r_indices):
            polygon_coords += segment.shape.coords
        return Polygon(polygon_coords)

    # collect all segments in the coarse mesh into a list
    def define_coarse_mesh_segments(self):
        segments = np.empty(self.z.shape+(2,),dtype=np.object)
        for i in range(self.z.shape[0]):
            for j in range(self.z.shape[1]):
                if(i+1 < self.z.shape[0]):
                    segments[i,j,0] = MEBSSegment(Point(self.z[i,j], self.r[i,j]), Point(self.z[i+1,j], self.r[i+1,j]), 
                                                  self.z_curv[i,j])
                if(j+1 < self.z.shape[1]):
                    segments[i,j,1] = MEBSSegment(Point(self.z[i,j], self.r[i,j]), Point(self.z[i,j+1], self.r[i,j+1]), 
                                                  self.r_curv[i,j])
        self.coarse_segments = segments
        return segments.flatten()

    def define_fine_mesh_segments(self):
        '''
        Uses interpolation to define longest possible fine mesh segments
        (i.e. coarse mesh length, interpolated between coarse mesh segments).
        '''
        segments = []
        r_interpolator = interp2d(self.z_indices,self.r_indices,self.r)
        z_interpolator = interp2d(self.z_indices,self.r_indices,self.z)
        r_curv_phys = np.copy(self.r_curv)
        # curvature = 0 actually means infinite radius, so make that explicit
        r_curv_phys[self.r_curv == 0] = np.inf
        # interpolate the inverse radius
        inv_r_curv = 1.0/r_curv_phys
        z_curv_phys = np.copy(self.z_curv)
        z_curv_phys[self.z_curv == 0] = np.inf
        inv_z_curv = 1.0/z_curv_phys
        inv_r_curv_interpolator = interp2d(self.z_indices,self.r_indices,inv_r_curv)
        inv_z_curv_interpolator = interp2d(self.z_indices,self.r_indices,inv_z_curv)
        # MEBS uses half-integer steps for the fine mesh
        step = 0.5
        # first, longtudinal fine mesh segments
        for r_index in np.arange(self.r_indices[0],self.r_indices[-1]+step,step):
            # find numpy index of last passed point in self.r_indices
            i = last_np_index(self.r_indices, r_index)
            # t is the percentage along the radial segment 
            # at which this longitudinal fine segment begins
            r_dist = r_index - self.r_indices[i]
            t = r_dist/(self.r_indices[i+1]-self.r_indices[i]) if r_dist else 0
            # make coarse mesh-sized segments from fine mesh
            # skip last point as second point in segment doesn't exist
            for j,z_index in enumerate(self.z_indices[:-1]):
                inv_curv = np.asscalar(inv_r_curv_interpolator(z_index,r_index))
                curv = 1.0/inv_curv if inv_curv != 0 else 0
                if(self.z_curv[i,j] != 0):
                        #inv_z_curv_interpolator(z_index,r_index) != np.inf):
                    # deal with curvature of radial segment on which point_a lies
                    point_a = Point(self.coarse_segments[i,j,0].shape.interpolate(t,normalized=True))
                else:
                    point_a = Point(np.asscalar(z_interpolator(z_index,r_index)),
                                    np.asscalar(r_interpolator(z_index,r_index)))
                if(self.z_curv[i,j+1] != 0):
                        # inv_z_curv_interpolator(self.z_indices[j+1],r_index) != np.inf):
                    # deal with curvature of radial segment on which point_b lies
                    point_b = Point(self.coarse_segments[i,j+1,0].shape.interpolate(t,normalized=True))
                else:
                    point_b = Point(np.asscalar(z_interpolator(self.z_indices[j+1],r_index)),
                                    np.asscalar(r_interpolator(self.z_indices[j+1],r_index)))

                segments.append(MEBSSegment(point_a, point_b, curv))
        # now, radial fine mesh segments
        for z_index in np.arange(self.z_indices[0],self.z_indices[-1]+step,step):
            # find numpy index of last passed point in self.z_indices
            j = last_np_index(self.z_indices, z_index)
            # t is the percentage along the longitudinal segment 
            # at which this radial fine segment begins
            z_dist = z_index-self.z_indices[j]
            t = z_dist/(self.z_indices[j+1]-self.z_indices[j]) if z_dist else 0
            for i,r_index in enumerate(self.r_indices[:-1]):
                inv_curv = np.asscalar(inv_z_curv_interpolator(z_index,r_index))
                curv = 1.0/inv_curv if inv_curv != 0 else 0
                if(self.r_curv[i,j] != 0):
                        # inv_r_curv_interpolator(z_index,r_index) != np.inf):
                    point_a = Point(self.coarse_segments[i,j,1].shape.interpolate(t,normalized=True))
                else:
                    point_a = Point(np.asscalar(z_interpolator(z_index,r_index)),
                                    np.asscalar(r_interpolator(z_index,r_index)))
                if(self.r_curv[i+1,j] != 0):
                        # inv_r_curv_interpolator(z_index,self.r_indices[i+1]) != np.inf):
                    point_b = Point(self.coarse_segments[i+1,j,1].shape.interpolate(t,normalized=True))
                else:
                    point_b = Point(np.asscalar(z_interpolator(z_index,self.r_indices[i+1])),
                                    np.asscalar(r_interpolator(z_index,self.r_indices[i+1])))

                segments.append(MEBSSegment(point_a, point_b, curv))
        self.fine_segments = segments
        return segments

    def plot_quad(self, z_indices, r_indices, color='k', linewidth=1):
        for seg in self.retrieve_MEBSSegments(z_indices, r_indices):
            z,r = seg.shape.xy
            plt.plot(z,r,color=color,linewidth=linewidth)
    
    def plot_field(self):
        '''
        Plots the magnetic field calculated by MEBS. Run after calc_field().

        No parameters. 
        '''
        try:
            with cd(self.dirname):
                z,B = np.genfromtxt(self.potname, dtype=float, skip_header=7, skip_footer=4, unpack=True)
                plt.plot(z,B)
                plt.xlabel('z (mm)')
                plt.ylabel('B (T)')
                plt.show()
        except OSError:
            self.olog.log.info('No file with name {} found. Run calc_field first.'.format(self.potname))
            raise FileNotFoundError

    def add_curvature(self):
        '''
        Converts optical element .dat file used for optics into a file 
        compatible with MEBS SOFEM by adding curvature coordinates.
        
        No arguments.
        '''

        if(hasattr(self,'r_curv')):
            raise AttributeError('Curvatures already defined.')
        self.r_curv = np.zeros_like(self.r)
        self.z_curv = np.zeros_like(self.z)
        self.so = True
         
    def autoformat_multi(self,template,contents):
        try: 
            string = check_len_multi(template.format(*contents),self.colwidth)
        except ValueError: # recursively increase colwidth
            if(self.automated):
                self.colwidth += 1
                string = self.autoformat_multi(template,contents)
            else:
                raise ValueError
        return string


class StrongMagLens(OpticalElement):
    '''
    class for reading and writing MEBS magnetic lens files that use hysteresis
    curves (H-B curves, in fact here without any hysteresis) for magnetic 
    materials and explicitly included coils to model magnetic lenses.

    Attributes:
        mag_mat_z_indices : list
        mag_mat_r_indices : list
            N two-element arrays for the r and z indices of 
            N different quads for magnetic materials

        self.mag_mat_curve_indices = [] 
            list of N indices with values from 1 to M denoting which 
            hysteresis curve to use for each magnetic material quad
        
        self.coil_z_indices = [] 
        self.coil_r_indices = []
            L two-element arrays for the r and z indices for L coil quads
        
        self.coil_curr = [] 
            L current values for these L quads (A-turns/cm^2)
        
        self.H_arrays = [] 
        self.B_arrays  = [] 
            these two lists store M different hysteresis curves for the magnetic materials

    User methods:
        plot_hyst
        calc_field
    '''

    lens_type = 'magnetic'
    lens_strength = 1

    def initialize_lists(self):
        # N two-element arrays for the r and z indices of 
        # N different quads for magnetic materials
        self.mag_mat_z_indices = [] 
        self.mag_mat_r_indices = []
        
        # list of N indices with values from 1 to M denoting which 
        # hysteresis curve to use for each magnetic material quad
        self.mag_mat_curve_indices = [] 
        
        # L two-element arrays for the r and z indices for L coil quads
        self.coil_z_indices = [] 
        self.coil_r_indices = []
        
        # L current values for these L quads (A-turns/cm^2)
        self.coil_curr = [] 
        
        # these two lists store M different hysteresis curves for the magnetic materials
        self.H_arrays = [] 
        self.B_arrays  = [] 
    
    
    def read_other_blocks(self, line_num):
        self.potname = self.basename_noext+'.axb' # name of potential file
        line_num = self.read_mag_mat(line_num)
        line_num = self.read_coil(line_num)
        self.plot_mesh_coarse(quads_on=True) if self.plot else 0
        while(self.infile[line_num].isspace() != True):
            line_num = self.read_hyst_curve(line_num)
            self.plot_hyst() if self.plot else 0
    
    def read_mag_mat(self, line_num):
        return self.read_quad(line_num, self.mag_mat_z_indices, self.mag_mat_r_indices, self.mag_mat_curve_indices, 
                              property_dtype=int)
    
    def read_coil(self, line_num):
        return self.read_quad(line_num, self.coil_z_indices, self.coil_r_indices, self.coil_curr, property_dtype=float)
    
    def read_hyst_curve(self, line_num):
        lines = []
        while(self.infile[line_num].isspace() != True):
            lines.append(np.fromstring(self.infile[line_num],dtype=float,count=2,sep=' '))
            line_num+=1
        lines = np.array(lines)
        self.H_arrays.append(lines[:,0])
        self.B_arrays.append(lines[:,1])
        return line_num+1 # start of next block
    
    def plot_hyst(self, i=-1):
        '''
        Plots H-B curve for specified magnetic material.

        Optional parameters:
            i : int
                python index denoting which magnetic material to plot.
                default -1 (last)
        '''
        plt.plot(self.H_arrays[i],self.B_arrays[i])
        plt.title(f"Hysteresis curve #{len(self.H_arrays)}")
        plt.xlabel("H (A-turns/m)")
        plt.ylabel("B (T)")
        plt.show()
        
    def write_other_blocks(self, f):
        self.write_mag_mat(f)
        self.write_coil(f)
        self.write_hyst(f) # leaves one blank line at end of section
        f.write("\n\n") # three blank lines for a strong magnetic lens
    
    def write_mag_mat(self, f):
        self.write_quad(f, self.mag_mat_z_indices, self.mag_mat_r_indices, self.mag_mat_curve_indices, self.int_fmt)
    
    def write_coil(self, f):
        self.write_quad(f, self.coil_z_indices, self.coil_r_indices, self.coil_curr, self.curr_fmt)

    def write_hyst(self, f):
        M = len(self.H_arrays)
        for m in range(M):
            for k in range(len(self.H_arrays[m])):
                f.write(check_len(self.field_fmt.format(self.H_arrays[m][k]),self.colwidth))
                f.write(check_len(self.field_fmt.format(self.B_arrays[m][k]),self.colwidth))
                f.write("\n")
            f.write("\n")
            
    def add_quads_to_plot(self):
        self.plot_mag_mat()
        self.plot_coil()
    
    def plot_mag_mat(self):
        N = len(self.mag_mat_r_indices)
        for n in range(N):
            self.plot_quad(self.mag_mat_z_indices[n], self.mag_mat_r_indices[n], color='k')
            
    def plot_coil(self):
        L = len(self.coil_r_indices)
        for l in range(L):
            self.plot_quad(self.coil_z_indices[l], self.coil_r_indices[l], color='r')

    def calc_field(self):
        '''
        Calls somlenss.exe to calculate magnetic field for this lens.

        No arguments.
        '''
        with cd(self.dirname):
            outputmode = subprocess.PIPE 
            try:
                output = subprocess.run(["somlenss.exe",self.basename_noext], stdout=outputmode, 
                                        timeout=self.timeout).stdout
                self.Mlog.log.info(output.decode('utf-8')) 
            except TimeoutExpired:
                self.olog.log.info('Field calculation timed out. Rerunning.')
                self.calc_field()


class WeakMagLens(StrongMagLens):
    '''
    class for reading and writing MEBS magnetic lens files that use a constant
    relative magnetic permeability to represent magnetic materials and uses
    explicitly included coils.'''
    
    def initialize_lists(self):
        # N relative permeabilities for each magnetic material quad
        self.mag_mat_mu_r = [] 

        self.mag_mat_z_indices = [] 
        self.mag_mat_r_indices = []

        self.coil_z_indices = [] 
        self.coil_r_indices = []
        
        self.coil_curr = [] 
        
    
    def read_other_blocks(self, line_num):
        line_num = self.read_mag_mat(line_num)
        line_num = self.read_coil(line_num)
        self.plot_mesh_coarse(quads_on=True) if self.plot else 0
    
    def read_mag_mat(self, line_num):
        return self.read_quad(line_num, self.mag_mat_z_indices, self.mag_mat_r_indices, self.mag_mat_mu_r, 
                              property_dtype=float)
        
    def write_other_blocks(self, f):
        self.write_mag_mat(f)
        self.write_coil(f)
        f.write("\n\n") # two blank lines for a weak magnetic lens
    
    def write_mag_mat(self, f):
        self.write_quad(f, self.mag_mat_z_indices, self.mag_mat_r_indices, self.mag_mat_mu_r, self.rel_perm_fmt)

    def calc_field(self):
        '''
        Calls somlensc.exe to calculate magnetic field for this lens.

        No arguments.
        '''
        with cd(self.dirname):
            try:
                self.Mlog.log.info(subprocess.run(["somlensc.exe",self.basename_noext], stdout=subprocess.PIPE, 
                                     timeout=self.timeout).stdout.decode('utf-8'))
            except TimeoutExpired:
                self.olog.log.info('Field calculation timed out. Rerunning.')
                self.calc_field()


class WeakMagLens_PP_Region(WeakMagLens):
    '''
    class for reading and writing MEBS magnetic lens files that only consider 
    the polepiece region and use a scalar magnetic potential to solve for
    the field.
    
    coil currents are specified only at boundaries.
    '''
    
    ## on defining coil currents:
    #
    ##  from "OPTICS_GUI_2.4" p. 42:
    # The first section specifies the potentials on the left-hand boundary;
    # The second section specifies the potentials on the upper boundary;
    # The third section specifies the potentials on the right-hand boundary.
    ## from "SOFEM-GUI" p. 20:
    ## first is top to bottom, second is left to right, third is top to bottom
    ## so indices on each boundary are strictly of one kind
    ## e.g. only r indices for the left boundary
    # On the boundaries, the program interpolates the potential linearly 
    # between the mesh-points at which the potential values are explicitly
    # specified."
    ## confusingly, it seems that the right-hand boundary exists even with a
    # symmetric solver
    #
    # I am guessing that MEBS delineates sections by a decreased index, i.e.
    # 1  25.0
    # 20 25.0
    # 1  0.0    <- MEBS decides this is a new section (?)

    def initialize_lists(self):
        self.boundary_coil_indices = [] 
        self.boundary_coil_currents = []

        self.mag_mat_z_indices = [] 
        self.mag_mat_r_indices = []
        
        self.mag_mat_mu_r = [] 

    
    def read_other_blocks(self, line_num):
        line_num = self.read_mag_mat(line_num)
        line_num += 1 # extra blank line here for scalar magnetic potential files
        line_num = self.read_coil(line_num)
        # when properly implemented, plot_mesh_coarse would also represent 
        # boundary currents in some way
        self.plot_mesh_coarse(quads_on=True) if self.plot else 0
        
    def write_other_blocks(self, f):
        self.write_mag_mat(f) # leaves one blank line at end of section
        f.write("\n") # extra blank line here for scalar potential files
        self.write_coil(f) # leaves one blank line at end of section
        f.write("\n") # two total blank lines for a weak magnetic lens
        
    def read_coil(self, line_num):
        lines = []
        while(self.infile[line_num].isspace() != True):
            lines.append(np.fromstring(self.infile[line_num],dtype=float,count=2,sep=' '))
            line_num+=1
        lines = np.array(lines)
        self.boundary_coil_indices = lines[:,0].astype(int)
        self.boundary_coil_currents = lines[:,1]
        section_starts = [] # list of numpy indices that denote the start of a section
        prev_index = 1 # counter
        for n,index in enumerate(self.boundary_coil_indices):
            if(index < prev_index):
                section_starts.append(n)
            prev_index = index
        # turn these two numpy arrays into a list containing each section as a numpy array
        self.boundary_coil_indices = np.split(self.boundary_coil_indices, section_starts)
        self.boundary_coil_currents = np.split(self.boundary_coil_currents, section_starts)
        return line_num+1 # start of next block
    
    def write_coil(self, f):
        M = len(self.boundary_coil_indices)
        for m in range(M):
            for k in range(len(self.boundary_coil_indices[m])):
                f.write(check_len(self.int_fmt.format(self.boundary_coil_indices[m][k]),self.colwidth))
                f.write(check_len(self.curr_fmt.format(self.boundary_coil_currents[m][k]),self.colwidth))
        f.write("\n")

    def add_quads_to_plot(self):
        self.plot_mag_mat()

    def calc_field(self):
        '''
        Calls somlensp.exe to calculate magnetic field for this lens.

        No arguments.
        '''
        with cd(self.dirname):
            try:
                self.Mlog.log.info(subprocess.run(["somlensp.exe",self.basename_noext], stdout=subprocess.PIPE, 
                                     timeout=self.timeout).stdout.decode('utf-8'))
            except TimeoutExpired:
                self.olog.log.info('Field calculation timed out. Rerunning.')
                self.calc_field()

class ElecLens(OpticalElement):
    '''
    Class for electrostatic lenses and mirrors.
    
    No arguments.
    
    User methods:
        mirror_type
        calc_field
    '''

    lens_type = 'electrostatic'

    def mirror_type(self, mirror, curved_mirror):
        '''
        Run after initializing ElecLens to classify lens.

        Parameters:
            mirror : bool
                Indicates whether optical element is a mirror.
                Starts as False (initialized in __init__).
            curved_mirror : bool
                Indicates whether mirror is curved.
                Starts as False (initialized in __init__).
        '''
        self.mirror = mirror
        self.curved_mirror = curved_mirror
        if(self.curved_mirror):
            self.freeze_xy_plane = False
        self.freeze_radial_boundary = True

    def initialize_lists(self):
        # N two-element arrays for the r and z indices of 
        # N different quads for electrodes
        self.electrode_z_indices = [] 
        self.electrode_r_indices = []
        
        # N lists of length-M arrays that delineate independence of electrodes
        self.electrode_unit_potentials = []
        
        # L two-element arrays for the r and z indices for L dielectric quads
        self.dielectric_z_indices = [] 
        self.dielectric_r_indices = []
        
        # L dielectric constants for these L quads (unitless)
        # i.e. relative permittivity
        self.dielectric_constants = [] 
        
        # list of arrays for each section
        # boundary_unit_potentials arrays are 2D, where second dimension is length-M
        self.boundary_indices = []
        self.boundary_unit_potentials = []
    
    
    def read_other_blocks(self, line_num):
        self.potname = self.basename_noext+'.axv' # name of potential file
        line_num = self.read_electrodes(line_num)
        line_num = self.read_dielectrics(line_num)
        self.plot_mesh_coarse(quads_on=True) if self.plot else 0
        line_num = self.read_boundaries(line_num)
    
    # read_quad with one line tweaked
    def read_electrodes(self, line_num):
        while(self.infile[line_num].isspace() != True):
            indices = np.fromstring(self.infile[line_num],dtype=int,count=4,sep=' ')
            self.electrode_z_indices.append(indices[:2])
            self.electrode_r_indices.append(indices[2:4])
            self.electrode_unit_potentials.append(np.fromstring(self.infile[line_num],dtype=float,sep=' ')[4:])
            line_num+=1
        return line_num+1 # start of next block
    
    def read_dielectrics(self, line_num):
        return self.read_quad(line_num, self.dielectric_z_indices, self.dielectric_r_indices, 
                              self.dielectric_constants, property_dtype=float)

    def read_boundaries(self, line_num):
        lines = []
        while(self.infile[line_num].isspace() != True):
            lines.append(np.fromstring(self.infile[line_num],dtype=float,sep=' '))
            line_num+=1
        lines = np.array(lines)
        self.boundary_indices = lines[:,0].astype(int)
        self.boundary_unit_potentials = lines[:,1:]
        section_starts = [] # list of numpy indices that denote the start of a section
        prev_index = 1 # counter
        for n,index in enumerate(self.boundary_indices):
            if(index < prev_index):
                section_starts.append(n)
            prev_index = index
        # turn these two numpy arrays into a list containing each section as a numpy array
        self.boundary_indices = np.split(self.boundary_indices, section_starts)
        self.boundary_unit_potentials = np.split(self.boundary_unit_potentials, section_starts)
    
    def write_other_blocks(self, f):
        self.write_electrodes(f)
        self.write_dielectrics(f)
        self.write_boundaries(f) # leaves one blank line at end of section
        f.write("\n") # two total blank lines for an electrostatic lens
    
    def write_electrodes(self, f):
        N = len(self.electrode_z_indices)
        M = len(self.electrode_unit_potentials[0])
        for n in range(N):
            f.write(self.autoformat_multi((self.int_fmt*2),self.electrode_z_indices[n]))
            f.write(self.autoformat_multi((self.int_fmt*2),self.electrode_r_indices[n]))
            f.write(self.autoformat_multi((self.voltage_fmt*M),self.electrode_unit_potentials[n]))
            f.write("\n")
        f.write("\n")
    
    def write_dielectrics(self, f):
        self.write_quad(f, self.dielectric_z_indices, self.dielectric_r_indices, self.dielectric_constants, 
                        self.rel_perm_fmt)
    def write_boundaries(self, f):
        M = len(self.electrode_unit_potentials[0])
        for p in range(len(self.boundary_indices)): # iterate sections
            for q in range(len(self.boundary_indices[p])): # iterate indices in sections
                f.write(check_len(self.int_fmt.format(self.boundary_indices[p][q]),self.colwidth))
                f.write(self.autoformat_multi((self.voltage_fmt*M),self.boundary_unit_potentials[p][q]))
                f.write("\n")
        f.write("\n")

    def add_quads_to_plot(self):
        self.plot_electrodes()
        self.plot_dielectrics()
    
    def plot_electrodes(self):
        N = len(self.electrode_r_indices)
        for n in range(N):
            self.plot_quad(self.electrode_z_indices[n], self.electrode_r_indices[n], color='k')
            
    def plot_dielectrics(self):
        L = len(self.dielectric_r_indices)
        for l in range(L):
            self.plot_quad(self.dielectric_z_indices[l], self.dielectric_r_indices[l], color='r')

    def calc_field(self):
        '''
        Calls soelens.exe to calculate electric field for this lens.

        No arguments.
        '''
        with cd(self.dirname):
            outputmode = subprocess.PIPE 
            try:
                output = subprocess.run(["soelens.exe",self.basename_noext], stdout=outputmode, 
                                        timeout=self.timeout).stdout
                self.Mlog.log.info(output.decode('utf-8')) 
            except TimeoutExpired:
                self.olog.log.info('Field calculation timed out. Rerunning.')
                self.calc_field()


class MirPotentials:
    '''
    Class for formatting potentials for electric lenses.
    '''
    def __init__(self, parent, voltages, flags, voltage_precision=6):
        '''
        Parameters:
            parent : OpticalElement object
                Used to pass several OpticalElement settings.
            voltages : list
                list of voltages to apply to electrodes
            flags : list
                list of string-type flags that indicate electrode 
                variability for autofocusing. options are 'f', 'v1', 'v2'
                and so on. 
        Optional Parameters:
            voltage_precision : int
                precision to use to format voltages. Default 6.
        '''
        self.parent = parent
        self.voltages = voltages
        self.flags = flags
        self.voltage_fmt = self.parent.rfloat_fmt.substitute(imgcondcolwidth=parent.colwidth,
                                                             precision=voltage_precision)
        self.string = ''
        if(len(flags) != len(voltages)):
            raise ValueError('Lengths of voltage and flag arrays not equal.')

    def format(self):
        if(self.string == ''): # if not set yet, set
            for i in range(len(self.voltages)):
                self.string += check_len(self.voltage_fmt.format(self.voltages[i]) + self.flags[i],
                                                                               self.parent.colwidth)
        return self.string

    def format_noflag(self):
        return self.parent.autoformat_multi((self.parent.voltage_fmt*len(self.voltages)),self.voltages)

# snippets for each property, with example line numbers on end, for reference
# --- denotes omitted lines
'''
                         FIRST-ORDER PROPERTIES                           # 78


 FOCUSING PROPERTIES:

      MAGNETIC LENS No. 1       EXCITATION  (Ampturns)   =  19200.0000



      ANGULAR MAGNIFICATION ..........................   = -7.8393e-01
      MAGNIFICATION ..................................   = -1.2756e+00
      ROTATION ANGLE ....................    (degrees)   =  1.7330e+02
---
Region (from    0.000 mm to    6.134 mm), including following lens(es):   
     Magnetic Lens      No.  1                                            # 101


    REAL PRINCIPAL PLANE       (Object Side) ZPreal (mm) =     6.6208
    REAL FOCAL PLANE           (Object Side) ZFreal (mm) =     3.4439
    FOCAL LENGTH (REAL)        (Object Side)  Freal (mm) =     3.1769

    ASYMPTOTIC PRINCIPAL PLANE (Object Side) ZPasym (mm) =    -4.9551
    ASYMPTOTIC FOCAL PLANE     (Object Side) ZFasym (mm) =   -12.5992
    FOCAL LENGTH (ASYMPTOTIC)  (Object Side)  Fasym (mm) =     7.6441
---
                 THIRD-ORDER ABERRATION COEFFICIENTS   (in S.I. units)    # 400



                                ISOTROPIC      ANISOTROPIC     FUNCTIONAL
                                   PART            PART        DEPENDENCE

------------------------------------------------------------------------------

 SPHERICAL ABERRATION          5.89490e-03     0.00000e+00      A  A  Ac  # 409
 ---
 ***** CHROMATIC AB *****                                                 # 438


 AXIAL                        -4.09665e-03     0.00000e+00      A  DV/V   # 441
'''
