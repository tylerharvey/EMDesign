#!/usr/bin/env python3
import sys,os,subprocess,shutil,datetime
from subprocess import TimeoutExpired
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from contextlib import contextmanager
from scipy.interpolate import interp2d

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

# safe chdir, found on stackexchange; do not understand it but
# when used as "with cd():" it returns to the previous directory
# no matter what
@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

class Point(object):
    def __init__(self,z,r):
        self.z = z
        self.r = r

    def print(self):
        print(f"z: {self.z}, r: {self.r}")

    def __eq__(self,other):
        return self.__dict__ == other.__dict__
        
def np_indices(indices,index_set):
    np_index_array = []
    for index in index_set:
        np_index_array.append(np.nonzero(indices == index)[0][0]) # should find way to parallelize with nonzero
    return np.array(np_index_array)

# takes a list of np indices np_list = [(1,0),(2,5),...]
# and casts to a format that can be used for numpy indexing
# i.e. array[index_array_from_list(index_list)]
def index_array_from_list(index_list):
    if(len(index_list) == 0 or index_list == ([],[])):
        return [],[]
    tmp_array = np.array(index_list)
    return (tmp_array[:,0],tmp_array[:,1])

def calculate_area(x,y):
    return 0.5*np.abs((x[:-1]*y[1:]-x[1:]*y[:-1]).sum()+x[-1]*y[0]-x[0]*y[-1])

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
            boolean flag used by MEBS to denote the symmetry of the element 
            through the x-y plane (between positive and negative z), 
            with 1 true and 0 false.
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
        c3 : float
            spherical aberration in mm (set by read_optical_properties)
        cc : float
            chromatic aberration in mm (set by read_optical_properties)
        f : float
            focal length in mm (set by read_optical_properties)
        f_real : float
            physical position of back focal plane in mm 
            (set by read_optical_properties)
        mag : float
            magnification (set by read_optical_proerties)
        rot : float
            image rotation in deg (set by read_optical_properties)
        lens_curr : float
            current in first magnetic lens in A-turns (set by read_optical ...)

    User methods:
        plot_mesh_coarse
        plot_field
        write_opt_img_cond_file
        read_optical_properties
        calc_rays
        add_curvature (converts optical element .dat file for FOFEM use)

    Hard-coded attributes:
        colwidth : int
            width of columns to use in the optical element file.
            default 12.
        imgcondcolwidth : int
            width of columns to use in optical properties file.
            default 40.
        timeout : float
            seconds to wait before killing MEBS.
            default 10 minutes.
    '''
    # output = 2
    # axsym = 1
    # title = 'optical element'
    # r_indices = np.array([])
    # z_indices = np.array([])
    # r = np.array([[]])
    # z = np.array([[]])
    
    ## infile = []
    imgcondcolwidth = 40
    colwidth = 12 # width of columns in written .dat files
    sp = " "*colwidth 
    int_fmt = Template("{:${colwidth}d}").substitute(colwidth=colwidth)
    float_fmt = Template("{:${colwidth}.${precision}g}")
    imgcondprop_fmt = Template("{:<${imgcondcolwidth}s}").substitute(imgcondcolwidth=imgcondcolwidth)
    imgcondsubprop_fmt = Template("  {:<${imgcondcolwidth}s}").substitute(imgcondcolwidth=imgcondcolwidth-2)
    imgcondtext_fmt = Template("{:>${imgcondcolwidth}s}").substitute(imgcondcolwidth=imgcondcolwidth)
    imgcondint_fmt = Template("{:>${imgcondcolwidth}d}").substitute(imgcondcolwidth=imgcondcolwidth)
    rfloat_fmt = Template("{:>${imgcondcolwidth}.${precision}g}")
    timeout = 10*60 # 6 minutes
    

    # verbose plots everything
    # so reads .dat files for the second-order solver
    # these are almost exactly the same except they have radii of curvature
    # sections after the mesh that looks like the mesh blocks
    def __init__(self,filename='',verbose=False,so=False):
        '''
        Initialize an instance of an optical element.

        Parameters:
            filename : path
                full filename pointing to the optical element .dat file.

        Optional flags:
            verbose : boolean
                captures as much output as possible and plots often.
                default False
            so : boolean
                denotes whether the optical element .dat file includes 
                curvature coordinates for compatibility with MEBS SOFEM.
                (OPTICS does not use curvature.) flag will automatically 
                flip to True if curvature coordinates are found.
                default False
        '''

        self.so=so
        self.mirror = False
        self.curved_mirror = False
        self.infile = []
        self.initialize_lists()
        self.verbose = verbose
        if(filename):
            self.read(filename)
            shutil.copyfile(filename,filename+'.bak')
            
    def initialize_lists(self):
        pass
    
    def read(self,filename):
        f = open(filename,'r')
        self.filename = filename
        self.basename_noext = os.path.splitext(os.path.basename(filename))[0] # name without directories or extension
        self.basename = os.path.basename(filename) # name without directories
        self.fitname = self.basename_noext+'.fit' # name of potential file
        self.filename_noext = os.path.splitext(filename)[0] # with directories
        self.dirname = os.path.dirname(filename)
        self.infile = list(f) # creates a list with one entry per line of f
        self.read_intro()
        print(f"Reading file {filename} \nwith title: {self.title}")
        line_num = self.read_mesh()
        if(self.so):
            line_num = self.read_curvature(line_num)
        # self.plot_mesh_coarse() if self.verbose else 0
        self.read_other_blocks(line_num)
        f.close()
        f = None

    def read_intro(self):
        self.title = self.infile[0].strip('\n')
        self.output,self.axsym = np.fromstring(self.infile[1],dtype=int,count=2,sep=' ')
    
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
            raise Exception("Read error! z indices read in first and second mesh block not identical!")
        line_num+=1 # move to r coordinates
        r_indices_2 = []
        while(self.infile[line_num].isspace() != True):
            r_indices_2.append(np.fromstring(self.infile[line_num],dtype=int,count=1,sep=' ')[0])
            r.append(np.fromstring(self.infile[line_num],dtype=float,sep=' ')[1:]) # exclude r index
            line_num+=1
        self.r = np.array(r)
        if(np.array_equal(r_indices,r_indices_2) != True):
            raise Exception("Read error! r indices read in first and second mesh block not identical!")
        line_num+=1 # skip blank line we just found
        if(self.so == False and (len(self.z_indices) != 5 and len(np.fromstring(self.infile[line_num],dtype=int,sep=' ')) != 5) or (len(self.z_indices) == 5 and np.array_equal(np.fromstring(self.infile[line_num],dtype=int,sep=' '),self.z_indices))):
            print('Warning! This data file seems to have curvature coordinates. Setting so=True.')
            self.so = True
        return line_num # save line number of the start of the next block
    
    def read_curvature(self,line_num):
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
    def read_quad(self,line_num,z_indices,r_indices,quad_property,property_dtype=float):
        while(self.infile[line_num].isspace() != True):
            indices = np.fromstring(self.infile[line_num],dtype=int,count=4,sep=' ')
            z_indices.append(indices[:2])
            r_indices.append(indices[2:4])
            quad_property.append(np.fromstring(self.infile[line_num],dtype=property_dtype,count=5,sep=' ')[-1])
            line_num+=1
        return line_num+1 # start of next block
    
    def write(self,outfilename,title=None,coord_precision=6,curr_precision=6,field_precision=6,rel_perm_precision=6,voltage_precision=6):
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
        self.filename_noext = os.path.splitext(outfilename)[0]
        self.dirname = os.path.dirname(outfilename)
        
        self.write_intro(f,title)
        self.write_coordinates(f,self.z)
        self.write_coordinates(f,self.r)
        if(self.so):
           self.write_coordinates(f,self.z_curv)
           self.write_coordinates(f,self.r_curv)
        self.write_other_blocks(f)
        f.close()
        f = None
        
    def write_intro(self,f,title):
        f.write(title+"\n")
        f.write(self.check_len(self.int_fmt.format(self.output)))
        f.write(self.check_len(self.int_fmt.format(self.axsym)))
        f.write("\n\n") # one blank line after intro

    def write_coordinates(self,f,coords,rounding=True):
        f.write(self.sp)
        # I use i and j here explicitly for clarity as 
        # this usage is consistent with the MEBS manual I and J
        for j in range(len(self.z_indices)):
            f.write(self.check_len(self.int_fmt.format(self.z_indices[j])))
        # equivalent: f.write(self.check_len((int_fmt*len(self.z_indices)).format(*self.z_indices)))
        f.write("\n")
        for i in range(len(self.r_indices)):
            f.write(self.check_len(self.int_fmt.format(self.r_indices[i],self.colwidth)))
            for j in range(len(self.z_indices)):
                coords_to_print = round(coords[i,j],self.colwidth-5) if rounding else coords[i,j]
                f.write(self.check_len(self.coord_fmt.format(coords_to_print)))
            f.write("\n")
        f.write("\n")
        
    def write_quad(self,f,z_indices,r_indices,quad_property,property_fmt):
        N = len(z_indices)
        for n in range(N):
            f.write(self.check_len_multi((self.int_fmt*2).format(*z_indices[n])))
            f.write(self.check_len_multi((self.int_fmt*2).format(*r_indices[n])))
            f.write(self.check_len(property_fmt.format(quad_property[n])))
            f.write("\n")
        f.write("\n")
        
    # these are dummy functions for the main class that are replaced 
    # by datafile-specific versions in the subclasses
    def read_other_blocks(self,line_num): 
        pass

    def write_other_blocks(self,f):
        pass
        
    def plot_mesh_coarse(self,quads_on=True,index_labels=True,adj=6,zlim=None,rlim=None):
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
        # plt.scatter(self.z.flatten(),self.r.flatten(),)
        for n in range(self.z.shape[0]):
            if(index_labels):
                # add r index label
                plt.text(self.z[n,:].max()+adj,self.r[n,np.argmax(self.z[n,:])],self.r_indices[n])
            # plot this iso-r-index line
            plt.plot(self.z[n,:],self.r[n,:],color='m')
        for n in range(self.z.shape[1]):
            if(index_labels):
                # add z index label
                plt.text(self.z[np.argmax(self.r[:,n]),n],self.r[:,n].max()+adj,self.z_indices[n])
            # plot this iso-z-index line
            plt.plot(self.z[:,n],self.r[:,n],color='m')
        self.plot_quads() if quads_on else 0
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
        plt.xlim(zlim)
        plt.ylim(rlim)
        # plt.title("Mesh (coarse)")
        plt.gca().set_aspect('equal')
        plt.show()
        
    def plot_mesh_segments(self,segments,quads_on=True):
        '''
        Plots mesh (coarse or fine) from a list of segments (Point1,Point2).

        Parameters:
            segments : list
                list of all segments to plot
            quads_on : boolean
                optional flag to also plot quads
        '''
        for segment in segments:
            plt.plot([segment[0].z,segment[1].z],[segment[0].r,segment[1].r],color='m')
        self.plot_quads() if quads_on else 0
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
        plt.gca().set_aspect('equal')
        plt.show()

    def plot_quads(self):
        pass
    
    def retrieve_segments(self,z_indices,r_indices):
        np_z_indices = np.nonzero((self.z_indices >= z_indices.min())*(self.z_indices <= z_indices.max()))[0]
        np_r_indices = np.nonzero((self.r_indices >= r_indices.min())*(self.r_indices <= r_indices.max()))[0]
        if(len(np_z_indices) == 0 or len(np_r_indices) == 0):
            raise ValueError('Quads that are not defined on the coarse mesh are not implemented.')
        seg_np_indices = []
        seg_np_indices.append((np.repeat(np_r_indices.min(),len(np_z_indices)),np_z_indices))
        seg_np_indices.append((np_r_indices,np.repeat(np_z_indices.max(),len(np_r_indices))))
        # reverse these two so that it's ccw order
        seg_np_indices.append((np.repeat(np_r_indices.max(),len(np_z_indices)),np_z_indices[::-1]))
        seg_np_indices.append((np_r_indices[::-1],np.repeat(np_z_indices.min(),len(np_r_indices))))
        return seg_np_indices
    
    # make a list of the numpy indices for points on the boundary of a given
    # type of quad. quad type is specificed by the inputs.
    # e.g. z_indices = self.mag_mat_z_indices or self.coil_z_indices for 
    # a magnetic lens.
    # with argument return_ind_array=True, reformats list into index arrays
    # to be called as self.r[unique_points]
    def retrieve_edge_points(self,z_indices,r_indices,return_ind_array=False):
        # points holds tuples of np indices for all points on the boundary
        points = []
        for n in range(len(r_indices)):
            for seg in self.retrieve_segments(z_indices[n],r_indices[n]):
                for m in range(len(seg[0])):
                     points.append((seg[0][m],seg[1][m])) 
        unique_points = list(dict.fromkeys(points)) # removes duplicate entries
        return index_array_from_list(unique_points) if return_ind_array else unique_points

    # same as above, but for a single quad
    def retrieve_single_quad_edge_points(self,quad_z_indices,quad_r_indices,return_ind_array=False):
        points = []
        for seg in self.retrieve_segments(quad_z_indices,quad_r_indices):
            for m in range(len(seg[0])):
                points.append((seg[0][m],seg[1][m]))
        unique_points = list(dict.fromkeys(points)) # removes duplicate entries
        return index_array_from_list(unique_points) if return_ind_array else unique_points

    def determine_quad_area(self,quad_z_indices,quad_r_indices):
        points = self.retrieve_single_quad_edge_points(quad_z_indices,quad_r_indices,return_ind_array=True)
        return calculate_area(self.z[points],self.r[points])

    # collect all segments in the coarse mesh into a list
    def define_coarse_mesh_segments(self):
        segments = []
        for i in range(self.z.shape[0]):
            for j in range(self.z.shape[1]):
                if(i+1 < self.z.shape[0]):
                    segments.append((Point(self.z[i,j],self.r[i,j]),Point(self.z[i+1,j],self.r[i+1,j])))
                if(j+1 < self.z.shape[1]):
                    segments.append((Point(self.z[i,j],self.r[i,j]),Point(self.z[i,j+1],self.r[i,j+1])))
        self.coarse_segments = segments
        return segments

    def define_fine_mesh_segments(self):
        segments = []
        r_interpolator = interp2d(self.z_indices,self.r_indices,self.r)
        z_interpolator = interp2d(self.z_indices,self.r_indices,self.z)
        # MEBS uses half-integer steps for the fine mesh
        step = 0.5
        for r_index in np.arange(self.r_indices[0],self.r_indices[-1]+step,step):
            # make coarse mesh-sized segments from fine mesh
            # skip last point as second point in segment doesn't exist
            for i,z_index in enumerate(self.z_indices[:-1]):
                segments.append((Point(z_interpolator(z_index,r_index),r_interpolator(z_index,r_index)),Point(z_interpolator(self.z_indices[i+1],r_index),r_interpolator(self.z_indices[i+1],r_index))))
        for z_index in np.arange(self.z_indices[0],self.z_indices[-1]+step,step):
            for i,r_index in enumerate(self.r_indices[:-1]):
                segments.append((Point(z_interpolator(z_index,r_index),r_interpolator(z_index,r_index)),Point(z_interpolator(z_index,self.r_indices[i+1]),r_interpolator(z_index,self.r_indices[i+1]))))
        self.fine_segments = segments
        return segments

    # makes the smallest possible fine mesh segments
    # not useful except for finding fine-fine intersections
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

    def plot_quad(self,z_indices,r_indices,color='k'):
        for seg in self.retrieve_segments(z_indices,r_indices):
            plt.plot(self.z[seg],self.r[seg],color=color)
        # np_z_indices = np.nonzero((self.z_indices >= z_indices.min())*(self.z_indices <= z_indices.max()))[0]
        # np_r_indices = np.nonzero((self.r_indices >= r_indices.min())*(self.r_indices <= r_indices.max()))[0]
        # seg_np_indices = []
        # seg_np_indices.append((np.repeat(np_r_indices.min(),len(np_z_indices)),np_z_indices))
        # seg_np_indices.append((np_r_indices,np.repeat(np_z_indices.max(),len(np_r_indices))))
        # seg_np_indices.append((np.repeat(np_r_indices.max(),len(np_z_indices)),np_z_indices))
        # seg_np_indices.append((np_r_indices,np.repeat(np_z_indices.min(),len(np_r_indices))))
        # for seg in seg_np_indices:
        #     plt.plot(self.z[seg],self.r[seg],color=color)
    
    def check_len(self,string):
        if(len(string.strip()) >= self.colwidth):
            raise Exception(f'Error: zero space between columns. Value: {string} with length {len(string)}, while column width is {self.colwidth}. Increase column width and rerun.')
        else:
            return string

    def check_len_multi(self,string):
        for item in string.split():
            if(len(item) >= self.colwidth):
              raise Exception('Error: zero space between columns. Increase column width and rerun.')
        return string

    def plot_field(self):
        '''
        Plots the magnetic field calculated by MEBS. Run after calc_field().

        No parameters. 
        '''
        try:
            with cd(self.dirname):
                z,B = np.genfromtxt(self.potname,dtype=float,skip_header=7,skip_footer=4,unpack=True)
                plt.plot(z,B)
                plt.xlabel('z (mm)')
                plt.ylabel('B (T)')
                plt.show()
        except OSError:
            print('No file with name {} found. Run calc_field first.'.format(self.potname))
            raise FileNotFoundError

    def add_curvature(self):
        '''
        Converts optical element .dat file used for optics into a file 
        compatible with MEBS SOFEM by adding curvature coordinates.
        
        No arguments.
        '''

        if(hasattr(self,'r_curv')):
            raise Exception('Curvatures already defined.')
        self.r_curv = np.zeros_like(self.r)
        self.z_curv = np.zeros_like(self.z)
        self.so = True

    def calc_rays(self):
        '''
        Run after write_raytrace_file() to calculate rays.

        No parameters.
        '''
        with cd(self.dirname):
            try:
                print(subprocess.run(["soray.exe",self.raytracebasename_noext],stdout=subprocess.PIPE,timeout=self.timeout).stdout.decode('utf-8'))
            except TimeoutExpired:
                print('Ray tracing timed out. Rerunnning.')
                self.calc_rays()

    def write_raytrace_file(self,mircondfilename,source_pos=90,source_size=1000,semiangle=10,energy=200000,initial_direction=180,lens_type='Electrostatic',lens_pos=0,lens_excitation=None,potentials=None,screen_pos=95,relativity=False,cyl_symm=True,r_samples=3,alpha_samples=3,precision=6,n_equipotentials=50):
        '''
        Creates an input file for SORAY.exe. Primarily for visualizing columns
        implemented in MIRROR. All physical parameters have same name, units
        and default as in write_mir_img_cond_file(), even when SORAY does not
        share units with MIRROR.

        Parameters:
            mircondfilename : path
                full filename to write imaging conditions file to.

        Optional parameters:
            source_pos : float
                Source z position (mm). The "source" is the starting 
                position of rays for auto-focusing. 
                Default 90.
            source_size: float
                Source size (microns). Not related to resolution; just the
                starting size of the ray bundle.
                Default 1000.
            semiangle : float
                Semiangle (mrad) for emission from source.
                Default 10.
            energy : float
                Beam kinetic energy (eV). Default 200000.
            initial_direction : float
                Polar angle (deg) of initial direction. 0 for forwards 
                propagation or 180 for reverse. 
                Default 180.
            lens_type : str
                Type of lens: 'Electrostatic' or 'Magnetic'. Multiple
                lenses not yet implemented.
                Default 'Electrostatic'.
            lens_pos : float
                Lens z position (mm). Default 10. **
            lens_excitation : string
                Specifies excitation strength of magnetic round lens or
                magnetic or electric multipole. Contains a floating point 
                number and a flag. Units are A-turns for magnetic or volts for 
                electric. Flag options are f for fixed, vn for variable, where
                n is an integer grouping lenses varied together (e.g. v1)
                during autofocusing, or d for dynamic. The purpose of the 
                dynamic option is unclear. Default None for unused.
            potentials : MirPotentials instance
                The MirPotentials class is defined in the ElecLens class.
                This class is used to sensibly store and format the string
                used for specifying the potentials of several electrodes.
                Default None for unused. **
            screen_pos : float
                Screen plane z position (mm). End position of rays for auto-
                focusing. Default 95.
            relativity : bool
                Determines whether relativistic effects are included. 
                Default False.
            cyl_symm : bool
                Determines whether to use 2D initial positions for rays.
                Default True.
            r_samples : int
                Number of samples of initial positions in the radial direction.
                Default 3.
            alpha_samples : int
                Number of samples of initial polar angles. Default 3.
            n_equipotentials : int
                Number of equally-spaced equipotentials to plot.
                Default 50.
        '''
        # SORAY uses mm
        x_positions = np.linspace(-source_size/2/1000,source_size/2/1000,r_samples,endpoint=True)
        if(cyl_symm):
            y_positions = np.array([0])
        else:
            y_positions = np.linspace(-source_size/2/1000,source_size/2/1000,r_samples,endpoint=True)

        # SORAY uses degrees
        angles = initial_direction + np.linspace(-semiangle*180/np.pi/1000,semiangle*180/np.pi/1000,alpha_samples,endpoint=True)

        self.mircondbasename_noext = os.path.splitext(os.path.basename(mircondfilename))[0] 
        self.raytracefile = os.path.join(self.dirname,self.mircondbasename_noext+'_rays'+'.dat')
        self.raytracebasename_noext = os.path.splitext(os.path.basename(self.raytracefile))[0] 
        self.mircondfloat_fmt = self.rfloat_fmt.substitute(imgcondcolwidth=self.imgcondcolwidth,precision=precision)
        cf = open(self.raytracefile,'w') 
        cf.write(f'Title raytrace file for {mircondfilename}\n\n')
        cf.write(f'\n{lens_type} lens\n')
        cf.write(self.imgcondsubprop_fmt.format("Filename")+self.imgcondtext_fmt.format(self.basename_noext)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(lens_pos)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Potentials")+self.imgcondtext_fmt.format(potentials.format_noflag())+"\n")
        cf.write('\n')
        cf.write(self.imgcondsubprop_fmt.format("Time step factor")+self.mircondfloat_fmt.format(0.1)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Screen plane")+self.mircondfloat_fmt.format(screen_pos)+"\n")
        relativity_str = 'on' if relativity else 'off'
        cf.write(self.imgcondsubprop_fmt.format("Relativity")+self.imgcondtext_fmt.format(relativity_str)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Save rays")+self.imgcondtext_fmt.format('on')+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Save xyz")+self.imgcondtext_fmt.format('off')+"\n")
        cf.write('\nInitial ray conditions\n')
        rayfloat_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=precision)
        for x in x_positions:
            for y in y_positions:
                for alpha in angles:
                    cf.write(self.check_len(rayfloat_fmt.format(x))+
                             self.check_len(rayfloat_fmt.format(y))+
                             self.check_len(rayfloat_fmt.format(source_pos))+
                             self.check_len(rayfloat_fmt.format(energy))+
                             self.check_len(rayfloat_fmt.format(alpha))+
                             self.check_len(rayfloat_fmt.format(0))+ # azimuthal angle
                             '\n')

        if(potentials is not None):
            pot_min = min(potentials.voltages)
            pot_max = max(potentials.voltages)
            pot_range = np.linspace(pot_min,pot_max,n_equipotentials,endpoint=True)
            cf.write('\nElectrostatic Equipotentials\n')
            for pot in pot_range:
                cf.write(rayfloat_fmt.format(pot)+'\n')
        cf.close()
        cf = None


    def write_mir_img_cond_file(self,mircondfilename,source_pos=90,source_shape='ROUND',source_size=200,intensity_dist='UNIFORM',ang_shape='ROUND',semiangle=10,ang_dist='UNIFORM',energy=200000,energy_width=1,energy_dist='Gaussian',lens_type='electrostatic',lens_pos=0,lens_scale=1,lens_excitation=None,potentials=None,ray_method="R",order=3,focus_mode="AUTO",img_pos=95,screen_pos=None,mir_screen_pos=None,save_trj=True,obj_pos=None,obj_semiangle=None,x_size=0.1,y_size=0.1,reverse_dir=True,turning_point=5,precision=6):
        '''
        Writes optical imaging conditions file for MIRROR. Must be run before
        calc_properties_mirror(). 
        
        All parameters are specified in more detail starting on p. 55 of the 
        IMAGE-GUI v3.1 manual. Note that parameters listed as optional below
        with two asterisks (**) are technically optional but in general should
        be specified as MIRROR may not successfully run with default inputs.
        Note also that this is not a comprehensive list of all possible MIRROR
        parameters, but rather a limited selection appropriate for the present
        application.

        Parameters:
            mircondfilename : path
                full filename to write imaging conditions file to.

        Optional parameters:
            source_pos : float
                Source z position (mm). The "source" is the starting 
                position of rays for auto-focusing. 
                Default 0. **
            source_shape : string
                Shape of source. Options in MEBS are "SQUARE","ROUND",
                "OBLONG" and "OVAL", but the latter two require separate 
                x and y source size, which is not implemented here.
                Default "ROUND".
            source_size: float
                Source size (microns). Not related to resolution; just the
                starting size of the ray bundle.
                Default 200.
            intensity_dist : string
                Shape of spatial intensity distribution. Options are "UNIFORM"
                and "GAUSSIAN".
                Default "Gaussian".
            ang_shape : string
                Angular shape of source. Options are "SQUARE", "ROUND", 
                "OBLONG" and "OVAL", but the latter two require separate
                x and y source size, which is not implemented here.
                Default "ROUND".
            semiangle : float
                Semiangle (mrad) for emission from source.
                Default 10.
            ang_dist : string
                Angular distribution of source. Options are "UNIFORM",
                "GAUSSIAN" or "LAMBERTIAN". Default "UNIFORM".
            energy : float
                Beam kinetic energy (eV). Default 200000.
            energy_width : float
                Beam energy spread (eV). Default 1.
            energy_dist : string
                Energy distribution of source. Options are "UNIFORM", 
                "GAUSSIAN", "MAXWELL-BOLTZMANN", or "SECONDARY". Default
                "GAUSSIAN".
            lens_type : string
                Specifies type of lens. Multiple lenses to be implemented.
                Options are "MAGNETIC" and "ELECTROSTATIC". Default
                "ELECTROSTATIC".
            lens_pos : float
                Lens z position (mm). Default 10. **
            lens_scale : float
                Scale factor to be applied to spatial extent of lens. Default 1.
            lens_excitation : string
                Specifies excitation strength of magnetic round lens or
                magnetic or electric multipole. Contains a floating point 
                number and a flag. Units are A-turns for magnetic or volts for 
                electric. Flag options are f for fixed, vn for variable, where
                n is an integer grouping lenses varied together (e.g. v1)
                during autofocusing, or d for dynamic. The purpose of the 
                dynamic option is unclear. Default None for unused.
            potentials : MirPotentials instance
                The MirPotentials class is defined in the ElecLens class.
                This class is used to sensibly store and format the string
                used for specifying the potentials of several electrodes.
                Default None for unused. **
            ray_method : string
                Specifies whether rays are computed cylindrically symmetrically
                ("R") or in full x-y-z space ("XY"). Default "R".
            order : integer
                Maximum power for order of axial field functions. Default 3.
            focus_mode : string
                Specifies whether MIRROR does manual or auto-focusing. Default 
                is "AUTO".
            img_pos : float
                Image plane z position (mm). Default 0. **
            screen_pos : float
                Screen plane z position (mm). End position of rays for auto-
                focusing. Default None, copied from img_pos. **
            mir_screen_pos : float
                Mirror screen plane z position (mm). End position of DA rays
                for aberration calculation. Default None, copied from 
                screen_pos.
            save_trj : bool
                Saves trajectories file. Not documented in manual. Default True.
            obj_pos : float
                Mirror object z position (mm). Start position of DA rays for
                aberration calculation. Default None, copied from source_pos. 
            obj_semiangle : float
                awaiting clarification. Default None, copied from semiangle.
            x_size : float
            y_size : float
                Extent of object (mm) used to launch DA rays for aberration
                calculation. Default 0.1
            reverse_dir : bool
                If True, beam will initially propagate in the negative z 
                direction. Default False.
            turning_point : float
                Estimated turning point along z (mm). MEBS will not successfully
                run if this is not within roughly +/-10mm of the turning point.
                Default 50. **
            precision : int
                Number of decimal places to print floats with.
        '''
        self.program = 'mirror'
        if(obj_pos == None):
            obj_pos = source_pos
        if(screen_pos == None):
            screen_pos = img_pos
        if(mir_screen_pos == None):
            mir_screen_pos = screen_pos
        if(obj_semiangle == None):
            obj_semiangle = semiangle
        if(reverse_dir):
            mir_energy = -energy
            energy = -energy
        else:
            mir_energy = energy

        self.mircondfloat_fmt = self.rfloat_fmt.substitute(imgcondcolwidth=self.imgcondcolwidth,precision=precision)
        self.mircondfilename = mircondfilename
        self.mircondbasename_noext = os.path.splitext(os.path.basename(mircondfilename))[0] 
        cf = open(self.mircondfilename,'w') 
        self.mir_cond_title = self.title+' Imaging Conditions for MIRROR'

        cf.write(f"Title     {self.mir_cond_title:>70}\n\n")
        cf.write("SOURCE\n")
        cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(source_pos)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Shape")+self.imgcondtext_fmt.format(source_shape)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Size")+self.mircondfloat_fmt.format(source_size)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Intensity distribution")+self.imgcondtext_fmt.format(intensity_dist)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Angular shape")+self.imgcondtext_fmt.format(ang_shape)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Half angle")+self.mircondfloat_fmt.format(semiangle)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Angular distribution")+self.imgcondtext_fmt.format(ang_dist)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Beam energy")+self.mircondfloat_fmt.format(energy)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Energy width parameter")+self.mircondfloat_fmt.format(energy_width)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Energy distribution")+self.imgcondtext_fmt.format(energy_dist)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Beam current")+self.mircondfloat_fmt.format(1)+"\n")
        cf.write("\nLENS\n")
        cf.write(self.imgcondsubprop_fmt.format("File")+self.imgcondtext_fmt.format(self.fitname)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Type")+self.imgcondtext_fmt.format(lens_type)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(lens_pos)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Size")+self.mircondfloat_fmt.format(lens_scale)+"\n")
        if(lens_excitation is not None): 
            cf.write(self.imgcondsubprop_fmt.format("Excitation")+self.mircondfloat_fmt.format(lens_excitation)+"\n")
        if(potentials is not None): 
            cf.write(self.imgcondsubprop_fmt.format("Potentials")+self.imgcondtext_fmt.format(potentials.format())+"\n")
        cf.write("\nGAUSSIAN IMAGE PLANE\n")
        cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(img_pos)+"\n")
        cf.write("\nSCREEN\n")
        cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(screen_pos)+"\n")
        cf.write("\nPARTICLES\n")
        cf.write(self.imgcondsubprop_fmt.format("Type")+self.imgcondtext_fmt.format("Electrons")+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Charge")+self.imgcondint_fmt.format(1)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Mass")+self.imgcondint_fmt.format(1)+"\n")
        cf.write("\nSIMULATION PARAMETERS\n")
        cf.write(self.imgcondsubprop_fmt.format("Initial Conditions")+self.imgcondtext_fmt.format("Systematic")+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Particles/bunch")+self.imgcondint_fmt.format(1)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Bunches")+self.imgcondint_fmt.format(1)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Seed")+self.imgcondint_fmt.format(1)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Error bound per step")+self.mircondfloat_fmt.format(1e-12)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Paraxial Ray Method")+self.imgcondtext_fmt.format(ray_method)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Order")+self.imgcondint_fmt.format(order)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Coulomb interactions")+self.imgcondtext_fmt.format("off")+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Simulation method")+self.imgcondtext_fmt.format("direct")+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Tree code parameter")+self.imgcondint_fmt.format(1)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Focus mode")+self.imgcondtext_fmt.format(focus_mode)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Interactive")+self.imgcondtext_fmt.format("no")+"\n")
        trjsavestr = 'yes' if save_trj else 'no'
        cf.write(self.imgcondsubprop_fmt.format("Save trajectories")+self.imgcondtext_fmt.format(trjsavestr)+"\n")
        cf.write("\nMIRROR OBJECT\n")
        cf.write(self.imgcondsubprop_fmt.format("Position (mm)")+self.mircondfloat_fmt.format(obj_pos)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Beam energy (eV)")+self.mircondfloat_fmt.format(mir_energy)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Alpha (mrad)")+self.mircondfloat_fmt.format(obj_semiangle)+"\n")
        # alpha values and azimuth values are unused in MIRROR
        cf.write(self.imgcondsubprop_fmt.format("Alpha values")+self.imgcondint_fmt.format(2)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Azimuth values")+self.imgcondint_fmt.format(16)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("X Size (mm)")+self.mircondfloat_fmt.format(x_size)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Y Size (mm)")+self.mircondfloat_fmt.format(y_size)+"\n")
        cf.write("\nMIRROR SCREEN\n")
        cf.write(self.imgcondsubprop_fmt.format("Position (mm)")+self.mircondfloat_fmt.format(mir_screen_pos)+"\n")
        cf.write("\nTURNING POINTS GUIDELINE\n")
        cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(turning_point)+"\n")
        cf.close()
        cf = None

    def write_opt_img_cond_file(self,imgcondfilename,n_intervals=200,energy=200000,energy_width=1,aperture_angle=30,obj_pos=0,img_pos=6,n_intermediate_images=0,lens_pos=0,lens_strength=1,lens_scale=1,precision=6,auto_focus=1):
        '''
        Writes optical imaging conditions file for OPTICS. Must be run before 
        calc_properties_optics().

        Parameters:
            imgcondfilename : path
                full filename to write imaging conditions file to.

        Optional parameters:
            n_intervals : int
                number of integration steps along z.
                default 200
            energy : float 
                electron energy in eV.
                default 200000
            energy_width : float
                electron energy width in eV.
                default 1
            aperture_angle : float
                semi-angle in mrad at image plane.
                default 30
            obj_pos : float
                object plane (z position in mm).
                default 0
            img_pos : float
                image plane (z position in mm).
                default 6
            n_intermediate_images : int
                number of intermediate image planes that should occur before the specified plane.
                default 0
            lens_pos :  float
                lens z position (in mm).
                default 0
            lens_strength : float
                scaling factor for the lens strength. should do nothing with
                autofocusing on.
                default 1
            precision : int
                number of digits with which to save floats.
                default 6.
            auto_focus : boolean integer
                if 1, MEBS ignores specified lens currents/voltages and 
                auto-focuses to specified image plane.
                if 0 and no image plane specified, uses specified currents in 
                optical element .dat file. unclear what happens if image plane
                is specified and auto_focus=1.
                default 1 and is highly recommended as MEBS will throw pop-ups
                asking about which image plane to use for computing optical
                properties if the lens strength is high enough to create 
                multiple image planes. 
        '''
        self.program = 'optics'
        self.imgcondfloat_fmt = self.rfloat_fmt.substitute(imgcondcolwidth=self.imgcondcolwidth,precision=precision)
        self.lensfloat_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=precision)
        self.imgcondfilename = imgcondfilename
        self.imgcondbasename_noext = os.path.splitext(os.path.basename(imgcondfilename))[0] 
        cf = open(self.imgcondfilename,'w') 
        self.img_cond_title = self.title+' Imaging Conditions for OPTICS+ABER5'
        on = 'on'
        off = 'off'
        cf.write(f"Title     {self.img_cond_title:>70}\n\n")
        cf.write(self.imgcondprop_fmt.format("Fifth Order Aberration")+self.imgcondtext_fmt.format(on)+'\n')
        cf.write(self.imgcondprop_fmt.format("Slope Aberration")+self.imgcondtext_fmt.format(on)+'\n')
        cf.write(self.imgcondprop_fmt.format("Output Paraxial Rays")+self.imgcondtext_fmt.format(on)+'\n')
        cf.write(self.imgcondprop_fmt.format("Terminal Display")+self.imgcondtext_fmt.format(on)+'\n')
        cf.write(self.imgcondprop_fmt.format("Lens Properties")+self.imgcondtext_fmt.format(on)+'\n')
        cf.write(self.imgcondprop_fmt.format("Number of Intervals")+self.imgcondint_fmt.format(n_intervals)+'\n')
        # cf.write(self.imgcondprop_fmt.format("Number of Intermediate Images")+self.imgcondint_fmt.format(n_intermediate_images)+'\n')
        cf.write('\n')
        cf.write(self.imgcondprop_fmt.format("Object Plane")+self.imgcondfloat_fmt.format(obj_pos)+'\n')
        cf.write(self.imgcondprop_fmt.format("Image Plane")+self.imgcondfloat_fmt.format(img_pos)+'\n')
        cf.write('\n')
        cf.write(self.imgcondprop_fmt.format("Aperture Angle")+self.imgcondfloat_fmt.format(aperture_angle)+'\n')
        cf.write(self.imgcondprop_fmt.format("Energy Spread")+self.imgcondfloat_fmt.format(energy_width)+'\n')
        cf.write(self.imgcondprop_fmt.format("Beam Voltage")+self.imgcondfloat_fmt.format(energy)+'\n')
        cf.write('\n')
        cf.write('Magnetic Lens\n')
        cf.write('\n')
        cf.write(self.lensfloat_fmt.format(lens_pos)+self.lensfloat_fmt.format(lens_strength))
        cf.write(self.lensfloat_fmt.format(lens_scale)+self.int_fmt.format(auto_focus)+"{:>40s}".format(self.potname)+'\n')
        cf.write('\n')
        cf.close()
        cf = None

    # this is bound to break when the .res file changes 
    # in ways I haven't foreseen. fix as needed.
    def read_optical_properties(self):
        '''
        Run after calc_properties_optics() to read in the computed optical 
        properties.

        No arguments.
        '''
        pf = open(os.path.join(self.dirname,self.imgcondbasename_noext+'.res'),'r')
        properties_lines = pf.readlines()
        # see end of this file for snippets of the .res file 
        # that are relevant to this parameter extraction
        for i,line in enumerate(properties_lines):
            if 'FIRST-ORDER PROPERTIES' in line:
                linenum_mag = i+10
                linenum_rot = i+11
                linenum_curr = i+5
            if 'Magnetic Lens      No.  1' in line: # change if more lenses
                linenum_f = i+9
                linenum_f_real = i+5
            if 'THIRD-ORDER ABERRATION COEFFICIENTS   (in S.I. units)' in line:
                linenum_c3 = i+9
            if ' ***** CHROMATIC AB *****' in line:
                linenum_cc = i+3
        self.mag = float(properties_lines[linenum_mag].split()[3])
        self.rot = float(properties_lines[linenum_rot].split()[5]) # deg
        self.lens_curr = float(properties_lines[linenum_curr].split()[7])
        ## I didn't need to implement this yet
        # self.lens_curr = []
        # i = 0
        # lens_curr_line = properties_lines[linenum_curr+i].split()
        # while(len(lens_curr_line) == 8):
        #     self.lens_curr.append(float(lens_curr_line[7]))
        #     i += 1
        #     lens_curr_line = properties_lines[linenum_curr+i].split()
        self.f = float(properties_lines[linenum_f].split()[8]) # mm 
        self.f_real = float(properties_lines[linenum_f_real].split()[8]) # mm
        self.c3 = float(properties_lines[linenum_c3].split()[2])*1e3 # m to mm
        self.cc = float(properties_lines[linenum_cc].split()[1])*1e3 # m to mm
        pf.close()
        pf = None

    # this is bound to break when the .res file changes 
    # in ways I haven't foreseen. fix as needed.
    def read_mir_optical_properties(self):
        '''
        Run after calc_properties_mirror() to read in the computed optical 
        properties.

        No arguments.
        '''
        pf = open(os.path.join(self.dirname,self.mircondbasename_noext+'.res'),'r')
        properties_lines = pf.readlines()
        # see end of this file for snippets of the .res file 
        # that are relevant to this parameter extraction
        for i,line in enumerate(properties_lines):
            if 'Results of 1st order Properties' in line:
                linenum_mag = i+6
                linenum_rot = i+7
            if 'Optical Element Settings After Focusing' in line:
                # assumes single electrostatic lens
                linenum_v = i+3
            if 'Turning point z(mm)' in line:
                linenum_turning = i
            if 'Results of 3rd order Calculation' in line:
                linenum_c3 = i+3
            if 'Chromatic aberration coefficients (2nd rank)' in line:
                linenum_cc = i+1
        self.mag = float(properties_lines[linenum_mag].split()[2])
        self.rot = float(properties_lines[linenum_rot].split()[4]) # deg
        self.lens_curr = None # not used 
        self.v = []
        j = 0
        while('Potential V' in properties_lines[linenum_v+j]):
            self.v.append(float(properties_lines[linenum_v+j].split()[3]))
            j+=1
        self.f = None
        self.f_real = None
        self.c3 = float(properties_lines[linenum_c3].split()[0]) # m to mm
        self.cc = float(properties_lines[linenum_cc].split()[0]) # m to mm
        pf.close()
        pf = None


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
    
    
    def read_other_blocks(self,line_num):
        self.potname = self.basename_noext+'.axb' # name of potential file
        line_num = self.read_mag_mat(line_num)
        line_num = self.read_coil(line_num)
        self.plot_mesh_coarse(quads_on=True) if self.verbose else 0
        while(self.infile[line_num].isspace() != True):
            line_num = self.read_hyst_curve(line_num)
            self.plot_hyst() if self.verbose else 0
    
    def read_mag_mat(self,line_num):
        return self.read_quad(line_num,self.mag_mat_z_indices,self.mag_mat_r_indices,self.mag_mat_curve_indices,property_dtype=int)
#         while(self.infile[line_num].isspace() != True):
#             line = np.fromstring(self.infile[line_num],dtype=int,count=5,sep=' ')
#             self.mag_mat_z_indices.append(line[:2])
#             self.mag_mat_r_indices.append(line[2:4])
#             self.mag_mat_curve_indices.append(line[4])
#             line_num+=1
#         return line_num+1 # start of next block
    
    def read_coil(self,line_num):
        return self.read_quad(line_num,self.coil_z_indices,self.coil_r_indices,self.coil_curr,property_dtype=float)
#         while(self.infile[line_num].isspace() != True):
#             line = np.fromstring(self.infile[line_num],dtype=float,count=5,sep=' ')
#             self.coil_z_indices.append(line[:2].astype(int))
#             self.coil_r_indices.append(line[2:4].astype(int))
#             self.coil_curr.append(line[4])
#             line_num+=1
#         return line_num+1 # start of next block
    
    def read_hyst_curve(self,line_num):
        lines = []
        while(self.infile[line_num].isspace() != True):
            lines.append(np.fromstring(self.infile[line_num],dtype=float,count=2,sep=' '))
            line_num+=1
        lines = np.array(lines)
        self.H_arrays.append(lines[:,0])
        self.B_arrays.append(lines[:,1])
        return line_num+1 # start of next block
    
    def plot_hyst(self,i=-1):
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
        
    def write_other_blocks(self,f):
        self.write_mag_mat(f)
        self.write_coil(f)
        self.write_hyst(f) # leaves one blank line at end of section
        f.write("\n\n") # three blank lines for a strong magnetic lens
    
    def write_mag_mat(self,f):
        self.write_quad(f,self.mag_mat_z_indices,self.mag_mat_r_indices,self.mag_mat_curve_indices,self.int_fmt)
    
    def write_coil(self,f):
        self.write_quad(f,self.coil_z_indices,self.coil_r_indices,self.coil_curr,self.curr_fmt)

    def write_hyst(self,f):
        M = len(self.H_arrays)
        for m in range(M):
            for k in range(len(self.H_arrays[m])):
                f.write(self.check_len(self.field_fmt.format(self.H_arrays[m][k])))
                f.write(self.check_len(self.field_fmt.format(self.B_arrays[m][k])))
                f.write("\n")
            f.write("\n")
            
    def plot_quads(self):
        self.plot_mag_mat()
        self.plot_coil()
    
    def plot_mag_mat(self):
        N = len(self.mag_mat_r_indices)
        for n in range(N):
            self.plot_quad(self.mag_mat_z_indices[n],self.mag_mat_r_indices[n],color='k')
            
    def plot_coil(self):
        L = len(self.coil_r_indices)
        for l in range(L):
            self.plot_quad(self.coil_z_indices[l],self.coil_r_indices[l],color='r')

    def calc_field(self):
        '''
        Calls somlenss.exe to calculate magnetic field for this lens.

        No arguments.
        '''
        with cd(self.dirname):
            outputmode = subprocess.PIPE if self.verbose else None
            try:
                output = subprocess.run(["somlenss.exe",self.basename_noext],stdout=outputmode,timeout=self.timeout).stdout
                print(output.decode('utf-8')) if self.verbose else None
            except TimeoutExpired:
                print('Field calculation timed out. Rerunning.')
                self.calc_field()
        # now = datetime.datetime.now()
        # # check if potential file exists and was created in last five minites
        # if(os.path.exists(self.filename_noext+'.axb') and 
        #         now - datetime.timedelta(minutes=5) < datetime.datetime.fromtimestamp(os.path.getmtime(self.filename_noext+'.axb')) < now):
        #     print('Field computation successful.')
        # else:
        #     raise Exception('Field computation failed. Run SOFEM GUI on this file for error message.')


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
        
    
    def read_other_blocks(self,line_num):
        line_num = self.read_mag_mat(line_num)
        line_num = self.read_coil(line_num)
        self.plot_mesh_coarse(quads_on=True) if self.verbose else 0
    
    def read_mag_mat(self,line_num):
        return self.read_quad(line_num,self.mag_mat_z_indices,self.mag_mat_r_indices,self.mag_mat_mu_r,property_dtype=float)
        
    def write_other_blocks(self,f):
        self.write_mag_mat(f)
        self.write_coil(f)
        f.write("\n\n") # two blank lines for a weak magnetic lens
    
    def write_mag_mat(self,f):
        self.write_quad(f,self.mag_mat_z_indices,self.mag_mat_r_indices,self.mag_mat_mu_r,self.rel_perm_fmt)

    def calc_field(self):
        '''
        Calls somlensc.exe to calculate magnetic field for this lens.

        No arguments.
        '''
        with cd(self.dirname):
            try:
                print(subprocess.run(["somlensc.exe",self.basename_noext],stdout=subprocess.PIPE,timeout=self.timeout).stdout.decode('utf-8'))
            except TimeoutExpired:
                print('Field calculation timed out. Rerunning.')
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

    
    def read_other_blocks(self,line_num):
        line_num = self.read_mag_mat(line_num)
        line_num += 1 # extra blank line here for scalar magnetic potential files
        line_num = self.read_coil(line_num)
        # when properly implemented, plot_mesh_coarse would also represent 
        # boundary currents in some way
        self.plot_mesh_coarse(quads_on=True) if self.verbose else 0
        
    def write_other_blocks(self,f):
        self.write_mag_mat(f) # leaves one blank line at end of section
        f.write("\n") # extra blank line here for scalar potential files
        self.write_coil(f) # leaves one blank line at end of section
        f.write("\n") # two total blank lines for a weak magnetic lens
        
    def read_coil(self,line_num):
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
        self.boundary_coil_indices = np.split(self.boundary_coil_indices,section_starts)
        self.boundary_coil_currents = np.split(self.boundary_coil_currents,section_starts)
        return line_num+1 # start of next block
    
    def write_coil(self,f):
        M = len(self.boundary_coil_indices)
        for m in range(M):
            for k in range(len(self.boundary_coil_indices[m])):
                f.write(self.check_len(self.int_fmt.format(self.boundary_coil_indices[m][k])))
                f.write(self.check_len(self.curr_fmt.format(self.boundary_coil_currents[m][k])))
        f.write("\n")

    def plot_quads(self):
        self.plot_mag_mat()

    def calc_field(self):
        '''
        Calls somlensp.exe to calculate magnetic field for this lens.

        No arguments.
        '''
        with cd(self.dirname):
            try:
                print(subprocess.run(["somlensp.exe",self.basename_noext],stdout=subprocess.PIPE,timeout=self.timeout).stdout.decode('utf-8'))
            except TimeoutExpired:
                print('Field calculation timed out. Rerunning.')
                self.calc_field()

class ElecLens(OpticalElement):
    '''
    Class for electrostatic lenses and mirrors.
    
    '''

    class MirPotentials:
        def __init__(self,parent,voltages,flags,voltage_precision=6):
            self.parent = parent
            self.voltages = voltages
            self.flags = flags
            self.voltage_fmt = self.parent.rfloat_fmt.substitute(imgcondcolwidth=parent.colwidth,precision=voltage_precision)
            self.string = ''
            if(len(flags) != len(voltages)):
                raise ValueError('Lengths of voltage and flag arrays not equal.')

        def format(self):
            if(self.string == ''): # if not set yet, set
                for i in range(len(self.voltages)):
                    self.string += self.parent.check_len(self.voltage_fmt.format(self.voltages[i]) + self.flags[i])
            return self.string

        def format_noflag(self):
            return self.parent.check_len_multi((self.parent.int_fmt*len(self.voltages)).format(*self.voltages))

    def lens_type(self,mirror,curved_mirror):
        '''
        Run after initializing ElecLens to classify lens.

        Parameters:
            mirror : bool
                Indicates whether optical element is a mirror.
            curved_mirror : bool
                Indicates whether mirror is curved.
        '''
        self.mirror = mirror
        self.curved_mirror = curved_mirror

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
    
    
    def read_other_blocks(self,line_num):
        self.potname = self.basename_noext+'.axv' # name of potential file
        line_num = self.read_electrodes(line_num)
        line_num = self.read_dielectrics(line_num)
        self.plot_mesh_coarse(quads_on=True) if self.verbose else 0
        line_num = self.read_boundaries(line_num)
    
    # read_quad with one line tweaked
    def read_electrodes(self,line_num):
        while(self.infile[line_num].isspace() != True):
            indices = np.fromstring(self.infile[line_num],dtype=int,count=4,sep=' ')
            self.electrode_z_indices.append(indices[:2])
            self.electrode_r_indices.append(indices[2:4])
            self.electrode_unit_potentials.append(np.fromstring(self.infile[line_num],dtype=float,sep=' ')[4:])
            line_num+=1
        return line_num+1 # start of next block
    
    def read_dielectrics(self,line_num):
        return self.read_quad(line_num,self.dielectric_z_indices,self.dielectric_r_indices,self.dielectric_constants,property_dtype=float)

    def read_boundaries(self,line_num):
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
        self.boundary_indices = np.split(self.boundary_indices,section_starts)
        self.boundary_unit_potentials = np.split(self.boundary_unit_potentials,section_starts)
    
    def write_other_blocks(self,f):
        self.write_electrodes(f)
        self.write_dielectrics(f)
        self.write_boundaries(f) # leaves one blank line at end of section
        f.write("\n") # two total blank lines for an electrostatic lens
    
    def write_electrodes(self,f):
        N = len(self.electrode_z_indices)
        M = len(self.electrode_unit_potentials[0])
        for n in range(N):
            f.write(self.check_len_multi((self.int_fmt*2).format(*self.electrode_z_indices[n])))
            f.write(self.check_len_multi((self.int_fmt*2).format(*self.electrode_r_indices[n])))
            f.write(self.check_len_multi((self.voltage_fmt*M).format(*self.electrode_unit_potentials[n])))
            f.write("\n")
        f.write("\n")
    
    def write_dielectrics(self,f):
        self.write_quad(f,self.dielectric_z_indices,self.dielectric_r_indices,self.dielectric_constants,self.rel_perm_fmt)
    def write_boundaries(self,f):
        M = len(self.electrode_unit_potentials[0])
        for p in range(len(self.boundary_indices)): # iterate sections
            for q in range(len(self.boundary_indices[p])): # iterate indices in sections
                f.write(self.check_len(self.int_fmt.format(self.boundary_indices[p][q])))
                f.write(self.check_len_multi((self.voltage_fmt*M).format(*self.boundary_unit_potentials[p][q])))
                f.write("\n")
        f.write("\n")
        # do this

    def plot_quads(self):
        self.plot_electrodes()
        self.plot_dielectrics()
    
    def plot_electrodes(self):
        N = len(self.electrode_r_indices)
        for n in range(N):
            self.plot_quad(self.electrode_z_indices[n],self.electrode_r_indices[n],color='k')
            
    def plot_dielectrics(self):
        L = len(self.dielectric_r_indices)
        for l in range(L):
            self.plot_quad(self.dielectric_z_indices[l],self.dielectric_r_indices[l],color='r')

    def calc_field(self):
        '''
        Calls soelens.exe to calculate electric field for this lens.

        No arguments.
        '''
        with cd(self.dirname):
            outputmode = subprocess.PIPE if self.verbose else None
            try:
                output = subprocess.run(["soelens.exe",self.basename_noext],stdout=outputmode,timeout=self.timeout).stdout
                print(output.decode('utf-8')) if self.verbose else None
            except TimeoutExpired:
                print('Field calculation timed out. Rerunning.')
                self.calc_field()


# example_strong = strong_mag_lens("/home/trh/MEBS/OPTICS/dat/OPTICS/Elements/MAG/LENS/mlenss1.dat",verbose=True)
# example_strong.write("/home/trh/data/test_strong.dat","a test")
# 
# example_weak = weak_mag_lens("/home/trh/MEBS/OPTICS/dat/OPTICS/Elements/MAG/LENS/mlensc1.dat",verbose=True)
# example_weak.write("/home/trh/data/test_weak.dat","a test")
# 
# example_pp = weak_mag_lens_pp_region("/home/trh/MEBS/OPTICS/dat/OPTICS/Elements/MAG/LENS/mlensp1.dat",verbose=True)
# example_pp.write("/home/trh/data/test_weak_pp.dat","a test")
# 

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
