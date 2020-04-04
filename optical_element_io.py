#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from string import Template


# definitions for comments:
# quad : four-pointed object used to define magnetic materials, coils, electrodes, etc. in MEBS

def np_indices(indices,index_set):
    np_index_array = []
    for index in index_set:
        np_index_array.append(np.nonzero(indices == index)[0][0]) # should find way to parallelize with nonzero
    return np.array(np_index_array)


class optical_element:
    '''
    class for a wide range of optical elements. 
    specific elements are implemented as subclasses. 
    
    title : string
        title printed as the first line of the .dat file
    output : integer
        integer flag that controls the field data outptuted.
        0 outputs data only along the z axis.
        1 outputs data at all FE mesh points.
        2 outputs data at all mesh points and inside magnetic circuits.
    axsym : integer
        boolean flag denoting the symmetry of the element through the x-y  
        plane (between positive and negative z), with 1 true and 0 false.
    r_indices : integer ndarray
        length N 1D array of the radial indices of the FE mesh. begins at 1.
        (denoted I in MEBS manual.)
        skipped values are interpolated.
    z_indices : integer ndarray
        length M 1D array of the axial indices of the FE mesh. begins at 1.
        (denoted J in MEBS manual.)
        skipped values are interpolated.
    r : float ndarray
        NxM 2D array of radial coordinates defined on the grid of all 
        (r_indices,z_indices) points.
    z : float ndarray
        NxM 2D array of axial coordinates defined on the grid of all
        (r_indices,z_indices) points.
    colwidth : int
        width of columns to use in the file
    '''
    # output = 2
    # axsym = 1
    # title = 'optical element'
    # r_indices = np.array([])
    # z_indices = np.array([])
    # r = np.array([[]])
    # z = np.array([[]])
    
    ## infile = []
    colwidth = 10 # width of columns in written .dat files
    sp = " "*colwidth 
    int_fmt = Template("{:${colwidth}d}").substitute(colwidth=colwidth)
    float_fmt = Template("{:${colwidth}.${precision}g}")
    

    def __init__(self,filename='',verbose=False):
        self.infile = []
        self.initialize_lists()
        self.verbose = verbose
        if(filename):
            self.read(filename)
            
    def initialize_lists(self):
        pass
    
    def read(self,filename):
        f = open(filename,'r')
        self.infile = list(f) # creates a list with one entry per line of f
        self.read_intro()
        print(f"Reading file {filename} \nwith title: {self.title}")
        line_num = self.read_mesh()
        # self.plot_mesh_coarse() if self.verbose else 0
        self.read_other_blocks(line_num)

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
    
    def write(self,outfilename,title='optical element',coord_precision=6,curr_precision=6,field_precision=6,rel_perm_precision=6):
        self.coord_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=coord_precision)
        self.curr_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=curr_precision)
        self.field_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=field_precision)
        self.rel_perm_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=rel_perm_precision)
        
        f = open(outfilename,'w')
        
        self.write_intro(f,title)
        self.write_coordinates(f,self.z)
        self.write_coordinates(f,self.r)
        self.write_other_blocks(f)
        f.close()
        f = None
        
    def write_intro(self,f,title):
        f.write(title+"\n")
        f.write(self.check_len(self.int_fmt.format(self.output)))
        f.write(self.check_len(self.int_fmt.format(self.axsym)))
        f.write("\n\n") # one blank line after intro

    def write_coordinates(self,f,coords):
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
                f.write(self.check_len(self.coord_fmt.format(coords[i,j])))
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
        
    def plot_mesh_coarse(self,quads_on=False,adj=6):
        # plt.scatter(self.z.flatten(),self.r.flatten(),)
        for n in range(self.z.shape[0]):
            # add r index label
            plt.text(self.z[n,:].max()+adj,self.r[n,np.argmax(self.z[n,:])],self.r_indices[n])
            # plot this iso-r-index line
            plt.plot(self.z[n,:],self.r[n,:],color='m')
        for n in range(self.z.shape[1]):
            # add z index label
            plt.text(self.z[np.argmax(self.r[:,n]),n],self.r[:,n].max()+adj,self.z_indices[n])
            # plot this iso-z-index line
            plt.plot(self.z[:,n],self.r[:,n],color='m')
        self.plot_quads() if quads_on else 0
        plt.xlabel("z (mm)")
        plt.ylabel("r (mm)")
        # plt.title("Mesh (coarse)")
        plt.gca().set_aspect('equal')
        plt.show()
        
    def plot_quads(self):
        pass
    
    def plot_quad(self,z_indices,r_indices,color='k'):
        np_z_indices = np.nonzero((self.z_indices >= z_indices.min())*(self.z_indices <= z_indices.max()))[0]
        np_r_indices = np.nonzero((self.r_indices >= r_indices.min())*(self.r_indices <= r_indices.max()))[0]
        seg_np_indices = []
        seg_np_indices.append((np.repeat(np_r_indices.min(),len(np_z_indices)),np_z_indices))
        seg_np_indices.append((np_r_indices,np.repeat(np_z_indices.max(),len(np_r_indices))))
        seg_np_indices.append((np.repeat(np_r_indices.max(),len(np_z_indices)),np_z_indices))
        seg_np_indices.append((np_r_indices,np.repeat(np_z_indices.min(),len(np_r_indices))))
        for seg in seg_np_indices:
            plt.plot(self.z[seg],self.r[seg],color=color)
    
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



class strong_mag_lens(optical_element):
    '''
    class for reading and writing MEBS magnetic lens files that use hysteresis
    curves (H-B curves, in fact here without any hysteresis) for magnetic 
    materials and explicitly included coils to model magnetic lenses.
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
    
    def plot_hyst(self):
        plt.plot(self.H_arrays[-1],self.B_arrays[-1])
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


class weak_mag_lens(strong_mag_lens):
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


class weak_mag_lens_pp_region(weak_mag_lens):
    '''
    class for reading and writing MEBS magnetic lens files that only consider 
    the polepiece region and use a scalar magnetic potential to solve for
    the field.
    
    coil currents are specified only at boundaries.
    '''
    
    ##  to be implemented; the documentation is wildly unclear on this
    ##  general form is a list of
    ##  <index> <current value>
    ##  from "OPTICS_GUI_2.4" p. 42:
    # The first section specifies the potentials on the left-hand boundary;
    # The second section specifies the potentials on the upper boundary;
    # The third section specifies the potentials on the right-hand boundary.
    #
    ## from symmetric example, p. 36: 
    # 1 -250.    # either the lowest r or z index
    # 20 -250.   # 20 is clearly the highest r index
    # 1 -250.    # either the lowest r or z index
    # 27 0.      # 27 is clearly the highest z index
    # 1 0.       # either the lowest r or z index
    # 20 0.      # 20 is clearly the highest r index
    ## no third section because the right-hand boundary is a mirror plane (?)
    ## from this example, one might guess that a "section" is three indices:
    ## say, r_low, then r_high, then z_boundary or z_low, z_high, r_boundary
    ## the asymmetric example does not follow that pattern
    ## from asymmetric example, also p. 36:
    # 1 -250.    # either the lowest r or z index
    # 21 -250.   # 21 is clearly the highest r index
    # 1 -250.    # either the lowest r or z index
    # 29 250.    # 29 is the z index where the pp mag. mat. begins
    # 40 250.    # 40 is clearly the highest z index
    # 1 250.     # either the lowest r or z index
    # 21 250.    # 21 is clearly the highest r index
    # 

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
        # when properly implemented, these lines would divide "sections" into
        # separate list entries
        self.boundary_coil_indices.append(lines[:,0].astype(int))
        self.boundary_coil_currents.append(lines[:,1])
        return line_num+1 # start of next block
    
    def write_coil(self,f):
        M = len(self.boundary_coil_indices)
        for m in range(M):
            for k in range(len(self.boundary_coil_indices[m])):
                f.write(self.check_len(self.int_fmt.format(self.boundary_coil_indices[m][k])))
                f.write(self.check_len(self.curr_fmt.format(self.boundary_coil_currents[m][k])))
                f.write("\n")
        f.write("\n")

    def plot_quads(self):
        self.plot_mag_mat()


# example_strong = strong_mag_lens("/home/trh/MEBS/OPTICS/dat/OPTICS/Elements/MAG/LENS/mlenss1.dat",verbose=True)
# example_strong.write("/home/trh/data/test_strong.dat","a test")
# 
# example_weak = weak_mag_lens("/home/trh/MEBS/OPTICS/dat/OPTICS/Elements/MAG/LENS/mlensc1.dat",verbose=True)
# example_weak.write("/home/trh/data/test_weak.dat","a test")
# 
# example_pp = weak_mag_lens_pp_region("/home/trh/MEBS/OPTICS/dat/OPTICS/Elements/MAG/LENS/mlensp1.dat",verbose=True)
# example_pp.write("/home/trh/data/test_weak_pp.dat","a test")
# 
