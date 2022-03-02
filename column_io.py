import os, subprocess, shutil
from subprocess import TimeoutExpired
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from scipy.interpolate import interp1d
from optical_element_io import parse_potentials_string, ElecLens, StrongMagLens, MirPotentials
from misc_library import Logger, cd, check_len, MEBSError

class OpticalColumn:
    '''
    Class for column-level functions and variables.

    Attributes:
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
        V : list
            list of voltage applied to each electrode after autofocus 
            (set by read_mir_optical...).

    User methods:
        use_opt_img_cond_file
        use_mir_img_cond_file
        write_raytrace_file
        write_opt_img_cond_file
        write_mir_img_cond_file
        read_optical_properties
        read_mir_optical_properties
        calc_rays
        plot_rays

    Hard-coded attributes:
        colwidth : int
            width of columns to use in the optical element file.
            default 12.
        imgcondcolwidth : int
            width of columns to use in optical properties file.
            default 40.
        timeout : float
            seconds to wait before killing MEBS programs.
            default 3 minutes.
    '''

    colwidth = 12 # width of columns in written .dat files
    int_fmt = Template("{:${colwidth}d}").substitute(colwidth=colwidth)
    float_fmt = Template("{:${colwidth}.${precision}g}")
    imgcondcolwidth = 40
    imgcondprop_fmt = Template("{:<${imgcondcolwidth}s}").substitute(imgcondcolwidth=imgcondcolwidth)
    imgcondsubprop_fmt = Template("  {:<${imgcondcolwidth}s}").substitute(imgcondcolwidth=imgcondcolwidth-2)
    imgcondtext_fmt = Template("{:>${imgcondcolwidth}s}").substitute(imgcondcolwidth=imgcondcolwidth)
    imgcondint_fmt = Template("{:>${imgcondcolwidth}d}").substitute(imgcondcolwidth=imgcondcolwidth)
    rfloat_fmt = Template("{:>${imgcondcolwidth}.${precision}g}")
    timeout = 3*60 # 3 minutes

    source_parameters = dict.fromkeys(['position','shape','size','intensity distribution','angular shape',
                         'half angle','angular distribution','beam energy','energy width parameter',
                         'energy distribution','beam current'])
    oe_parameters = dict.fromkeys(['file','type','position','size','potentials'])
    imgplane_parameters = dict.fromkeys(['position'])
    screen_parameters = dict.fromkeys(['position'])
    particle_parameters = dict.fromkeys(['type','charge','mass'])
    simulation_parameters = dict.fromkeys(['initial conditions','particles/bunch','bunches','seed',
                             'error bound per step','paraxial ray method','order','coulomb interactions',
                             'simulation method','tree code parameter','focus mode','interactive',
                             'save trajectories'])
    mirror_object_parameters = dict.fromkeys(['position (mm)','beam energy (ev)','alpha (mrad)','alpha values','azimuth values',
                                'x size (mm)','y size (mm)'])
    mirror_screen_parameters = dict.fromkeys(['position (mm)'])
    turning_points_parameters = dict.fromkeys(['position'])

    def __init__(self, oe=None, obj=None,mir=None,tl_list=None, oe_list=None, filename=None):
        '''
        Parameters:
            oe : OpticalElement object
                passed to self.oe for usage of oe methods and attributes.
        '''

        self.Mlog = Logger('MEBS')
        self.olog = Logger('output')
        self.ilog = Logger('internal')
        if(filename):
            self.read(filename)
        if(oe == None and oe_list == None):
            self.obj = obj
            self.mir = mir
            self.tl_list = tl_list
            self.oe_list = []
            if(obj):
                self.oe_list.append(obj)
            if(tl_list):
                self.oe_list.extend(tl_list)
                self.oe = tl_list[0]
            if(mir):
                self.oe_list.append(mir)
            try:
                self.dirname = self.oe.dirname
            except:
                pass
            self.single = False
        elif(oe_list):
            self.oe_list = oe_list
            self.oe = tl_list[0]
            self.dirname = self.oe.dirname
            self.single = False
        else:
            self.dirname = oe.dirname
            self.oe_list = [oe] 
            self.oe = oe 
            self.single = True
        self.img_source_offset = 0.0001

    def calc_rays(self,i=0,max_attempts=1):
        '''
        Run after write_raytrace_file() to calculate rays.

        No parameters.
        '''
        with cd(self.dirname):
            try:
              self.Mlog.log.debug(subprocess.run(["soray.exe", self.raytracebasename_noext], stdout=subprocess.PIPE, 
                                   timeout=self.timeout).stdout.decode('utf-8'))
            except TimeoutExpired:
                i+=1
                if(i >= max_attempts):
                    self.olog.log.critical('Maximum attempts reached.')
                    raise MEBSError
                else:
                    self.olog.log.info('Ray tracing timed out. Rerunnning.')
                    self.calc_rays(i)

    def load_rays(self):
        step, z, r, x, y = np.loadtxt(os.path.join(self.dirname,self.raytracebasename_noext+'.raf'), 
                                      skiprows=8, unpack=True)
        cyl_symm = True if (y == 0).all() else False
        split_indices = np.squeeze(np.argwhere(step[:-1] > step[1:]), axis=1)+1
        steps = np.split(step, split_indices)
        zs = np.split(z, split_indices)
        rs = np.split(r, split_indices)
        n_rays = len(zs)
        if(cyl_symm):
            xs, ys = (None,)*n_rays,(None,)*n_rays
            r_ref = max([r[0] for r in rs])
        else:
            xs = np.split(x, split_indices)
            ys = np.split(y, split_indices)
            r_ref = max([np.sqrt(x[0]**2+y[0]**2) for x,y in zip(xs,ys)])
        return zs, rs, xs, ys, n_rays, r_ref, cyl_symm

    def evaluate_retracing(self):
        '''
        Determines distance to retracing condition.
        '''
        zs, rs, xs, ys, n_rays, r_ref, cyl_symm = self.load_rays()
        # iterate over rays
        self.ray_dev = np.zeros(n_rays,dtype=float)
        for ray_i in range(n_rays):
            self.ray_dev[ray_i] = self.retracing_dev_for_single_ray(zs[ray_i], rs[ray_i], r_ref, xs[ray_i], ys[ray_i])
        return self.ray_dev.sum()

    def retracing_dev_for_single_ray(self, z, r, r_ref, x=None, y=None):
       turnaround_index = np.argmin(z)+1
       if(turnaround_index == len(z)):
           return 100 # ray blocked, or something
       z_out = z[:turnaround_index]
       z_back = z[turnaround_index:]
       r_out = r[:turnaround_index]
       r_back = r[turnaround_index:]
       if(x and y):
           x_out = x[:turnaround_index]
           x_back = x[turnaround_index:]
           y_out = y[:turnaround_index]
           y_back = y[turnaround_index:]
           raise NotImplementedError
       else: # cyl_symm = True
           if(np.min(z_back) <= np.min(z_out) and np.max(z_back) >= np.max(z_out)):
               # interpolate coordinates, as z_out steps
               # may not perfectly match z_back steps
               r_func = interp1d(z_back, r_back)
               # normalize difference in paths by number of steps
               # and maximum r value reached by any ray
               return (np.abs(((r_func(z_out)-r_out)/len(r_out))/r_ref)**2).sum()
           # this case shouldn't happen, but just in case
           elif(np.min(z_out) <= np.min(z_back) and np.max(z_out) >= np.max(z_back)):
               # need to interpolate longer trajectory
               r_func = interp1d(z_out, r_out)
               return (np.abs(((r_func(z_back)-r_back)/len(r_back))/r_ref)**2).sum()
           else: # mixed
               r_func = interp1d(z_back, r_back)
               # need to clearly define validity region
               validity = (z_out > np.min(z_back))*(z_out < np.max(z_back))
               return (np.abs(((r_func(z_out[validity])-r_out[validity])/len(r_out[validity]))/r_ref)**2).sum()

    def plot_rays(self, width=15, height=5, mirror=False, xlim=None, ylim=None, coarse_mesh=True, 
                  boundary_mesh=False, only_radial=True, equal_aspect=True, savefile=''):
        '''
        Run after calc_rays() to plot rays.

        Optional parameters:
            cyl_symm : bool
                Determines whether separate x and y values are plotted, or
                just r.
            width : float
                Plot width passed to plt.figure(figsize=(width,height)).
                default 15.
            height : float
                Plot width passed to plt.figure(figsize=(width,height)).
                default 5.
            mirror : bool
                If True, mirrors rays across r=0.
                Default False.
            xlim : list or None
                List of [xmin,xmax] or None for default limits.
                Default None.
            ylim : list or None
                list of [ymin,ymax] or None for default limits.
                Default None.
            coarse_mesh : bool
                If True, add coarse mesh to plot. Default True.
            boundary_mesh : bool
                If True, add boundary of mesh to plot. Default False.
            only_radial : bool
                If True, plot only radial component even if x,y ray
                information exists.
                Default True.
            equal_aspect : bool
                If True, sets aspect ratio to 'equal'. 
                Default True.
            savefile : string
                Full filepath to save to.
                Default '' for no saving.
        '''
        zs, rs, xs, ys, n_rays, _, cyl_symm = self.load_rays()

        plt.figure(figsize=(width, height))

        for oe in self.oe_list:
            if(coarse_mesh):
                oe.add_coarse_mesh_to_plot() 
            if(boundary_mesh):
                oe.add_boundary_mesh_to_plot()
            oe.add_quads_to_plot() 
            
        colors = ['b','g','c']
        for i in range(n_rays):
            if(cyl_symm or only_radial):
                if(mirror):
                    plt.plot(zs[i], -rs[i], color=colors[i%3])
                plt.plot(zs[i], rs[i], color=colors[i%3])
            else:
                plt.plot(zs[i], xs[i], color=colors[i%3], label='x component of ray')
                plt.plot(zs[i], ys[i], color=colors[i%3], linestyle=':', label='y component of ray')
        plt.xlabel('z (mm)')
        if(cyl_symm or only_radial):
            plt.ylabel('r (mm)') 
        else:
            plt.ylabel('x and y (mm)')
            plt.legend()
        plt.title('Rays')
        if(xlim is not None):
            plt.xlim(xlim)
        if(ylim is not None):
            plt.ylim(ylim)
        if(equal_aspect):
            plt.gca().set_aspect('equal')
        if(savefile):
            plt.savefig(savefile,dpi=600,bbox_inches='tight')
        plt.show()

    def raytrace_from_saved_values(self):
        self.write_raytrace_file(
            self.mircondfilename, source_pos=self.source_pos, source_size=self.source_size, semiangle=self.semiangle, 
            energy=self.energy, initial_direction=self.initial_direction, lens_type=self.lens_type, 
            lens_pos=self.lens_pos, lens_excitation=self.lens_excitation, excitation_flag=self.excitation_flag, 
            potentials=self.potentials, screen_pos=self.screen_pos, minimum_rays=True)

        self.calc_rays()

    def write_raytrace_file(self, mircondfilename, source_pos=90, source_size=200, semiangle=10, energy=200000, 
                            initial_direction=180, lens_type='Electrostatic', lens_pos=0, lens_excitation=None, 
                            excitation_flag=None, potentials=None, screen_pos=95, relativity=False, cyl_symm=True, 
                            r_samples=3, alpha_samples=3, minimum_rays=False, precision=10, n_equipotentials=50):
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
                Default 200.
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
                Type of lens: 'Electrostatic' or 'Magnetic'. 
                Unused for multi-element columns.
                Default 'Electrostatic'.
            lens_pos : float
                Lens z position (mm). 
                Unused for multi-element columns.
                Default 10. **
            lens_excitation : float
                Specifies excitation strength of magnetic round lens or
                magnetic or electric multipole. Units are A-turns for magnetic 
                or volts for electric. 
                Use None if unused or for multi-element columns.
                Default None.
            excitation_flag : string
                Flag associated with lens_excitation.
                Flag options are f for fixed, vn for variable, where
                n is an integer grouping lenses varied together (e.g. v1)
                during autofocusing, or d for dynamic. The purpose of the 
                dynamic option is unclear. 
                Use None if unused or for multi-element columns.
                Default None.
            potentials : MirPotentials instance
                The MirPotentials class is used to sensibly store and format the 
                string used for specifying the potentials of several electrodes.
                Use None if unused or for multi-element columns.
                Default None. **
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
            minimum_rays : bool
                If True, overrides alpha_samples and r_samples and produces 
                exactly two rays.
                default False.
            precision : int
                Sets precision with which to print ray-related floating point
                numbers. 
                default 8 here to allow tiny source_pos - img_pos difference.
            n_equipotentials : int
                Number of equally-spaced equipotentials to plot.
                Default 50.
        '''
        if(minimum_rays == False):
            # SORAY uses mm
            x_positions = np.linspace(-source_size/2/1000,source_size/2/1000,r_samples,endpoint=True)
            if(cyl_symm):
                y_positions = np.array([0])
            else:
                y_positions = np.linspace(-source_size/2/1000,source_size/2/1000,r_samples,endpoint=True)

            # SORAY uses degrees
            angles = initial_direction + np.linspace(-semiangle*180/np.pi/1000,semiangle*180/np.pi/1000,alpha_samples,
                                                     endpoint=True)

        self.mircondfilename = mircondfilename
        self.mircondbasename_noext = os.path.splitext(os.path.basename(mircondfilename))[0] 
        self.raytracefile = os.path.join(self.dirname, self.mircondbasename_noext+'_rays'+'.dat')
        self.raytracebasename_noext = os.path.splitext(os.path.basename(self.raytracefile))[0] 
        self.mircondfloat_fmt = self.rfloat_fmt.substitute(imgcondcolwidth=self.imgcondcolwidth, precision=precision)
        cf = open(self.raytracefile,'w') 
        cf.write(f'Title raytrace file for {mircondfilename}\n\n')
        if(self.single):
            cf.write(f'\n{lens_type} lens\n')
            cf.write(self.imgcondsubprop_fmt.format("Filename")+
                     self.imgcondtext_fmt.format(self.oe.basename_noext)+"\n")
            cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(lens_pos)+"\n")
            if(potentials):
                cf.write(self.imgcondsubprop_fmt.format("Potentials")+
                         self.imgcondtext_fmt.format(potentials.format_noflag())+"\n")
            elif(lens_excitation is not None):
                cf.write(self.imgcondsubprop_fmt.format("Excitation")+
                         self.mircondfloat_fmt.format(lens_excitation)+"\n")
            else:
                raise ValueError('No potentials or lens excitation defined!')

        else:
            for oe in self.oe_list:
                cf.write(f'\n{oe.lens_type} lens\n')
                cf.write(self.imgcondsubprop_fmt.format("Filename")+
                         self.imgcondtext_fmt.format(oe.basename_noext)+"\n")
                cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(oe.lens_pos)+"\n")
                if(oe.potentials):
                    cf.write(self.imgcondsubprop_fmt.format("Potentials")+
                             self.imgcondtext_fmt.format(oe.potentials.format_noflag())+"\n")
                else:
                    cf.write(self.imgcondsubprop_fmt.format("Excitation")+
                             self.imgcondtext_fmt.format(oe.lens_excitation)+"\n")

        cf.write('\n')
        cf.write(self.imgcondsubprop_fmt.format("Time step factor")+self.mircondfloat_fmt.format(0.1)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Screen plane")+self.mircondfloat_fmt.format(screen_pos)+"\n")
        relativity_str = 'on' if relativity else 'off'
        cf.write(self.imgcondsubprop_fmt.format("Relativity")+self.imgcondtext_fmt.format(relativity_str)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Save rays")+self.imgcondtext_fmt.format('on')+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Save xyz")+self.imgcondtext_fmt.format('off')+"\n")
        cf.write('\nInitial ray conditions\n')
        rayfloat_fmt = self.float_fmt.substitute(colwidth=self.colwidth, precision=precision)
        if(minimum_rays):
            for x,y,alpha in zip([source_size/2/1000,0],[0,0],
                                 [initial_direction,initial_direction+semiangle*180/np.pi/1000]):
                cf.write(check_len(rayfloat_fmt.format(x), self.colwidth)+
                         check_len(rayfloat_fmt.format(y), self.colwidth)+
                         check_len(rayfloat_fmt.format(source_pos), self.colwidth)+
                         check_len(rayfloat_fmt.format(energy), self.colwidth)+
                         check_len(rayfloat_fmt.format(alpha), self.colwidth)+
                         check_len(rayfloat_fmt.format(0), self.colwidth)+ # azimuthal angle
                         '\n')
        else:
            for x in x_positions:
                for y in y_positions:
                    for alpha in angles:
                        cf.write(check_len(rayfloat_fmt.format(x), self.colwidth)+
                                 check_len(rayfloat_fmt.format(y), self.colwidth)+
                                 check_len(rayfloat_fmt.format(source_pos), self.colwidth)+
                                 check_len(rayfloat_fmt.format(energy), self.colwidth)+
                                 check_len(rayfloat_fmt.format(alpha), self.colwidth)+
                                 check_len(rayfloat_fmt.format(0), self.colwidth)+ # azimuthal angle
                                 '\n')

        if(potentials is not None):
            pot_min = min(potentials.voltages)
            pot_max = max(potentials.voltages)
            pot_range = np.linspace(pot_min, pot_max, n_equipotentials, endpoint=True)
            cf.write('\nElectrostatic Equipotentials\n')
            for pot in pot_range:
                cf.write(rayfloat_fmt.format(pot)+'\n')
        elif(hasattr(self,'mir') and hasattr(self.mir,'potentials')):
            pot_min = min(self.mir.potentials.voltages)
            pot_max = max(self.mir.potentials.voltages)
            pot_range = np.linspace(pot_min, pot_max, n_equipotentials, endpoint=True)
            cf.write('\nElectrostatic Equipotentials\n')
            for pot in pot_range:
                cf.write(rayfloat_fmt.format(pot)+'\n')

        cf.close()
        cf = None


    def use_mir_img_cond_file(self, mircondfilename):
        '''
        Uses existing optical imaging conditions file for MIRROR.
        
        Settings are not saved in memory, so raytrace_from_saved_values() 
        cannot be run after running this function.

        Parameters:
            mircondfilename : path
                full filename for imaging conditions file.
        '''
        self.program = 'mirror'
        for oe in self.oe_list:
            oe.program = 'mirror'
        self.mircondfilename = mircondfilename
        self.mircondbasename_noext = os.path.splitext(os.path.basename(mircondfilename))[0] 

    def use_opt_img_cond_file(self, imgcondfilename):
        '''
        Uses existing optical imaging conditions file for OPTICS.

        Not tested yet.
        
        Parameters:
            mircondfilename : path
                full filename for imaging conditions file.
        '''
        self.program = 'optics'
        for oe in self.oe_list:
            oe.program = 'optics'
        self.imgcondfilename = imgcondfilename
        self.imgcondbasename_noext = os.path.splitext(os.path.basename(imgcondfilename))[0] 

    def read_mir_img_cond_file(self, mircondfilename, write_safe=True):
        self.program = 'mirror'
        self.mircondfilename = mircondfilename
        self.mircondbasename_noext = os.path.splitext(os.path.basename(mircondfilename))[0] 
        self.dirname = os.path.dirname(mircondfilename)
        f = open(mircondfilename,'r') 
        if(write_safe):
            # rename this to avoid overwriting anything in the future
            self.mircondbasename_noext += '_test'
            self.mircondfilename = os.path.splitext(mircondfilename)[0]+'_test.dat'
            shutil.copyfile(mircondfilename,os.path.join(self.dirname,self.mircondfilename))
        self.infile = list(f)

        self.mir_cond_title = self.infile[0].strip('\n')
        line_num_source = 2
        line_num = self.read_section(line_num_source,'source',self.source_parameters)
        oe_types = ['lens','dipole','quadrupole','hexapole',
                    'octopole','decapole','dodecapole']
        self.oe_list = []
        while(self.infile[line_num].strip().lower() in oe_types):
            line_num = self.read_oe(line_num)
            oe_fitname = self.oe_parameters['file']
            oe_basename_noext = os.path.splitext(os.path.basename(oe_fitname))[0]
            oe_filename = os.path.join(self.dirname,oe_basename_noext+'.dat')
            oe_lens_type = self.oe_parameters['type']
            if(oe_lens_type == 'electrostatic'):
                oe = ElecLens(oe_filename)
            elif(oe_lens_type == 'magnetic'):
                oe = StrongMagLens(oe_filename)
            else:
                raise ValueError(f'Lens type {oe.lens_type} not recognized.')
            oe.lens_type = oe_lens_type
            oe.lens_pos = float(self.oe_parameters['position'])
            oe.lens_scale = float(self.oe_parameters['size'])
            oe.potentials = MirPotentials(oe,*self.oe_parameters['potentials'])
            self.oe_list.append(oe)
            self.oe = oe
        if(len(self.oe_list) == 1):
            self.lens_type = oe.lens_type
            self.lens_pos = oe.lens_pos
            self.lens_scale = oe.lens_scale
            self.potentials = oe.potentials
            self.single = True

        line_num = self.read_section(line_num,'gaussian image plane',self.imgplane_parameters)
        line_num = self.read_section(line_num,'screen',self.screen_parameters)
        line_num = self.read_section(line_num,'particles',self.particle_parameters)
        line_num = self.read_section(line_num,'simulation parameters',self.simulation_parameters)
        line_num = self.read_section(line_num,'mirror object',self.mirror_object_parameters)
        line_num = self.read_section(line_num,'mirror screen',self.mirror_screen_parameters)
        line_num = self.read_section(line_num,'turning points guideline',self.turning_points_parameters)

        # for raytracing
        self.source_pos = float(self.source_parameters['position'])
        self.source_size = float(self.source_parameters['size'])
        self.semiangle = float(self.source_parameters['half angle'])
        self.energy = float(self.source_parameters['beam energy'])
        self.energy_negative = self.energy < 0
        self.energy = abs(self.energy)
        self.initial_direction=180 if self.energy_negative else 0
        self.screen_pos = float(self.screen_parameters['position'])
        self.lens_excitation = None
        self.excitation_flag = None
        # for generating new column files
        self.reverse_dir = True if self.energy_negative else False
        self.img_pos = float(self.imgplane_parameters['position']) 
        self.intensity_dist = self.source_parameters['intensity distribution']
        self.ang_shape = self.source_parameters['angular shape']
        self.ang_dist = self.source_parameters['angular distribution']
        self.energy_width = float(self.source_parameters['energy width parameter'])
        self.energy_dist = self.source_parameters['energy distribution']
        self.ray_method = self.simulation_parameters['paraxial ray method']
        self.order = int(self.simulation_parameters['order'])
        self.focus_mode = self.simulation_parameters['focus mode']
        self.interactive = self.simulation_parameters['interactive']
        self.obj_pos = float(self.mirror_object_parameters['position (mm)'])
        self.obj_semiangle = float(self.mirror_object_parameters['alpha (mrad)'])
        self.x_size = float(self.mirror_object_parameters['x size (mm)'])
        self.y_size = float(self.mirror_object_parameters['y size (mm)'])
        self.mir_screen_pos = float(self.mirror_screen_parameters['position (mm)'])
        self.turning_point = float(self.turning_points_parameters['position'])

    def read_section(self,line_num,section,parameters):
        if(self.infile[line_num].strip().lower() != section):
            raise ValueError(f'Unexpected string on line {line_num}: \n'
                             f'Expected {section}; got {self.infile[line_num].strip()}')
        line_num += 1
        while(line_num < len(self.infile) and self.infile[line_num].isspace() != True):
            line_split = self.infile[line_num].split()
            parameter_name = ' '.join(line_split[:-1]).lower() 
            if(parameter_name in parameters):
                parameters[parameter_name] = line_split[-1]
            else:
                raise ValueError(f'Parameter {parameter_name} not a recognized'
                                  ' parameter; possibly not implemented yet.')
            line_num += 1
        line_num += 1 # space after section
        return line_num

    def read_oe(self,line_num):
        if(self.infile[line_num].strip().lower() != 'lens'):
            raise NotImplementedError
        line_num += 1
        while(self.infile[line_num].isspace() != True):
            line_split = self.infile[line_num].split()
            parameter_name = ' '.join(line_split[:-1]).lower() 
            if(parameter_name in self.oe_parameters):
                self.oe_parameters[parameter_name] = line_split[-1]
            elif(line_split[0].lower() in self.oe_parameters):
                self.oe_parameters[line_split[0].lower()] = parse_potentials_string(line_split[1:])
            else:
                raise ValueError(f'Parameter {parameter_name} not a recognized'
                                  ' oe parameter; possibly not implemented yet.')
            line_num += 1
        line_num += 1 # space after source section
        return line_num
    

    def write_mir_img_cond_file(self, mircondfilename, source_pos=90, source_shape='ROUND', source_size=200, 
                                intensity_dist='UNIFORM', ang_shape='ROUND', semiangle=10, ang_dist='UNIFORM', 
                                energy=200000, energy_width=1, energy_dist='Gaussian', lens_type='electrostatic', 
                                lens_pos=0, lens_scale=1, lens_excitation=None, excitation_flag=None, 
                                potentials=None, ray_method="R", order=3, focus_mode="AUTO", img_pos=95, 
                                screen_pos=None, mir_screen_pos=None, save_trj=True, obj_pos=None, obj_semiangle=None, 
                                x_size=0.1, y_size=0.1, reverse_dir=True, turning_point=5, precision=10):
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
                Options are "MAGNETIC" and "ELECTROSTATIC". 
                Unused for multi-element columns.
                Default "ELECTROSTATIC".
            lens_pos : float
                Lens z position (mm). 
                Unused for multi-element columns.
                Default 10. **
            lens_scale : float
                Scale factor to be applied to spatial extent of lens. 
                Unused for multi-element columns.
                Default 1.
            lens_excitation : float
                Specifies excitation strength of magnetic round lens or
                magnetic or electric multipole. Units are A-turns for magnetic 
                or volts for electric. 
                Use None if unused or for multi-element columns.
                Default None.
            excitation_flag : string
                Flag associated with lens_excitation.
                Flag options are f for fixed, vn for variable, where
                n is an integer grouping lenses varied together (e.g. v1)
                during autofocusing, or d for dynamic. The purpose of the 
                dynamic option is unclear. 
                Use None if unused or for multi-element columns.
                Default None.
            potentials : MirPotentials instance
                The MirPotentials class is used to sensibly store and format the 
                string used for specifying the potentials of several electrodes.
                Use None if unused or for multi-element columns.
                Default None. **
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
        for oe in self.oe_list:
            oe.program = 'mirror'
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

        self.mircondfloat_fmt = self.rfloat_fmt.substitute(imgcondcolwidth=self.imgcondcolwidth, precision=precision)
        self.mircondfilename = mircondfilename
        self.mircondbasename_noext = os.path.splitext(os.path.basename(mircondfilename))[0] 
        cf = open(self.mircondfilename,'w') 
        self.mir_cond_title = 'Imaging Conditions for MIRROR'

        # save settings for later raytraces
        self.source_pos=source_pos
        self.source_size=source_size
        self.semiangle=semiangle
        self.energy=np.abs(energy)
        self.initial_direction=180*reverse_dir
        self.lens_type=lens_type
        self.lens_pos=lens_pos
        self.lens_excitation=lens_excitation
        self.excitation_flag = excitation_flag
        self.potentials=potentials
        self.screen_pos=screen_pos
        
        cf.write(f"Title     {self.mir_cond_title:>70}\n\n")
        cf.write("SOURCE\n")
        cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(source_pos)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Shape")+self.imgcondtext_fmt.format(source_shape)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Size")+self.mircondfloat_fmt.format(source_size)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Intensity distribution")+
                 self.imgcondtext_fmt.format(intensity_dist)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Angular shape")+self.imgcondtext_fmt.format(ang_shape)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Half angle")+self.mircondfloat_fmt.format(semiangle)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Angular distribution")+self.imgcondtext_fmt.format(ang_dist)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Beam energy")+self.mircondfloat_fmt.format(energy)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Energy width parameter")+
                 self.mircondfloat_fmt.format(energy_width)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Energy distribution")+self.imgcondtext_fmt.format(energy_dist)+"\n")
        cf.write(self.imgcondsubprop_fmt.format("Beam current")+self.mircondfloat_fmt.format(1)+"\n")
        if(self.single):
            cf.write("\nLENS\n")
            cf.write(self.imgcondsubprop_fmt.format("File")+self.imgcondtext_fmt.format(self.oe.fitname)+"\n")
            cf.write(self.imgcondsubprop_fmt.format("Type")+self.imgcondtext_fmt.format(lens_type)+"\n")
            cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(lens_pos)+"\n")
            cf.write(self.imgcondsubprop_fmt.format("Size")+self.mircondfloat_fmt.format(lens_scale)+"\n")
            if(potentials):
                cf.write(self.imgcondsubprop_fmt.format("Potentials")+
                         self.imgcondtext_fmt.format(potentials.format())+"\n")
            elif(lens_excitation is not None):
                cf.write(self.imgcondsubprop_fmt.format("Excitation")+
                         self.mircondfloat_fmt.format(lens_excitation)+excitation_flag+"\n")
            else:
                raise ValueError('No potentials or lens excitation defined!')
        else:
            raise NotImplementedError
            # these oe attributes may not be defined anywhere
            # only read_mir_img_cond_file defines them currently
            for oe in self.oe_list:
                cf.write("\nLENS\n")
                cf.write(self.imgcondsubprop_fmt.format("File")+self.imgcondtext_fmt.format(oe.fitname)+"\n")
                cf.write(self.imgcondsubprop_fmt.format("Type")+self.imgcondtext_fmt.format(oe.lens_type)+"\n")
                cf.write(self.imgcondsubprop_fmt.format("Position")+self.mircondfloat_fmt.format(oe.lens_pos)+"\n")
                cf.write(self.imgcondsubprop_fmt.format("Size")+self.mircondfloat_fmt.format(oe.lens_scale)+"\n")
                if(oe.potentials):
                    cf.write(self.imgcondsubprop_fmt.format("Potentials")+
                             self.imgcondtext_fmt.format(oe.potentials.format())+"\n")
                else:
                    cf.write(self.imgcondsubprop_fmt.format("Excitation")+
                             self.mircondfloat_fmt.format(oe.lens_excitation)+oe.excitation_flag+"\n")
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

    def write_opt_img_cond_file(self, imgcondfilename, n_intervals=200, energy=200000, energy_width=1, 
                                aperture_angle=30, obj_pos=0, img_pos=6, n_intermediate_images=0, lens_pos=0, 
                                lens_strength=1, lens_scale=1, precision=6, auto_focus=1):
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
        for oe in self.oe_list:
            oe.program = 'optics'
        self.imgcondfloat_fmt = self.rfloat_fmt.substitute(imgcondcolwidth=self.imgcondcolwidth, precision=precision)
        self.lensfloat_fmt = self.float_fmt.substitute(colwidth=self.colwidth, precision=precision)
        self.imgcondfilename = imgcondfilename
        self.imgcondbasename_noext = os.path.splitext(os.path.basename(imgcondfilename))[0] 
        cf = open(self.imgcondfilename,'w') 
        self.img_cond_title = 'Imaging Conditions for OPTICS+ABER5'
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
        if(self.single and oe.lens_type == 'magnetic'):
            cf.write('Magnetic Lens\n')
            cf.write('\n')
            cf.write(self.lensfloat_fmt.format(lens_pos)+self.lensfloat_fmt.format(lens_strength))
            cf.write(self.lensfloat_fmt.format(lens_scale)+self.int_fmt.format(auto_focus)+
                     "{:>40s}".format(self.oe.potname)+'\n')
            cf.write('\n')
        elif(self.single and oe.lens_type == 'electrostatic'):
            raise NotImplementedError('Electric lenses not yet implemented here; use MIRROR functions instead.')
        elif(self.single == False):
            magnetic_label = False
            for oe in self.oe_list:
                # OPTICS formats multiple lenses as
                # Lens Type
                #    lens 1
                # 
                #    lens 2 ....
                # 
                # Other Lens Type
                if(oe.lens_type == 'magnetic'):
                    if(magnetic_label == False): 
                        cf.write('Magnetic Lens\n')
                        magnetic_label = True
                    cf.write('\n')
                    cf.write(self.lensfloat_fmt.format(oe.lens_pos)+self.lensfloat_fmt.format(oe.lens_strength))
                    cf.write(self.lensfloat_fmt.format(oe.lens_scale)+self.int_fmt.format(auto_focus)+
                             "{:>40s}".format(oe.potname)+'\n')
                    cf.write('\n')
            for oe in self.oe_list:
                if(oe.lens_type == 'electrostatic'):
                    raise NotImplementedError('Electric lenses not yet implemented here; use MIRROR functions instead.')

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
        try:
            pf = open(os.path.join(self.dirname,self.imgcondbasename_noext+'.res'),'r')
        except FileNotFoundError:
            self.olog.log.critical('No .res file found. This is likely because '
                    'the optical element and column files are in different directories.')
            self.olog.log.critical(f'{self.imgcondfilename=}')
            for oe in self.oe_list:
                self.olog.log.critical(f'{self.oe.filename=}')
            raise FileNotFoundError
        properties_lines = pf.readlines()
        # see end of this file for snippets of the .res file 
        # that are relevant to this parameter extraction
        n_lenses = len(self.oe_list)
        for i,line in enumerate(properties_lines):
            if 'FIRST-ORDER PROPERTIES' in line:
                linenum_mag = i+9+n_lenses
                linenum_rot = i+10+n_lenses
                if(n_lenses > 1):
                    linenum_curr = []
                    for j in range(n_lenses):
                        linenum_curr.append(i+5+j)
                else:
                    linenum_curr = i+5
            if 'Magnetic Lens      No.  1' in line: 
                linenum_f = i+8+n_lenses
                linenum_f_real = i+4+n_lenses
            if 'THIRD-ORDER ABERRATION COEFFICIENTS   (in S.I. units)' in line:
                linenum_c3 = i+9
            if ' ***** CHROMATIC AB *****' in line:
                linenum_cc = i+3
        self.mag = float(properties_lines[linenum_mag].split()[3])
        self.rot = float(properties_lines[linenum_rot].split()[5]) # deg
        if(n_lenses > 1):
            self.lens_curr = []
            for j,oe in enumerate(self.oe_list):
                self.lens_curr.append(float(properties_lines[linenum_curr[j]].split()[7]))
                oe.lens_curr = self.lens_curr[-1]
        else:
            self.lens_curr = float(properties_lines[linenum_curr].split()[7])
            self.oe.lens_curr = self.lens_curr
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
    def read_mir_optical_properties(self, raytrace=True):
        '''
        Run after calc_properties_mirror() to read in the computed optical 
        properties.

        Optional parameters:
            raytrace : bool
                Determines whether raytrace file with same parameters is 
                automatically written and calc_rays() is run.
        '''
        try:
            pf = open(os.path.join(self.dirname, self.mircondbasename_noext+'.res'),'r')
        except FileNotFoundError:
            self.olog.log.critical('No .res file found. This is likely because '
                    'the optical element and column files are in different directories.')
            self.olog.log.critical(f'{self.mircondfilename=}')
            for oe in self.oe_list:
                self.olog.log.critical(f'{self.oe.filename=}')
            raise FileNotFoundError
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
        self.V = []
        j = 0
        while('Potential V' in properties_lines[linenum_v+j]):
            self.V.append(float(properties_lines[linenum_v+j].split()[3]))
            j+=1
        if(hasattr(self,'potentials')):
            self.potentials.voltages = self.V # update saved voltages
        self.oe.V = self.V # save to optical element
        self.f = None
        self.f_real = None
        self.c3 = float(properties_lines[linenum_c3].split()[0]) # m to mm
        self.cc = float(properties_lines[linenum_cc].split()[0]) # m to mm
        pf.close()
        pf = None

        if(raytrace):
            self.raytrace_from_saved_values()



