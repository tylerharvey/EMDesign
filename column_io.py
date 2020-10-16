import sys,os,subprocess,shutil,datetime
from subprocess import TimeoutExpired
import numpy as np
import matplotlib.pyplot as plt
from string import Template
from contextlib import contextmanager
from optical_element_io import cd, check_len, check_len_multi

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
            default 10 minutes.
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
    timeout = 10*60 # 10 minutes

    def __init__(self,oe):
        self.dirname = oe.dirname
        self.oe_list = [oe] # will need to be changed to allow multiple optical elements
        self.oe = oe # will need to be changed to allow multiple optical elements

    # this will ultimately make more sense as an independent object
    # writing it here for now
    # class OpticalConfiguration:
    #     def __init__(self,mircondfilename,source_pos,source_size,semiangle,energy,initial_direction,lens_type,lens_pos,lens_excitation,potentials,screen_pos):
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

    def plot_rays(self,cyl_symm=True):
        '''
        Run after calc_rays() to plot rays.

        Optional parameters:
            cyl_symm : bool
                Determines whether separate x and y values are plotted, or
                just r.
        '''
        step,z,r,x,y = np.loadtxt(os.path.join(self.dirname,self.raytracebasename_noext+'.raf'),skiprows=8,unpack=True)
        split_indices = np.squeeze(np.argwhere(step[:-1] > step[1:]))+1
        steps = np.split(step,split_indices)
        zs = np.split(z,split_indices)
        rs = np.split(r,split_indices)
        xs = np.split(x,split_indices)
        ys = np.split(y,split_indices)

        for oe in self.oe_list:
            oe.add_mesh_to_plot() 
            oe.add_quads_to_plot() 
            
        colors = ['b','g','c']
        for i in range(len(steps)):
            if(cyl_symm):
                plt.plot(zs[i],rs[i])
            else:
                plt.plot(zs[i],xs[i],color=colors[i%3],label='x component of ray')
                plt.plot(zs[i],ys[i],color=colors[i%3],linestyle=':',label='y component of ray')
        plt.xlabel('z (mm)')
        if(cyl_symm):
            plt.ylabel('r (mm)') 
        else:
            plt.ylabel('x and y (mm)')
            plt.legend()
        plt.title('Rays')
        plt.gca().set_aspect('equal')
        plt.show()

    def raytrace_from_saved_values(self):
        self.write_raytrace_file(self.mircondfilename,source_pos=self.source_pos,source_size=self.source_size,semiangle=self.semiangle,energy=self.energy,initial_direction=self.initial_direction,lens_type=self.lens_type,lens_pos=self.lens_pos,lens_excitation=self.lens_excitation,potentials=self.potentials,screen_pos=self.screen_pos)
        self.calc_rays()

    def write_raytrace_file(self,mircondfilename,source_pos=90,source_size=200,semiangle=10,energy=200000,initial_direction=180,lens_type='Electrostatic',lens_pos=0,lens_excitation=None,potentials=None,screen_pos=95,relativity=False,cyl_symm=True,r_samples=3,alpha_samples=3,precision=6,n_equipotentials=50):
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
        cf.write(self.imgcondsubprop_fmt.format("Filename")+self.imgcondtext_fmt.format(self.oe.basename_noext)+"\n")
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
                    cf.write(check_len(rayfloat_fmt.format(x),self.colwidth)+
                             check_len(rayfloat_fmt.format(y),self.colwidth)+
                             check_len(rayfloat_fmt.format(source_pos),self.colwidth)+
                             check_len(rayfloat_fmt.format(energy),self.colwidth)+
                             check_len(rayfloat_fmt.format(alpha),self.colwidth)+
                             check_len(rayfloat_fmt.format(0),self.colwidth)+ # azimuthal angle
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
            raytrace : bool
                Determines whether raytrace file with same parameters is 
                automatically written and calc_rays() is run.
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

        self.mircondfloat_fmt = self.rfloat_fmt.substitute(imgcondcolwidth=self.imgcondcolwidth,precision=precision)
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
        self.potentials=potentials
        self.screen_pos=screen_pos
        
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
        cf.write(self.imgcondsubprop_fmt.format("File")+self.imgcondtext_fmt.format(self.oe.fitname)+"\n")
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
        for oe in self.oe_list:
            oe.program = 'optics'
        self.imgcondfloat_fmt = self.rfloat_fmt.substitute(imgcondcolwidth=self.imgcondcolwidth,precision=precision)
        self.lensfloat_fmt = self.float_fmt.substitute(colwidth=self.colwidth,precision=precision)
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
        cf.write('Magnetic Lens\n')
        cf.write('\n')
        cf.write(self.lensfloat_fmt.format(lens_pos)+self.lensfloat_fmt.format(lens_strength))
        cf.write(self.lensfloat_fmt.format(lens_scale)+self.int_fmt.format(auto_focus)+"{:>40s}".format(self.oe.potname)+'\n')
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
    def read_mir_optical_properties(self,raytrace=True):
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
        self.V = []
        j = 0
        while('Potential V' in properties_lines[linenum_v+j]):
            self.V.append(float(properties_lines[linenum_v+j].split()[3]))
            j+=1
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



