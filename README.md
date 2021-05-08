# EMDesign
EMDesign is a python interface for Munro's Electron Beam Software (mebs.co.uk) and an automation engine that employs that interface for nearly hands-free design of electron optiocs.

Installation:
1. MEBS is commercial software and must be purchased and installed in order for EMDesign to work. Contact MEBS for more info.
2. For command-line operation of MEBS binaries, they must be added to the Windows file path. As EMDesign currently calls OPTICS, SOFEM, HERM1, and MIRROR, the following must be added to the Windows path:
```
< ... >\MEBS\OPTICS\bin\
< ... >\MEBS\SOFEM\bin\CL\
< ... >\MEBS\MIRROR\bin\MIRROR\
< ... >\MEBS\HERM1\bin\CL\
```
(To edit the Windows path, right click on This PC and then select Properties. On the left, click Avdanced System Settings, and Environment Variables at the bottom. Choose PATH in the User variables list and then click Edit.)

3. Install numpy, scipy, matplotlib and Shapely, and Jupyter Lab:
```
pip install numpy scipy matplotlib Shapely jupyterlab
```
4. Clone this repository, and it should immediately work if MEBS does.

Usage:
`optical_element_io` and `column_io` contain functions for reading, writing and manipulating MEBS optical element and column files, respectively. `optical_element_io` also calls the appropriate MEBS binary to calculate fields for the optical element of choice, and `column_io` calls `soray.exe` to calculate rays. `calc_optical_properties` calls the appropriate MEBS binary for optical properties calculations. `automation` contains all top-level automation functions, and `automation_library` contains low-level automation functions. `automation_archive` and `optical_element_archive` contain outdated functions, but could be inspiration for future code. There are several example notebooks at the moment for quick usage of a number of these functions, and more will be added over time.

`lensmin`, `mirmin` and `retracing` accept input files to operate automation functions for lens optimization, mirror optimization, and ray-retrace optimizization, respectively. 

Code has been tested in a small range of use cases so far and is bound to break for MEBS output that looks very different. Please notify us if that happens.
