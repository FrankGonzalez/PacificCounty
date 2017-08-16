"""
Module to set up run time parameters for Clawpack.

The values set in the function setrun are then written out to data files
that will be read in by the Fortran code.

"""

import os
import sys
import numpy as np

try:
    CLAW = os.environ['CLAW']
except:
    raise Exception("*** Must first set CLAW environment variable")

ModelingDir = os.environ['ModelingDir']
sys.path.append(ModelingDir + '/FranksLib')
import FGgauges_module as gm
import interptools

# Project directory for storing topo and dtopo files:

try:
    PacificCounty = os.environ['PacificCounty']
except:
    raise Exception("Need to set PacificCounty environment variable")

project_dir = os.environ["PacificCounty"]

#==================================================================
#  Input
#==================================================================

# SET START AND END TIME FOR SIMULATION
t_start_min =  0
t_run_min   = 360  # Integer.  Duration of run in minutes
dt_minutes_output = 1  # Output interval in minutes

# RED REGION = COMPUTATIONAL DOMAIN
CompDom = (-127.5,-123.9,45.8,46.8)
x_num_cells = 36
y_num_cells = 10

AMR_ratios = (5,4,6,9)
AMR_levels_max = len(AMR_ratios)+1

# REGION 0 = AUTOMATIC.  COMPUTATIONAL DOMAIN + GHOST CELLS

# REGION 1 = SOURCE
dxy = 0.05
SourceName = 'L1_60s_linear_x1.11.tt3'
SourceExtent = (CompDom[0]-dxy,CompDom[1]+dxy,CompDom[2]-dxy,CompDom[3]+dxy)

# Topo files
TopoName1 = 'etopo1_-127.7_-123.7_45.6_47.0.asc'
xextra = 6./60.  # Etopo1 res is 1 min = 1/60 deg.  So this adds 3 cells.
yextra = 6./60.
TopoName1Extent = (CompDom[0]-xextra,CompDom[1]+xextra,CompDom[2]-yextra,CompDom[3]+yextra)  # etopo1
TopoName2 = "astoria_v3_M-124.1_-124.0_46.31_46.38.asc"
TopoName2Extent = (-124.1,-124.0,46.31,46.38) # Medium
TopoName3 = 'astoria_v3_S-124.07_-124.045_46.341_46.356.asc'
TopoName3Extent = (-124.07,-124.045,46.341,46.356) # Berm

# Fixed Grids

FGextent = {}
FGextent[1]   = (TopoName2Extent[0],TopoName2Extent[1], \
    TopoName2Extent[2],TopoName2Extent[3]) # Medium grid
FGextent[2]   = (TopoName3Extent[0],TopoName3Extent[1], \
    TopoName3Extent[2],TopoName3Extent[3]) # Small Grid = Berm

# REGION 5 = AUTOMATIC -- AROUND FIXED GRID

#------------------------------
def setrun(claw_pkg='geoclaw'):
#------------------------------

    """
    Define the parameters used for running Clawpack.

    INPUT:
        claw_pkg expected to be "geoclaw" for this setrun.

    OUTPUT:
        rundata - object of class ClawRunData

    """

    print '=================================================================='
    print '\nCompDom = ', CompDom
    print 'x_num_cells, y_num_cells =', x_num_cells, y_num_cells
    print 'AMR_ratios = ', AMR_ratios
    print 'AMR_levels_max = ', AMR_levels_max
    print 'SourceName = ', SourceName
    print 'SourceExtent = ', SourceExtent
    print ("TopoName1 = ", TopoName1)
    print ("TopoName1Extent = ", TopoName1Extent)
    print ("TopoName2 = ", TopoName2)
    print ("TopoName2Extent = ", TopoName2Extent)
    print ("TopoName3 = ", TopoName3)
    print ("TopoName3Extent = ", TopoName3Extent)
    print '\n*************************\n'
    print 't_start_min = ', t_start_min
    print 't_run_min   = ', t_run_min
    print '\n=================================================================='

    from clawpack.clawutil import data

    assert claw_pkg.lower() == 'geoclaw',  "Expected claw_pkg = 'geoclaw'"

    num_dim = 2
    rundata = data.ClawRunData(claw_pkg, num_dim)
    
    #------------------------------------------------------------------
    # Problem-specific parameters to be written to setprob.data:
    #------------------------------------------------------------------
    
    #probdata = rundata.new_UserData(name='probdata',fname='setprob.data')

    #------------------------------------------------------------------
    # GeoClaw specific parameters:
    #------------------------------------------------------------------
    rundata = setgeo(rundata)

    #------------------------------------------------------------------
    # Standard Clawpack parameters to be written to claw.data:
    #   (or to amr2ez.data for AMR)
    #------------------------------------------------------------------
    clawdata = rundata.clawdata  # initialized when rundata instantiated

    # Set single grid parameters first.
    # See below for AMR parameters.


    # ---------------
    # Spatial domain:
    # ---------------

    # Number of space dimensions:
    clawdata.num_dim = num_dim

    # Lower and upper edge of computational domain:
    # note that x1_domain,y1_domain,x2_domain,y2_domain are used in setting regions
    
    # Set "Computational Domain" (RED borders in GE) (Close approximation to PMEL grid A):

    clawdata.lower[0] = x1_domain = CompDom[0]     # SET west longitude boundary
    clawdata.upper[0] = x2_domain = CompDom[1]     # SET east longitude boundary

    clawdata.lower[1] = y1_domain = CompDom[2]      # SET south latitude boundary
    clawdata.upper[1] = y2_domain = CompDom[3]      # SET north latitude boundary
    
    print 'x1_domain,x2_domain =', x1_domain,x2_domain
    print 'y1_domain,y2_domain =', y1_domain,y2_domain
    

    ######################################################################

    # Number of grid cells: Coarsest grid
    clawdata.num_cells[0] = x_num_cells    # SET Longitude resolution
    clawdata.num_cells[1] = y_num_cells    # SET Latitude resolution
    AMR1_xresolution = (x2_domain-x1_domain)/clawdata.num_cells[0]
    AMR1_yresolution = (y2_domain-y1_domain)/clawdata.num_cells[1]
    
    print '\n ************************************************************************'
    print 'AMR1_xresolution =', AMR1_xresolution
    print 'AMR1_yresolution =', AMR1_yresolution
    
    # ---------------
    # Size of system:
    # ---------------

    # Number of equations in the system:
    clawdata.num_eqn = 3

    # Number of auxiliary variables in the aux array (initialized in setaux)
    clawdata.num_aux = 3

    # Index of aux array corresponding to capacity function, if there is one:
    clawdata.capa_index = 2
    
    # -------------
    # Initial time:
    # -------------

    clawdata.t0 = t_start_min*60.  # SET start time (seconds)

    # Restart from checkpoint file of a previous run?
    # Note: If restarting, you must also change the Makefile to set:
    #    RESTART = True
    # If restarting, t0 above should be from original run, and the
    # restart_file 'fort.chkNNNNN' specified below should be in 
    # the OUTDIR indicated in Makefile.

    clawdata.restart = False               # True to restart from prior results
    clawdata.restart_file = 'fort.chk00036'  # File to use for restart data

    # -------------
    # Output times:
    #--------------

    # Specify at what times the results should be written to fort.q files.
    # Note that the time integration stops after the final output time.
    # The solution at initial time t0 is always written in addition.

    clawdata.output_style = 2

    if clawdata.output_style==1:
        # Output nout frames at equally spaced times up to tfinal:
        
        t_minutes_run = int(t_run_min)    # MUST BE AN INTEGER
        clawdata.tfinal           = int(t_minutes_run*60)  # SET end time
        clawdata.output_t0        = True     # output at initial (or restart) time?
        clawdata.num_output_times = int((clawdata.tfinal-clawdata.t0)/(60.*dt_minutes_output)) # SET number of output frames

        print '\nclawdata.t0:', clawdata.t0
        print 'clawdata.output_style:',clawdata.output_style
        print 't_minutes_run:', t_minutes_run
        print 'dt_minutes_output:',dt_minutes_output
        print 'clawdata.tfinal:',clawdata.tfinal
        print 'clawdata.num_output_times:', clawdata.num_output_times

    elif clawdata.output_style == 2:
        # Specify a list of output times.
        clawdata.output_times = [0,5] + range(60,60*t_run_min+60,60)
        # clawdata.output_times = range(-120,300,60)
        print 'clawdata.output_style:',clawdata.output_style
        print 'Output times (sec): ', clawdata.output_times

    elif clawdata.output_style == 3:
        # Output every iout timesteps with a total of ntot time steps:
        clawdata.output_step_interval = 1
        clawdata.total_steps = 3
        clawdata.output_t0 = True
        
    clawdata.output_format = 'binary'      # 'ascii' or 'binary' 

    clawdata.output_q_components = 'all'   # need all
    clawdata.output_aux_components = 'none'  # eta=h+B is in q
    clawdata.output_aux_onlyonce = False    # output aux arrays each frame

    # ---------------------------------------------------
    # Verbosity of messages to screen during integration:
    # ---------------------------------------------------

    # The current t, dt, and cfl will be printed every time step
    # at AMR levels <= verbosity.  Set verbosity = 0 for no printing.
    #   (E.g. verbosity == 2 means print only on levels 1 and 2.)
    clawdata.verbosity = 2

    # --------------
    # Time stepping:
    # --------------

    # if dt_variable==1: variable time steps used based on cfl_desired,
    # if dt_variable==0: fixed time steps dt = dt_initial will always be used.
    clawdata.dt_variable = True

    # Initial time step for variable dt.
    # If dt_variable==0 then dt=dt_initial for all steps:
    clawdata.dt_initial = 1.0

    # Max time step to be allowed if variable dt used:
    clawdata.dt_max = 1e+99

    # Desired Courant number if variable dt used, and max to allow without
    # retaking step with a smaller dt:
    clawdata.cfl_desired = 0.75
    clawdata.cfl_max = 1.0

    # Maximum number of time steps to allow between output times:
    clawdata.steps_max = 5000

    # ------------------
    # Method to be used:
    # ------------------

    # Order of accuracy:  1 => Godunov,  2 => Lax-Wendroff plus limiters
    clawdata.order = 2
    
    # Use dimensional splitting? (not yet available for AMR)
    clawdata.dimensional_split = 'unsplit'
    
    # For unsplit method, transverse_waves can be 
    #  0 or 'none'      ==> donor cell (only normal solver used)
    #  1 or 'increment' ==> corner transport of waves
    #  2 or 'all'       ==> corner transport of 2nd order corrections too
    clawdata.transverse_waves = 2

    # Number of waves in the Riemann solution:
    clawdata.num_waves = 3
    
    # List of limiters to use for each wave family:  
    # Required:  len(limiter) == num_waves
    # Some options:
    #   0 or 'none'     ==> no limiter (Lax-Wendroff)
    #   1 or 'minmod'   ==> minmod
    #   2 or 'superbee' ==> superbee
    #   3 or 'mc'       ==> MC limiter
    #   4 or 'vanleer'  ==> van Leer
    clawdata.limiter = ['mc', 'mc', 'mc']

    clawdata.use_fwaves = True    # True ==> use f-wave version of algorithms
    
    # Source terms splitting:
    #   src_split == 0 or 'none'    ==> no source term (src routine never called)
    #   src_split == 1 or 'godunov' ==> Godunov (1st order) splitting used, 
    #   src_split == 2 or 'strang'  ==> Strang (2nd order) splitting used,  not recommended.
    clawdata.source_split = 'godunov'


    # --------------------
    # Boundary conditions:
    # --------------------

    # Number of ghost cells (usually 2)
    clawdata.num_ghost = 2

    # Choice of BCs at xlower and xupper:
    #   0 => user specified (must modify bcN.f to use this option)
    #   1 => extrapolation (non-reflecting outflow)
    #   2 => periodic (must specify this at both boundaries)
    #   3 => solid wall for systems where q(2) is normal velocity

    clawdata.bc_lower[0] = 'extrap'  # Longitude boundary
    clawdata.bc_upper[0] = 'extrap'  # Longitude boundary

    clawdata.bc_lower[1] = 'extrap'  # Latitude boundary
    clawdata.bc_upper[1] = 'extrap'  # Latitude boundary

    # --------------
    # Checkpointing:
    # --------------

    # Specify when checkpoint files should be created that can be
    # used to restart a computation.

    clawdata.checkpt_style = 0

    if clawdata.checkpt_style == 0:
        # Do not checkpoint at all
        pass

    elif clawdata.checkpt_style == 1:
        # Checkpoint only at tfinal.
        pass

    elif clawdata.checkpt_style == 2:
        # Specify a list of checkpoint times.  
        clawdata.checkpt_times = [0.1,0.15]

    elif clawdata.checkpt_style == 3:
        # Checkpoint every checkpt_interval timesteps (on Level 1)
        # and at the final time.
        clawdata.checkpt_interval = 5

    # ---------------
    # AMR parameters:
    # ---------------
    amrdata = rundata.amrdata

    ##################################### CHECK the following ###########
    # List of refinement ratios at each level (length at least mxnest-1)
    # max number of refinement levels:
    # initial run use less levels to start  
    # amrdata.amr_levels_max = len(amrdata.refinement_ratios_x)+1  # SET Maximum number of refinement levels
    amrdata.amr_levels_max = AMR_levels_max  # SET Maximum number of refinement levels
    print 'amrdata.amr_levels_max =', amrdata.amr_levels_max

	# AMR ratios
    amrdata.refinement_ratios_x = AMR_ratios  # SET x Refinement ratios
    amrdata.refinement_ratios_y = AMR_ratios  # SET y Refinement ratios
    amrdata.refinement_ratios_t = AMR_ratios  # SET t Refinement ratios
        
    AMR2_xresolution = AMR1_xresolution/amrdata.refinement_ratios_x[0]
    AMR2_yresolution = AMR1_yresolution/amrdata.refinement_ratios_y[0]
    AMR3_xresolution = AMR2_xresolution/amrdata.refinement_ratios_x[1]
    AMR3_yresolution = AMR2_yresolution/amrdata.refinement_ratios_y[1]
    AMR4_xresolution = AMR3_xresolution/amrdata.refinement_ratios_x[2]
    AMR4_yresolution = AMR3_yresolution/amrdata.refinement_ratios_y[2]
    AMR5_xresolution = AMR4_xresolution/amrdata.refinement_ratios_x[3]
    AMR5_yresolution = AMR4_yresolution/amrdata.refinement_ratios_y[3]

    print '\nIn degrees:'
    print 'AMR1_xresolution =', AMR1_xresolution
    print 'AMR1_yresolution =', AMR1_yresolution
    print 'AMR2_xresolution =', AMR2_xresolution
    print 'AMR2_yresolution =', AMR2_yresolution
    print 'AMR3_xresolution =', AMR3_xresolution
    print 'AMR3_yresolution =', AMR3_yresolution
    print 'AMR4_xresolution =', AMR4_xresolution
    print 'AMR4_yresolution =', AMR4_yresolution
    print 'AMR5_xresolution =', AMR5_xresolution
    print 'AMR5_yresolution =', AMR5_yresolution
        
    ######################################################################

    # Specify type of each aux variable in amrdata.auxtype.
    # This must be a list of length maux, each element of which is one of:
    #   'center',  'capacity', 'xleft', or 'yleft'  (see documentation).

    amrdata.aux_type = ['center','capacity','yleft']

    # Flag using refinement routine flag2refine rather than richardson error
    amrdata.flag_richardson = False    # use Richardson?
    amrdata.flag2refine = True

    # steps to take on each level L between regriddings of level L+1:
    amrdata.regrid_interval = 3
    # amrdata.regrid_interval = int(2.0*t_minutes_run*60.) # Must be integer.  Here set to twice the run duration, to prevent AMR regridding (regions forced to remain at one level of resolution.  See regions.append statements, below, in which minlevel == maxlevel.)

    # width of buffer zone around flagged points:
    # (typically the same as regrid_interval so waves don't escape):
    amrdata.regrid_buffer_width  = 3
    # amrdata.regrid_buffer_width  = int(2.0*t_minutes_run*60.)
    
    # clustering alg. cutoff for (# flagged pts) / (total # of cells refined)
    # (closer to 1.0 => more small grids may be needed to cover flagged cells)
    amrdata.clustering_cutoff = 0.700000

    # print info about each regridding up to this level:
    amrdata.verbosity_regrid = 0
    
    print 'amrdata.flag_richardson:', amrdata.flag_richardson 
    print 'amrdata.flag2refine:', amrdata.flag2refine
    print 'amrdata.regrid_interval:', amrdata.regrid_interval
    print 'amrdata.regrid_buffer_width:', amrdata.regrid_buffer_width
    print 'amrdata.clustering_cutoff:', amrdata.clustering_cutoff
    print '*****************************************\n'

    #  ----- For developers ----- 
    # Toggle debugging print statements:
    amrdata.dprint = False      # print domain flags
    amrdata.eprint = False      # print err est flags
    amrdata.edebug = False      # even more err est flags
    amrdata.gprint = False      # grid bisection/clustering
    amrdata.nprint = False      # proper nesting output
    amrdata.pprint = False      # proj. of tagged points
    amrdata.rprint = False      # print regridding summary
    amrdata.sprint = False      # space/memory output
    amrdata.tprint = True       # time step reporting each level
    amrdata.uprint = False      # update/upbnd reporting
    
    # More AMR parameters can be set -- see the defaults in pyclaw/data.py

    ############################## CHECK the following regions always ####################################
    #
    # ---------------
    # Regions:
    # ---------------
    
    rundata.regiondata.regions = []
    # to specify regions of refinement append lines of the form
    #  [minlevel,maxlevel,t1,t2,x1,x2,y1,y2], the t1, t2 in seconds

    cosyavg = np.cos(np.radians(0.5*(CompDom[2]+CompDom[3])))
    print '\nIn meters:'
    print 'AMR1_xresolution =', AMR1_xresolution*111132.*cosyavg
    print 'AMR1_yresolution =', AMR1_yresolution*111132.
    print 'AMR2_xresolution =', AMR2_xresolution*111132.*cosyavg
    print 'AMR2_yresolution =', AMR2_yresolution*111132.
    print 'AMR3_xresolution =', AMR3_xresolution*111132.*cosyavg
    print 'AMR3_yresolution =', AMR3_yresolution*111132.
    print 'AMR4_xresolution =', AMR4_xresolution*111132.*cosyavg
    print 'AMR4_yresolution =', AMR4_yresolution*111132.
    print 'AMR5_xresolution =', AMR5_xresolution*111132.*cosyavg
    print 'AMR5_yresolution =', AMR5_yresolution*111132.
    print ''

    # Region 0 = Computational Domain + Extra Cells
    xextra = 2.*AMR1_xresolution
    yextra = 2.*AMR1_yresolution
    print 'Extra cells.  xextra, yextra =', xextra, yextra
    minlev = 2; maxlev = 3
    print '\nComp Dom + extra cells minlev, maxlev = ', minlev, maxlev
    rundata.regiondata.regions.append([minlev,maxlev,-1.e10,1.e10,x1_domain-xextra, \
        x2_domain+xextra,y1_domain-yextra,y2_domain+yextra])
    
    # Region 1 = Source
    minlev = 2; maxlev = 3
    print '\nSource minlev, maxlev = ', minlev, maxlev
    rundata.regiondata.regions.append([minlev,maxlev,-1.e10,1e10, \
        SourceExtent[0],SourceExtent[1],SourceExtent[2],SourceExtent[3]])
    
    rundata.regiondata.regions.append([4,5,-1.e10,1e10, \
        TopoName2Extent[0],TopoName2Extent[1],TopoName2Extent[2],TopoName2Extent[3]])

        
    # -------------------------------------------------
    # Fixed Grids
    # -------------------------------------------------
    
    for iFGgrid in range(1,len(FGextent)+1):
        print '\niFGgrid = ', iFGgrid
        x1=FGextent[iFGgrid][0]; x2=FGextent[iFGgrid][1];
        y1=FGextent[iFGgrid][2]; y2=FGextent[iFGgrid][3]
        print 'x1,x2,y1,y2 = ', x1,x2,y1,y2
        x1c,y1c = gm.adjust(x1,y1,AMR3_xresolution,AMR3_yresolution,x1_domain,y1_domain)
        x2c,y2c = gm.adjust(x2,y2,AMR3_xresolution,AMR3_yresolution,x1_domain,y1_domain)
        print 'x1c,x2c,y1c,y2c = ', x1c,x2c,y1c,y2c
        rundata.regiondata.regions.append([5,5, -1e10, 1e10, x1c,x2c,y1c,y2c])
            
    # ---------------
    # Gauges:
    # ---------------
    rundata.gaugedata.gauges = []
    #     
    # SET Gauge Locations
    filename = project_dir+'/AllSites/LongBeach/_GaugeStatements/set_gauges.py'
    execfile(filename)    
    
    # # SET Assembly Areas and Critical Facilities Locations
    # filename = project_dir+'/AllSites/NeahMakah/_GaugeStatements/run_CF_gauges.py'
    # execfile(filename)
    #######################################################################################################
    
    return rundata
    # end of function setrun
    # ----------------------

#-------------------
def setgeo(rundata):
#-------------------
    """
    Set GeoClaw specific runtime parameters.
    For documentation see ....
    """

    try:
        geo_data = rundata.geo_data
    except:
        print "*** Error, this rundata has no geo_data attribute"
        raise AttributeError("Missing geo_data attribute")
       
    # == Physics ==
    geo_data.gravity = 9.81
    geo_data.coordinate_system = 2
    geo_data.earth_radius = 6367.5e3

    # == Forcing Options
    geo_data.coriolis_forcing = False

    # == Algorithm and Initial Conditions ==
    geo_data.sea_level = 0.0
    geo_data.dry_tolerance = 1.e-3
    geo_data.friction_forcing = True
    geo_data.manning_coefficient =.025
    geo_data.friction_depth = 1e6

    # Refinement settings
    refinement_data = rundata.refinement_data
    refinement_data.variable_dt_refinement_ratios = True
    refinement_data.wave_tolerance = 1.e-1
    refinement_data.deep_depth = 1e2
    refinement_data.max_level_deep = 3

    # import os
    # try:
    #     PacificCounty = os.environ['PacificCounty']
    # except:
    #     raise Exception("Need to set PacificCounty environment variable")
          ############################################################################################
    # == settopo.data values ==  (BathyTopo files)
    topo_data = rundata.topo_data
    # for topography, append lines of the form
    #    [topotype, minlevel, maxlevel, t1, t2, fname]
    
    fname = project_dir+'/topo/'+TopoName1
    print 'fname = ', fname
    topo_data.topofiles.append([3,1,5,0.,1.e10,fname])

    fname = project_dir+'/topo/'+TopoName2
    print 'fname = ', fname
    topo_data.topofiles.append([3,1,5,0.,1.e10,fname])

    fname = project_dir+'/topo/'+TopoName3
    print 'fname = ', fname
    topo_data.topofiles.append([3,1,5,0.,1.e10,fname])

    # == setdtopo.data values == (Source files)
    dtopo_data = rundata.dtopo_data
    # for moving topography, append lines of the form :   (<= 1 allowed for now!)
    #   [topotype, minlevel,maxlevel,fname]
    
    fname = project_dir+'/dtopo/'+SourceName
    print 'fname = ', fname
    dtopo_data.dtopofiles.append([3,1,5,fname])
    rundata.dtopo_data.dt_max_dtopo = 1.0  # max time step while topo moving
    ############################################################################################

    # == setqinit.data values ==
    rundata.qinit_data.qinit_type = 0
    rundata.qinit_data.qinitfiles = []
    # for qinit perturbations, append lines of the form: (<= 1 allowed for now!)
    #   [minlev, maxlev, fname]

    # == setfixedgrids.data values ==
    fixed_grids = rundata.fixed_grid_data
    
    # for fixed grids append lines of the form
    # [t1,t2,noutput,x1,x2,y1,y2,xpoints,ypoints,\
    #  ioutarrivaltimes,ioutsurfacemax]
    
    # == fgmax.data values ==
    fgmax_files = rundata.fgmax_data.fgmax_files
    # for fixed grids append to this list names of any fgmax input files
    
    for iFGgrid in range(1,len(FGextent)+1):
        FGname = 'fgmax'+str(int(iFGgrid))+'.txt'
        print '\niFGgrid, FGname = ', iFGgrid, FGname
        fgmax_files.append(FGname) 

    # rundata.fgmax_data.num_fgmax_val = 1  # Save depth only
    # rundata.fgmax_data.num_fgmax_val = 2  # Save depth and speed
    rundata.fgmax_data.num_fgmax_val = 5  # Save depth, speed, momentum, mom flux and min depth

    return rundata
    # end of function setgeo
    # ----------------------

if __name__ == '__main__':
    # Set up run-time parameters and write all data files.
    import sys
    rundata = setrun(*sys.argv[1:])
    rundata.write()
    from clawpack.geoclaw import kmltools
    kmltools.regions2kml(rundata)
    kmltools.gauges2kml(rundata)