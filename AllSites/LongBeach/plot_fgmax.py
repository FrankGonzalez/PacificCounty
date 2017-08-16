"""
Plot fgmax output from GeoClaw run.
"""
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy import ma
from clawpack.geoclaw import fgmax_tools, geoplot, dtopotools
import LibFGmaxPlot as LFGP
from openpyxl import load_workbook
# For info on openpyxl go to  https://openpyxl.readthedocs.org

# Set directories
ModelingDir = os.environ['ModelingDir']
sys.path.append(ModelingDir + '/FranksLib')
import interptools

# Set directory folders

ProjectDir = 'PacificCounty'
AllSitesDir = 'AllSites'  # AllSites Directory (in Project Directory)
SiteDir = 'LongBeach' # Individual Site Directory in SitesDir
outdir = '_output'     # GeoClaw data output directory
plotdir='_plots'             # Save results in this directory
otherdir = '_other_figures'  # directory for other figures for each site

print '\nDIRECTORY FOLDERS:'
print 'ProjectDir=',ProjectDir
print 'SitesDir=',AllSitesDir   # where to find output
print 'SiteDir=',SiteDir   # where to find output
print 'outdir=',outdir
print 'plotdir=',plotdir  # where to put plots
print 'otherdir=',otherdir  # where to put plots

# ProjectDir = os.environ[ProjectDir]
ProjectDir = os.path.join(ModelingDir,ProjectDir)
if not os.path.isdir(ProjectDir):
    raise Exception("Missing directory: %s" % ProjectDir)
AllSitesDir = os.path.join(ProjectDir,AllSitesDir)
if not os.path.isdir(AllSitesDir):
    raise Exception("Missing directory: %s" % AllSitesDir)
SiteDir = os.path.join(AllSitesDir,SiteDir)
if not os.path.isdir(SiteDir):
    raise Exception("Missing directory: %s" % SiteDir)
outdir = os.path.join(SiteDir, outdir)   # where to find output
if not os.path.isdir(outdir):
    raise Exception("Missing directory: %s" % outdir)
plotdir = os.path.join(SiteDir,plotdir)  # plots directory
print 'plotdir=',plotdir
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)
otherdir = os.path.join(plotdir,otherdir) # where to put these fgmax plots
print 'otherdir=',otherdir
if not os.path.isdir(otherdir):
    os.mkdir(otherdir)
    
print 'DIRECTORIES:'
print 'ProjectDir=',ProjectDir
print 'AllSitesDir=',AllSitesDir   # where to find output
print 'SiteDir=',SiteDir   # where to find output
print 'outdir=',outdir
print 'plotdir=',plotdir  # plots directory 
print 'otherdir=',otherdir  # where to put the fgmax plots

# Google Earth image for plotting on top of...
# need to change for specific location on coast:
# This version will plot multiple fixed grid (FG) solutions
# Add or subtract the dictionary entries, below, for each FG, as appropriate

dtoponame = 'L1_60s_linear_x1.11.tt3'
dtopofile = ProjectDir + '/dtopo/' + dtoponame
dtopotype = 3  # format of dtopo file


FGextent = {}
TopoName3 = 'astoria_v3_S-124.07_-124.045_46.341_46.356.asc'
TopoName3Extent = (-124.07,-124.045,46.341,46.356) # Berm
FGextent[0]   = (TopoName3Extent[0],TopoName3Extent[1], \
    TopoName3Extent[2],TopoName3Extent[3]) # Berm
    
ngrids = len (FGextent)
print 'ngrids=',ngrids

Site = {}
Site[0] = ('LongBeachBerm')

GEmap = range(ngrids)

# Specify the plots to be created (assigned 'True') for each fixed grid

plot_zeta = True                           # Plot zeta w/o GE image
plot_speed = True                          # Plot max flow speed w/o GE Map 
plot_haz_depth = False                 
plot_arrival_time = True                  # Plot first arrival w/o GE image
plot_zeta_arrival_time = True                     # Plot max first arrival w/o GE image
plot_hs = False                           # Plot hs w/o GE image
plot_hss = False                           # Plot hss w/o GE image
# plot_arrival_time_on_zeta = False          # Plot contours of arrival time on depth?

plot_zeta_map = False                        # Plot zeta on GE image
plot_speed_map = False                      # Plot max flow speed on GE Map
plot_haz_depth_map = False
plot_arrival_time_map = False              # Plot first arrival on GE image
plot_zeta_arrival_time_map = False                 # Plot max first arrival on GE image
plot_hs_map = False                          # Plot hs on GE image
plot_hss_map = False                        # Plot hss on GE image


mask_offshore = True  # Mask offshore values (or not)

# Set contour line levels

clines_zeta = [0.,1.,2.,3.,4.,5.]   # meter
clines_arrival_time = [0,2,4,6,8,10,15,20,30,40,50,60]  # minutes
clines_zeta_arrival_time = [0,2,4,6,8,10,15,20,30,40,50,60]  # minutes
clines_speed = [0,.5,1,2,3,4,5,6,8,10,12,14,16]  # contours for speed m/s
clines_hs = [0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]     # contours for momentum
clines_hss = [0,50,100,200,300,400,500,600,700,800,900,1000] # momentum flux
clines_haz_depth = [0,1,2,3,4,5]     # contours for hazardous depth due to drawdown

# Define criteria (tolerances, thresholds or conditions) for values that are onshore, offshore, dry, or zero
sea_level = 0.      # NOTE:  Sea level is relative to the datum of the computational grid (usually MHW)
dry_land = .01
zero_speed = 0.1
safe_depth = 5.0

# Specify the units for each variable
units_zeta = '[m]'
units_arrival_time = '[min]'
units_zeta_arrival_time = '[min]'
units_speed = '[m/s]'
units_hs = '[m^2/s]'
units_hss = '[m^3/s^2]'
units_haz_depth = '[m]'

gridnos = []
gridnos = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028']

print ''
print 'type(Site)=',type(Site)
print 'shape(Site)=',np.shape(Site)
print 'type(GEmap)=',type(GEmap)
print 'shape(GEmap)=',np.shape(GEmap)
print 'type(FGextent)=',type(FGextent)
print 'shape(FGextent)=',np.shape(FGextent)
print 'type(gridnos)=',type(gridnos)
print 'shape(gridnos)=',np.shape(gridnos)
print ''
print 'Site =\n', Site
print 'GEmap =\n', GEmap
print 'FGextent =\n', FGextent

ngrids = len (FGextent)
print 'ngrids=',ngrids

for gridno in range(1,ngrids+1):
    
    # Specify input file names for each FG
    fgmax_txt_file = 'fgmax'+str(int(gridno))+'.txt' # FG description file name (Frank: use dictionary)
    fgmax_txt_file = os.path.join(SiteDir, fgmax_txt_file)      # grid description  (this is the file to be read in)
    # fname = outdir + '/fort.FG%s.valuemax' % gridno        # FG data file name (GeoClaw output file)
    fname = outdir + '/fort.FG'+gridnos[gridno-1]+'.valuemax'       # FG data file name (GeoClaw output file)
    
    print '\n********* Start ', Site[gridno-1], ' plots ************'
    print "gridno = ", gridno
    print "Site = ", Site[gridno-1]
    print "Grid Extent.  x1,x2,y1,y2 = ", FGextent[gridno-1]
    print "Reading output from ",outdir
    print "Using fgmax input from ",fgmax_txt_file
    print "Reading %s ..." % fname
    print "fname = ", fname
    
    # Read in the data for this FGmax grid and form the variables to be plotted
    fg = fgmax_tools.FGmaxGrid()
    fg.read_input_data(input_file_name=fgmax_txt_file)  # Read in the FG description file
    fg.read_output(fgno=gridnos[gridno-1],outdir=outdir) # Read the GeoClaw results on the fgmax grid numbered *fgno*.
    # Read in topography and find the average bottom deformation
    # Code to get B0, B, and dzi on the fixed grid defined by X and Y
    dtopo = dtopotools.DTopography()
    dtopo.read(dtopofile, dtopotype)

    try:
        dzi = interptools.interp(fg.X,fg.Y,dtopo.X,dtopo.Y,dtopo.dZ[1,:,:])
        print "Average subsidence/uplift over ENTIRE fgmax grid:  %s m"  % dzi.mean()
    except:
        try:
            xmid = fg.X.mean()
            ymid = fg.Y.mean()
            dzi = interptools.interp(xmid,ymid,dtopo.X,dtopo.Y,dtopo.dZ[1,:,:])
            print "Subsidence/uplift at MIDPOINT of fgmax grid:  %s m"  % dzi
        except:
            dzi = 0.
            print "NOTE: fgmax grid does not overlap dtopo, so no subsidence/uplift"
    
    x,y = fg.X, fg.Y  # grid
    y_ave = y.mean()  # for scaling by latitude
    print 'y_ave=',y_ave
    h = fg.h  # max depth
    arrival_time = fg.arrival_time / 60. # First wave arrival time, converted to minutes
    zeta_arrival_time = fg.h_time / 60.  # Maximum flooding arrival time, converted to minutes
    speed = fg.s
    hs = fg.hs
    # hs = np.ma.getdata(hs)
    hss = fg.hss
    haz_depth = -fg.hmin
    B = fg.B  # Post-seismic topography, on which GeoClaw computations are made
    B0 = B - dzi # Pre-seismic (original) topography, i.e., Post-seismic topo B, adjusted by seismic subsidence/uplift
    eta = h + B # eta is the wave height, referenced to sea_level (B is positive up, negative down)    
    
    # Logical Conditions for the masking operations
    speed_is_zero = (speed < zero_speed)          # Speed is zero when speed < zero_speed
    cell_is_onshore = (B0 >= sea_level)   # Cell is onshore relative to original, pre-seismic bathymetry, B0
    cell_is_offshore = (B0 < sea_level)   # Cell is offshore relative to original, pre-seismic bathymetry, B0
    # cell_is_onshore = (B >= sea_level)  # Cell is onshore relative to post-seismic bathymetry, B
    # cell_is_offshore = (B < sea_level)  # Cell is offshore relative to post-seismic bathymetry, B
    cell_is_dry = (h < dry_land)           # Cell is dry
    cell_depth_is_safe = (haz_depth > safe_depth)      # Cell is at a safe depth for vessels
    
    # Form the variables to be plotted
    # NOTE:  The speed, hs, hss, haz_depth data might not be available:
    #        To get them, set rundata.fgmax_data.num_fgmax_val = 2 or 5 in setrun.py
    
    zeta = np.where(cell_is_onshore, h, eta)   # DEFINE zeta == h onshore, eta offshore
    print 'Created zeta'
    print 'type(zeta)=',type(zeta)
    print 'shape(zeta)=',np.shape(zeta)
    
    if mask_offshore:
        
        # Find only onshore, wet cell, non-zero zeta values
        # zeta = np.where(cell_is_onshore, h, eta)   # DEFINE zeta == h onshore, eta offshore
        zeta = ma.masked_where(cell_is_offshore, zeta)  # Mask all offshore cells
        zeta = ma.masked_where(cell_is_dry, zeta)  # Mask all dry cells
        # Find only onshore, wet cell, first wave arrival time values
        arrival_time = ma.masked_where(cell_is_offshore, arrival_time) # Mask all offshore cells
        arrival_time = ma.masked_where(cell_is_dry, arrival_time)  # Mask all dry cells
        # Find only onshore, wet cell, maximum wave arrival time values
        zeta_arrival_time = ma.masked_where(cell_is_offshore, zeta_arrival_time) # Mask all offshore cells
        zeta_arrival_time = ma.masked_where(cell_is_dry, zeta_arrival_time)  # Mask all dry cells
        # Find only onshore, wet cell, non-zero speed values
        # For ports and maritime hazard comment out the next line that masks offshore cells  
        # speed = ma.masked_where(cell_is_offshore, speed)  # Mask all offshore cells 
        speed = ma.masked_where(speed_is_zero, speed) # Mask all cells with zero water speed
        speed = ma.masked_where(cell_is_dry, speed) # Mask all dry cells
        # Find only onshore, wet cell, non-zero momentum (hs) values
        hs = ma.masked_where(cell_is_dry, hs) # Mask all dry cells
        hs = ma.masked_where(cell_is_offshore, hs)  # Mask all offshore cells
        # Find only onshore, wet cell, non-zero momentum flux (hss) values
        hss = ma.masked_where(cell_is_offshore, hss)  # Mask all offshore cells
        hss = ma.masked_where(cell_is_dry, hss) # Mask all dry cells
        # Find only offshore, hazardously shallow, depth values
        haz_depth = ma.masked_where(cell_is_onshore, haz_depth) # Mask all onshore cells
        haz_depth = ma.masked_where(cell_depth_is_safe, haz_depth) # Mask all cells at a safe depth

    cmap_name='FG_cmap'; minmax=False
    if plot_zeta:
        on_map = False
        LFGP.plot_v('01','max_zeta',units_zeta,x,y,zeta,clines_zeta,cmap_name,minmax,on_map,B0,\
           sea_level,GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)
    if plot_zeta_map:
        on_map = True
        LFGP.plot_v('02','max_zeta',units_zeta,x,y,zeta,clines_zeta,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)

    cmap_name='FG_cmap'; minmax=False
    if plot_speed:
       on_map = False
       LFGP.plot_v('03','max_speed',units_speed,x,y,speed,clines_speed,\
            cmap_name,minmax,on_map,B0,sea_level,\
            GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)
    if plot_speed_map:
       on_map = True
       LFGP.plot_v('04','max_speed',units_speed,x,y,speed,clines_speed,\
            cmap_name,minmax,on_map,B0,sea_level,\
            GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)
            
    cmap_name='FG_cmap_r'; minmax=False
    if plot_haz_depth:
        on_map = False
        LFGP.plot_v('05','min_haz_depth',units_haz_depth,x,y,haz_depth,\
           clines_haz_depth,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)
    if plot_haz_depth_map:
        on_map = True
        LFGP.plot_v('06','min_haz_depth',units_haz_depth,x,y,haz_depth,\
           clines_haz_depth,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)

    cmap_name='FG_cmap_r'; minmax=False
    if plot_arrival_time:
       on_map = False
       # clines_arrival_time=clines_arrival_time.reverse()
       LFGP.plot_v('07','first_arrival_time',units_arrival_time,x,y,arrival_time,\
          clines_arrival_time,cmap_name,minmax,on_map,B0,sea_level,\
          GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)
    if plot_arrival_time_map:
       on_map = True
       # clines_arrival_time=clines_arrival_time.reverse()
       LFGP.plot_v('08','first_arrival_time',units_arrival_time,x,y,arrival_time,\
          clines_arrival_time,cmap_name,minmax,on_map,B0,sea_level,\
          GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)

    cmap_name='FG_cmap_r'; minmax=False
    if plot_zeta_arrival_time:
        on_map = False
        LFGP.plot_v('09','max_zeta_arrival_time',units_zeta_arrival_time,\
           x,y,zeta_arrival_time,clines_zeta_arrival_time,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)
    if plot_zeta_arrival_time_map:
        on_map = True
        LFGP.plot_v('10','max_zeta_arrival_time',units_zeta_arrival_time,x,y,zeta_arrival_time,\
           clines_zeta_arrival_time,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)

    cmap_name='FG_cmap'; minmax=False
    if plot_hs:
        on_map = False
        LFGP.plot_v('11','max_hs',units_hs,x,y,hs,clines_hs,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)
    if plot_hs_map:
        on_map = True
        LFGP.plot_v('12','max_hs',units_hs,x,y,hs,clines_hs,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)

    cmap_name='FG_cmap'; minmax=False
    if plot_hss:
        on_map = False
        LFGP.plot_v('13','max_hss',units_hss,x,y,hss,clines_hss,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)
    if plot_hss_map:
        on_map = True
        LFGP.plot_v('14','max_hss',units_hss,x,y,hss,clines_hss,cmap_name,minmax,on_map,B0,sea_level,\
           GEmap[gridno-1],FGextent[gridno-1],Site[gridno-1],otherdir)