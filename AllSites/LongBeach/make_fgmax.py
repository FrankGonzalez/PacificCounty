"""
NeahMakah
Create fgmax_gridno.txt input files 
This version adjusts the grid to center on domain 
"""
from clawpack.geoclaw import fgmax_tools
import numpy as np
import math as ma
import os
import sys

ModelingDir = os.environ['ModelingDir']
sys.path.append(ModelingDir + '/FranksLib')
import FGgauges_module as gm


# SET START AND END TIME FOR SIMULATION
t_start_min = -1.
t_run_min   =  5  # Must be integer

# RED REGION = COMPUTATIONAL DOMAIN
CompDom = (-127.5,-123.9,45.8,46.8)
x_num_cells = 36
y_num_cells = 10

AMR_ratios = (5,4,6,9)
AMR_levels_max = len(AMR_ratios)+1

# REGION 0 = AUTOMATIC.  COMPUTATIONAL DOMAIN + GHOST CELLS

# REGION 1 = SOURCE
dxy = 0.05
# SourceName = 'seattlefault_PMEL_2mt.tt3'
SourceName = 'L1_60s_linear_x1.11.tt3'
SourceExtent = (CompDom[0]-dxy,CompDom[1]+dxy,CompDom[2]-dxy,CompDom[3]+dxy)

# Topo files
TopoName1 = 'etopo1_-127.5_-123.9_45.8_46.8.asc'
xextra = 3./60.  # Etopo1 res is 1 min = 1/60 deg.  So this adds 3 cells.
yextra = 3./60.
TopoName1Extent = (CompDom[0]-xextra,CompDom[1]+xextra, \
    CompDom[2]-yextra,CompDom[3]+yextra)
TopoName2 = "astoria_v3_M-124.1_-124.0_46.31_46.38.asc"
TopoName2Extent = (-124.1,-124.0,46.31,46.38)
TopoName3 = 'astoria_v3_S-124.07_-124.045_46.341_46.356.asc'
TopoName3Extent = (-124.07,-124.045,46.341,46.356) # Berm

# Fixed Grids

FGextent = {}
FGextent[1]   = (TopoName2Extent[0],TopoName2Extent[1], \
    TopoName2Extent[2],TopoName2Extent[3]) # Medium grid
FGextent[2]   = (TopoName3Extent[0],TopoName3Extent[1], \
    TopoName3Extent[2],TopoName3Extent[3]) # Small Grid = Berm

print 'type(FGextent) = ', type(FGextent)
print 'np.shape(FGextent) = ', np.shape(FGextent)
print 'len(FGextent) = ', len(FGextent)

# REGION 5 = AUTOMATIC -- AROUND FIXED GRID

print '==================================================================='
print 'CompDom = ', CompDom
print 'x_num_cells, y_num_cells =', x_num_cells, y_num_cells
print 'AMR_ratios = ', AMR_ratios
print 'AMR_levels_max = ', AMR_levels_max
print 'SourceName = ', SourceName
print 'SourceExtent = ', SourceExtent
# print 'TopoName1 = ', TopoName1
# print 'RegionExtent1 = ', RegionExtent1
print '=================================================================='

# Computational Domain and AMR settings

x1_domain = CompDom[0]     # SET west longitude boundary
x2_domain = CompDom[1]     # SET east longitude boundary
y1_domain = CompDom[2]       # SET south latitude boundary
y2_domain = CompDom[3]       # SET north latitude boundary

AMR1_xresolution = (x2_domain-x1_domain)/x_num_cells
AMR1_yresolution = (y2_domain-y1_domain)/y_num_cells
AMR2_xresolution = AMR1_xresolution/AMR_ratios[0]
AMR2_yresolution = AMR1_yresolution/AMR_ratios[0]
AMR3_xresolution = AMR2_xresolution/AMR_ratios[1]
AMR3_yresolution = AMR2_yresolution/AMR_ratios[1]
AMR4_xresolution = AMR3_xresolution/AMR_ratios[2]
AMR4_yresolution = AMR3_yresolution/AMR_ratios[2]
AMR5_xresolution = AMR4_xresolution/AMR_ratios[3]
AMR5_yresolution = AMR4_yresolution/AMR_ratios[3]
print 'x1_domain,x2_domain:', x1_domain,x2_domain
print 'y1_domain,y2_domain:', y1_domain,y2_domain
print 'x_num_cells,y_num_cells',x_num_cells,y_num_cells
print 'AMR_ratios:',AMR_ratios
print 'AMR_levels_max: ', AMR_levels_max
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

# Default values (might be changed below)
tstart_max =  4.       # when to start monitoring max values
tend_max = 1.e10       # when to stop monitoring max values
dt_check = 60.         # target time (sec) increment between updating max values
min_level_check = 4   # which levels to monitor max on (FOR 5 LEVEL RUN)
arrival_tol = 1.e-2    # tolerance for flagging arrival

fg = fgmax_tools.FGmaxGrid()
fg.point_style = 2       # will specify a 2d grid of points
fg.tstart_max = tstart_max
fg.tend_max = tend_max
fg.dt_check = dt_check
fg.min_level_check = min_level_check
fg.arrival_tol = arrival_tol
print 'fg.tstart_max = ', fg.tstart_max
print 'fg.tend_max = ', fg.tend_max
print 'fg.dt_check = ', fg.dt_check
print 'fg.min_level_check = ', fg.min_level_check
print 'fg.arrival_tol = ', fg.arrival_tol

for iFGgrid in range(1,len(FGextent)+1):
    print '\n==================================================================\n'
    print '***** Fixed Grid Number ', iFGgrid
    x1=FGextent[iFGgrid][0]; x2=FGextent[iFGgrid][1]; y1=FGextent[iFGgrid][2]; y2=FGextent[iFGgrid][3]
    fg.x1,fg.x2,fg.nx,fg.y1,fg.y2,fg.ny=gm.adjust_fgmax_grid(x1,x2,y1,y2,x1_domain,y1_domain,AMR3_xresolution,AMR3_yresolution)
    dx1 = x1 - fg.x1
    dx2 = x2 - fg.x2
    dy1 = y1 - fg.y1
    dy2 = y2 - fg.y2
    dx1_m = dx1*ma.cos(ma.radians(y1))*111.132e3
    dx2_m = dx2*ma.cos(ma.radians(y2))*111.132e3
    dy1_m = dy1*111.132e3
    dy2_m = dy2*111.132e3

    print 'x1,fg.x1,dx1,dx1_m:',x1,fg.x1,dx1,dx1_m
    print 'x2,fg.x2,dx2,dx2_m:',x2,fg.x2,dx2,dx2_m
    print 'y1,fg.y1,dy1,dy1_m:',y1,fg.y1,dy1,dy1_m
    print 'y2,fg.y2,dy2,dy2_m:',y2,fg.y2,dy2,dy2_m
    fg.input_file_name = 'fgmax'+str(iFGgrid)+'.txt'
    fg.write_input_data()
    print '\n==================================================================\n'