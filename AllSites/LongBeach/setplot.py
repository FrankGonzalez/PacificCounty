""" 
Used with clawpack-5.3.1

Set up the plot figures, axes, and items to be done for each frame.

This module is imported by the plotting routines and then the
function setplot is called to set the plot parameters.

Gauge plot data are in the GeoClaw output file:  _output/fort.gauge
which has 7 columns:   gaugeno level t h hu hv eta

In setrun.py, set rundata.fgmax_data.num_fgmax_val = 5  # To save h hu hv eta
""" 

import pylab
import glob, os
from numpy import loadtxt
import matplotlib.pyplot as plt
from matplotlib import image

CompDom = (-127.5,-123.9,45.8,46.8)
LongBeachMediumExtent = (-124.1,-124.0,46.31,46.38) # Medium
LongBeachBermExtent = (-124.068,-124.043,46.3425,46.3575) # Berm


# Set directory folders

ProjectDir = 'PacificCounty'
SitesDir = 'AllSites'  # AllSites Directory (in Project Directory)
SiteDir = 'LongBeach' # Individual Site Directory in SitesDir
outdir = '_output'     # GeoClaw data output directory
plotdir='_plots'             # Save results in this directory
otherdir = '_other_figures'  # directory for other figures for each site

print '\nDIRECTORY FOLDERS:'
print 'ProjectDir=',ProjectDir
print 'SitesDir=',SitesDir   # where to find output
print 'SiteDir=',SiteDir   # where to find output
print 'outdir=',outdir
print 'plotdir=',plotdir  # where to put plots
print 'otherdir=',otherdir  # where to put plots

# Set directory paths (or create the directory folder and path)
ProjectDir = os.environ[ProjectDir]
if not os.path.isdir(ProjectDir):
    raise Exception("Missing directory: %s" % ProjectDir)
SitesDir = os.path.join(ProjectDir,SitesDir)
if not os.path.isdir(SitesDir):
    raise Exception("Missing directory: %s" % SitesDir)
SiteDir = os.path.join(SitesDir,SiteDir)
if not os.path.isdir(SiteDir):
    raise Exception("Missing directory: %s" % SiteDir)
outdir = os.path.join(SiteDir, outdir)   # where to find output
if not os.path.isdir(outdir):
    raise Exception("Missing directory: %s" % outdir)
plotdir = os.path.join(SiteDir,plotdir)  # where to put plots
if not os.path.isdir(plotdir):
    print '*** Make plotdir=',plotdir
    os.mkdir(plotdir)
otherdir = os.path.join(plotdir,otherdir)  # where to put other figures
if not os.path.isdir(otherdir):
    print '*** make otherdir =',otherdir
    os.mkdir(otherdir)
    
print '\nDIRECTORIES:'
print 'ProjectDir=',ProjectDir
print 'SitesDir=',SitesDir   # where to find output
print 'SiteDir=',SiteDir   # where to ixed output
print 'outdir=',outdir
print 'plotdir=',plotdir  # where to put plots
print 'otherdir=',otherdir  # where to put plots

FGsite = {}
fgmax_input_file = {}
FGextent = {}

FGsite[1] = "LongBeachBerm"
fgmax_input_file[1] = "fgmax1.txt"
FGextent[1] = (-124.07,-124.045,46.341,46.356) # Berm

ngrids = len(FGsite)
gridnos = range(1,ngrids+1)
for gridno in gridnos:
    print 'gridno, FGsite =',gridno, FGsite[gridno]

# --------------------------
def setplot(plotdata):
# --------------------------
    
    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of clawpack.visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    
    """ 

    from clawpack.visclaw import colormaps, geoplot

    plotdata.clearfigures()  # clear any old figures,axes,items dat
    plotdata.format = 'binary'

    clim_ocean = 8.0
    clim_coast = 8.0
    
    sealevel = 0.  # Level of tide in run relative to MHW
    cmax_ocean = clim_ocean + sealevel
    cmin_ocean = -clim_ocean + sealevel
    cmax_coast = clim_coast + sealevel
    cmin_coast = -clim_coast + sealevel
    
    
    # To plot gauge locations on pcolor or contour plot, use this as
    # an afteraxis function:
    
    def addgauges(current_data):
        from clawpack.visclaw import gaugetools
        gaugetools.plot_gauge_locations(current_data.plotdata, \
             gaugenos='all', format_string='ko', add_labels=True, \
             markersize=5, fontsize=10, xoffset=0, yoffset=0)
    
    def timeformat(t):
        from numpy import mod, sign
        signt = sign(t)
        t = abs(t)
        hours = int(t/3600.) # seconds to integer number of hours
        tmin = mod(t,3600.)  # seconds of remaining time beyond integer number of hours
        min = int(tmin/60.)  # seconds to integer number of minutes
        sec = int(mod(tmin,60.)) # remaining integer sec
        tenth_sec = int(10*(t - int(t)))
        timestr = '%s:%s:%s.%s' % (hours,str(min).zfill(2),str(sec).zfill(2),str(tenth_sec).zfill(1))
        if signt < 0:
            timestr = '-' + timestr
        return timestr
        
    def title_hours(current_data):
        from pylab import title
        t = current_data.t
        timestr = timeformat(t)
        title('%s after earthquake' % timestr)

    # -----------------------------------------
    # Figure for Computational Domain
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Computational Domain', figno=101)
    plotfigure.kwargs = {'figsize': (8,10)}
    plotfigure.show = True
    
    # Set up for axes in this figure
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'Computational Domain'
    plotaxes.scaled = False
    print 'CompDom = ', CompDom
    plotaxes.xlimits = [CompDom[0], CompDom[1]] 
    plotaxes.ylimits = [CompDom[2], CompDom[3]]

    def aa(current_data):
        from pylab import ticklabel_format, xticks, gca, cos, pi, savefig
        title_hours(current_data)
        ticklabel_format(format='plain',useOffset=False)
        xticks(rotation=20)
        a = gca()
        a.set_aspect(1./cos(48*pi/180.))
    plotaxes.afteraxes = aa
    plotaxes.afteraxes = addgauges
    
    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.surface_or_depth
    my_cmap = colormaps.make_colormap({-1.0: [0.0,0.0,1.0], \
                                     -0.5: [0.5,0.5,1.0], \
                                      0.0: [1.0,1.0,1.0], \
                                      0.5: [1.0,0.5,0.5], \
                                      1.0: [1.0,0.0,0.0]})
    plotitem.imshow_cmap = my_cmap
    # plotitem.imshow_cmin = cmin_ocean
    # plotitem.imshow_cmax = cmax_ocean
    plotitem.imshow_cmin = -15.
    plotitem.imshow_cmax = 15.
    plotitem.add_colorbar = True
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0]
    
    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.land
    plotitem.imshow_cmap = geoplot.land_colors
    plotitem.imshow_cmin = 0.0
    plotitem.imshow_cmax = 100.0
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0]
    
    # Add contour lines of bathymetry:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = False
    plotitem.plot_var = geoplot.topo
    from numpy import arange, linspace
    plotitem.contour_levels = linspace(-6000,0,7)
    plotitem.amr_contour_colors = ['g']  # color on each level
    plotitem.kwargs = {'linestyles':'solid'}
    plotitem.amr_contour_show = [0,0,1,0]  # show contours only on finest level
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0
    
    # Add contour lines of topography:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = False
    plotitem.plot_var = geoplot.topo
    from numpy import arange, linspace
    plotitem.contour_levels = arange(0., 11., 1.)
    plotitem.amr_contour_colors = ['g']  # color on each level
    plotitem.kwargs = {'linestyles':'solid'}
    plotitem.amr_contour_show = [0,0,0,1]  # show contours only on finest level
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0
    
    # -----------------------------------------
    # Figure for Long Beach Coast
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Long Beach Coast', figno=102)
    plotfigure.kwargs = {'figsize': (8,10)}
    plotfigure.show = True
    
    # Set up for axes in this figure
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'Long Beach Coast'
    plotaxes.scaled = False
    print 'CompDom = ', CompDom
    plotaxes.xlimits = [-124.5, CompDom[1]] 
    plotaxes.ylimits = [CompDom[2], CompDom[3]]

    def aa(current_data):
        from pylab import ticklabel_format, xticks, gca, cos, pi, savefig
        title_hours(current_data)
        ticklabel_format(format='plain',useOffset=False)
        xticks(rotation=20)
        a = gca()
        a.set_aspect(1./cos(48*pi/180.))
    plotaxes.afteraxes = aa
    plotaxes.afteraxes = addgauges
    
    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.surface_or_depth
    my_cmap = colormaps.make_colormap({-1.0: [0.0,0.0,1.0], \
                                     -0.5: [0.5,0.5,1.0], \
                                      0.0: [1.0,1.0,1.0], \
                                      0.5: [1.0,0.5,0.5], \
                                      1.0: [1.0,0.0,0.0]})
    plotitem.imshow_cmap = my_cmap
    # plotitem.imshow_cmin = cmin_ocean
    # plotitem.imshow_cmax = cmax_ocean
    plotitem.imshow_cmin = -15.
    plotitem.imshow_cmax = 15.
    plotitem.add_colorbar = True
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0]
    
    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.land
    plotitem.imshow_cmap = geoplot.land_colors
    plotitem.imshow_cmin = 0.0
    plotitem.imshow_cmax = 100.0
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0]
    
    # Add contour lines of bathymetry:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = False
    plotitem.plot_var = geoplot.topo
    from numpy import arange, linspace
    plotitem.contour_levels = linspace(-6000,0,7)
    plotitem.amr_contour_colors = ['g']  # color on each level
    plotitem.kwargs = {'linestyles':'solid'}
    plotitem.amr_contour_show = [0,0,1,0]  # show contours only on finest level
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0
    
    # Add contour lines of topography:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = False
    plotitem.plot_var = geoplot.topo
    from numpy import arange, linspace
    plotitem.contour_levels = arange(0., 11., 1.)
    plotitem.amr_contour_colors = ['g']  # color on each level
    plotitem.kwargs = {'linestyles':'solid'}
    plotitem.amr_contour_show = [0,0,0,1]  # show contours only on finest level
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0

    #-----------------------------------------
    # Figure for LongBeach Medium Grid
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name="LongBeachMedium", figno=103)
    plotfigure.show = True
    plotfigure.kwargs = {'figsize': (8,10)}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Long Beach Medium Grid"
    plotaxes.scaled = False
    plotaxes.xlimits = [LongBeachMediumExtent[0],LongBeachMediumExtent[1]]
    plotaxes.ylimits = [LongBeachMediumExtent[2],LongBeachMediumExtent[3]]
    plotaxes.afteraxes = aa
    plotaxes.afteraxes = addgauges

    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.surface_or_depth
    plotitem.imshow_cmap = my_cmap
    # plotitem.imshow_cmin = cmin_coast
    # plotitem.imshow_cmax = cmax_coast
    plotitem.imshow_cmin = -15.
    plotitem.imshow_cmax = 15.
    plotitem.add_colorbar = True
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0]

    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.land
    plotitem.imshow_cmap = geoplot.land_colors
    plotitem.imshow_cmin = 0.0
    plotitem.imshow_cmax = 100.0
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0]
    
    #-----------------------------------------
    # Figure for Long Beach Berm Grid
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name="LongBeachBerm", figno=104)
    plotfigure.show = True
    plotfigure.kwargs = {'figsize': (8,10)}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = "Long Beach Berm Grid"
    plotaxes.scaled = False
    plotaxes.xlimits = [LongBeachBermExtent[0],LongBeachBermExtent[1]]
    plotaxes.ylimits = [LongBeachBermExtent[2],LongBeachBermExtent[3]]
    plotaxes.afteraxes = aa
    plotaxes.afteraxes = addgauges

    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.surface_or_depth
    plotitem.imshow_cmap = my_cmap
    # plotitem.imshow_cmin = cmin_coast
    # plotitem.imshow_cmax = cmax_coast
    plotitem.imshow_cmin = -15.
    plotitem.imshow_cmax = 15.
    plotitem.add_colorbar = True
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0]

    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.land
    plotitem.imshow_cmap = geoplot.land_colors
    plotitem.imshow_cmin = 0.0
    plotitem.imshow_cmax = 100.0
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.amr_patchedges_show = [0]
    
    
    #-----------------------------------------
    # Figures for gauges
    #-----------------------------------------
    
    print 'Plot gauge data'
    
    # Plot eta, wave height wrt MHW
    plotfigure = plotdata.new_plotfigure(name='eta, Wave Height',figno=301,type='each_gauge')
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'eta'
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    def variables(current_data):
        from numpy import where, sqrt
        q = current_data.q
        h = q[0,:]
        hu = q[1,:]
        hv = q[2,:]
        eta = q[3,:]
        hss = where(h>0, (hu**2 + hv**2)/h, 0.)
        return eta
    plotitem.plot_var = variables
    plotitem.plotstyle = 'b-'
    print 'Completed wave height, eta, plot.'
    
    
    # Plot h, flow depth
    plotfigure = plotdata.new_plotfigure(name='Flood Depth',figno=302,type='each_gauge')
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'h, Flood Depth'
    # plotaxes.ylimits = [-10, 10]
    plotaxes.xlimits = [-60., 1800] 
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    def variables(current_data):
        from numpy import where
        from numpy import sqrt
        q = current_data.q
        h = q[0,:]
        hu = q[1,:]
        hv = q[2,:]
        return h
    
    plotitem.plot_var = variables
    plotitem.plotstyle = 'b-'
    plotaxes.xlimits = [-60., 1800] 
    # plotaxes.ylimits = [-10, 10]
    print 'Completed flow depth, h, plot.'
    # Plot topo as green curve:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    def gaugetopo(current_data):
        q = current_data.q
        h = q[0,:]
        eta = q[3,:]
        topo = eta - h
        return topo       
    plotitem.plot_var = gaugetopo
    plotitem.plotstyle = 'g-'
    # Plot zero line as black line:
    def add_zeroline(current_data):
        from pylab import plot, legend
        t = current_data.t
        legend(('surface','topography'),loc='lower left')
        plot(t, 0*t, ':k')
    plotaxes.afteraxes = add_zeroline
    
    #Plot speed, s
    plotfigure = plotdata.new_plotfigure(name='Speed',figno=303,type='each_gauge')
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 's, Current Speed'
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    def variables(current_data):
        from numpy import where
        from numpy import sqrt
        q = current_data.q
        h = q[0,:]
        hu = q[1,:]
        hv = q[2,:]
        ss = where(h>0, (hu**2 + hv**2)/h**2, 0.)
        s = sqrt(ss)
        return s
    plotitem.plot_var = variables
    plotitem.plotstyle = 'b-'
    plotaxes.xlimits = [-60., 1800]
    # plt.hold(True)
    # plotitem.plot_var = gaugetopo
    # plotitem.plotstyle = 'g-'
    # plt.hold(True)
    # plotaxes.afteraxes = add_zeroline
    # plt.hold(True)
    print 'Completed speed, s, plot.'
    
    # Plot momentum flux, hss
    plotfigure = plotdata.new_plotfigure(name='Momentum Flux',figno=304,type='each_gauge')
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'hss, Momentum Flux'
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    def variables(current_data):
        from numpy import where, sqrt
        q = current_data.q
        h = q[0,:]
        hu = q[1,:]
        hv = q[2,:]
        hss = where(h>0, (hu**2 + hv**2)/h, 0.)
        return hss
    plotitem.plot_var = variables
    plotitem.plotstyle = 'b-'
    plotaxes.xlimits = [-60., 1800]
    # plt.hold(True)
    # plotitem.plot_var = gaugetopo
    # plotitem.plotstyle = 'g-'
    # plt.hold(True)
    # plotaxes.afteraxes = add_zeroline
    # plt.hold(True)
    print 'Completed momentum flux, hss, plot.'
           
    #-----------------------------------------
    # Other Figures for this Site -- fgmax values, gauge stack plots, site summaries, etc.
    #-----------------------------------------
    for gridno in gridnos:
        print '*** Look for files starting with ', FGsite[gridno]
        print '*** in otherdir =',otherdir
        for filename in os.listdir(otherdir):
            if filename.startswith(FGsite[gridno],3):
                print '\nfilename=',filename
                path='_other_figures'+'/'+filename
                otherfigure = plotdata.new_otherfigure(name=filename,fname=path)
                print 'Added other figure: ',path
    #---------------------------------------------------------------

    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via clawpack.visclaw.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    # plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_framenos = 'all'         # list of frames to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.print_gaugenos = 'all'          # list of gauges to print
    # plotdata.print_gaugenos = [1,2,3,4,5]          # list of gauges to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?

    return plotdata