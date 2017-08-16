import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.ma as ma
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects

# Additional colors for plotting.  See selection, e.g., at
#  http://www.discoveryplayground.com/computer-programming-for-kids/rgb-colors/
DeepPink  = '#ff1493'
LightPink = '#ffb6c1'
DO = '#BF3EFF' # DarkOrchid1
BY = '#fcffc9' # BrightYellow
DSG = '#2f4f4f' # Dark Slate Gray
SG = '#708090' # Slate Gray	112-138-144	
LSG = '#778899' # Light Slate Gray
Gry = '#bebebe'  # Gray
LG = '#d3d3d3'  # Light Gray
DG = '#696969'  # Dim Gray
PG = '#98fb98'  # Pale Green
SpGn = '#00ff7f'  # Spring Green
LwnGn = '#7cfc00' # Lawn Green
Char = '#7fff00' # Chartreuse
MSG = '#00fa9a' # Medium Spring Green
GY = '#adff2f' # Green Yellow  
SBr = '#8b4513' #Saddle Brown
Char = '#7fff00' # Chartreuse
Bl = '#0000ff' # Blue
DoBl = '#1e90ff' # Dodger Blue
DSBl = '#00bfff' # Deep Sky Blue
#===========================================================================================
'''
Make an FGmax plot
I => Image, i.e., the final plot product space
F => Figure, i.e., the figure space that contains the final plot

Iscale = Ih/Iw = [(lat1-lat0)/(lon1-lon0)]/cos(lat_avg)
SET:  Iw = width in inches of the Image longitude axis
SET:  Fpad = width in inches of the padding around the Image in the Figure space
Then
  Ih = Iw*Iscale = width in inches of the height of Image latitude axis
  Ix0 = Fpad
  Iy0 = Fpad
  (Ix0,Iy0) = Figure coordinates of image origin = (lon0,lat0)
If we require that
  Fscale = Fh/Fw = Iscale
  Fw = Iw + 2.0*Fpad
then
  Fh = Iscale*(Iw + 2.0*Fpad)
'''

def plot_v(nvar,v_name,vunits,x,y,v,vclines_in,vcmap_name,minmax,on_map,topo,sea_level,\
            GEmap,GEextent,Site,plotdir):

# Input Parameters
    lon0=np.amin(x)
    lon1=np.amax(x)
    lat0=np.amin(y)
    lat1=np.amax(y)
    lat_avg=np.mean(y)
    print '\n============ PLOT site, nvar, variable, GEmap:', Site,nvar,v_name,on_map
    print 'Image Extent (degrees):  lon0,lon1,lat0,lat1=',lon0,lon1,lat0,lat1
    print 'Average Latitude =',lat_avg
    
# Specify the contour and color bar levels
    print '\nSpecify the contour and color bar levels'
    print 'vclines_in=',vclines_in
    vclines = []
    vlabels = []
    vcolors = []
    if minmax:
        vmin=round(np.amin(v),2)
        vmax=round(np.amax(v),2)
        print 'vmin,vmax=',vmin,vmax
        vclines=ma.compressed(ma.masked_outside(vclines_in,vmin,vmax))
        nvclines=len(vclines)
        if nvclines < 2:
            dv = (vmax-vmin)/4.
            vclines = np.arange(vmin,vmax+dv,dv)
        if vclines[nvclines-1] < vmax:
            vclines=np.append(vclines,vmax)
        if vmin < vclines[0]:
            vclines=np.append(vmin,vclines)
        vclines=sorted(vclines)
        nvclines=len(vclines)
        for iv in vclines:
            vlabels.append(str(iv))
            vcolors.append('w')
        vcolors[0]='g'
        vlabels[0]=''
        vlabels[nvclines-1]=''
    else:
        vclines = vclines_in
        nvclines=len(vclines)
        for iv in vclines:
            vlabels.append(str(iv))
            vcolors.append('w')
    print 'Contour and Color Bar v levels:'
    print ' vclines=',vclines
    print ' vlabels=',vlabels
    print ' vcolors=',vcolors

# Create the plot
    # print '\n** Create the plot'

# Set the plot parameters (inches)
    Fdpi=600.
    Fpad=1.0 # inches
    # print 'Figure dpi & padding (inches): Fdpi,Fpad=',Fdpi,Fpad
    Iscale=((lat1-lat0)/(lon1-lon0))/np.cos(lat_avg*np.pi/180.)
    # print 'Image scale: Iscale=',Iscale

    # print '\nSet the plot parameters (inches).'
    Iw = 10.
    Fpad = 1. # Padding around the plot
    Ih = Iw*Iscale
    Ix0 = Fpad
    Iy0 = Fpad
    # print 'Fpad,Ix0,Iy0,Iw,Ih=',Fpad,Ix0,Iy0,Iw,Ih

# Set the figure parameters (inches)
    # print '\n** Set the figure parameters (inches)'
    Fw = Iw+2.0*Fpad
    Fh = Iscale*(Iw+2.0*Fpad)
    # print 'Fw,Fh=',Fw,Fh

# Create the figure space
    plt.clf()
    fig = plt.figure(figsize=(Fw,Fh),dpi=Fdpi)

# Add the image axes
    # print '\n** Add the image axes'
    nIx0=Ix0/Fw; nIy0=Iy0/Fh; nIw=Iw/Fw; nIh=Ih/Fh # Normalize
    # print 'Normalized Image parameters:  nIx0,nIy0,nIw,nIh=',nIx0,nIy0,nIw,nIh
    Iaxes = fig.add_axes([nIx0,nIy0,nIw,nIh])

# Plot the variable

# Form a plot file name that assumes no Google Earth image will be used
    print 'nvar, Site, v_name =', nvar, Site, v_name
    title = str(nvar)+'_'+Site+'_'+str(v_name)
    print 'title = ', title
    
    # title = Site+'_'+v_name

# If on_map = True, then use a Google Earth image as background

    if on_map:
        title = title+'_map' # Update the plot file name to reflect usage of a GE image
        # Iaxes.imshow(GEmap,extent=GEextent,aspect=1.0/Iscale,zorder=0) # Plot on a GE map image
        Iaxes.imshow(GEmap,extent=GEextent,aspect=Iscale,zorder=0) # Plot on a GE map image
    # If on_map = False, then plot the topography as background
    else:
        cell_is_onshore = (topo >= sea_level)   # Cell is onshore
        cell_is_offshore = (topo < sea_level)   # Cell is offshore
        topo_onshore  = ma.masked_where(cell_is_offshore, topo)  # Mask all offshore cells
        topo_offshore = ma.masked_where(cell_is_onshore, topo) # Mask all onshore cells
        # Specify the topo line characteristics
        land_clines = range(0,205,5)
        # water_clines = range(-500,-60,50)
        water_clines = [-500,-450,-400,-350,-300,-250,-200,-150,-100,-50,-10]
        topo_clines = [-10.,-5.,0.,5.,10.]
        topo_colors = ['k','k','k','k','k']
        Iaxes.contourf(x,y,topo_onshore, zorder=0,levels=land_clines,cmap=get_colormap('land_cmap'))
        # Iaxes.contour(x,y,topo_offshore,colors=topo_colors,zorder=0,levels=water_clines)
        # if not v_name == 'max_speed':
            # Iaxes.contourf(x,y,topo_offshore,zorder=0,levels=topo_clines,cmap=get_colormap('water_cmap'))
    # Plot the variable, v
    vcmap=get_colormap(vcmap_name)
    im=Iaxes.contourf(x,y,v,cmap=vcmap,levels=vclines,alpha=0.625,zorder=1) # Plot the filled levels
    cont_v=Iaxes.contour(x,y,v,colors=vcolors,levels=vclines,linewidth=0.1,zorder=2) # Contour the filled levels
    plt.ticklabel_format(style='plain',useOffset=False)
    plt.xticks(fontsize=0,rotation=20.)
    plt.yticks(fontsize=0)
    plt.tick_params(axis='both',length=2.,direction='both')
    plt.text(lon0,lat1,title,fontsize=20,fontweight='bold',\
      horizontalalignment='left',verticalalignment='top', \
      bbox=dict(facecolor='white', alpha=1.0))  # Title
    # Plot and label the topography
    topo_lw = [.5,1.,4.,1.,.5]
    topo_clines = [-10.,-5.,0.,5.,10.]
    topo_colors = ['k','k','k','k','k']
    cont_topo=Iaxes.contour(x,y,topo,levels=topo_clines,linewidths=topo_lw,\
      colors=topo_colors,zorder=3) 
    labels=[]
    for lbl in topo_clines:
        labels.append(str(lbl))
        # print 'lbl,labels=',lbl,labels
    for i in range(len(labels)):
      cont_topo.collections[i].set_label(labels[i])
      # leg=plt.legend(loc='upper right',title='Topography [m]')
      leg=plt.legend(loc='lower left',title='Topography [m]')
      plt.setp(leg.get_title(),fontweight='bold')

    # Define the colorbar axes (inches) and plot the colorbar
    # print '\n** Define the colorbar axes (inches) and plot the colorbar'
    # NOTE:  If this is a map plot, do NOT add a color bar (color bar needs fixing)
    if not on_map:
        Cpad=0.0
        Cx0=Fpad+Iw+Cpad; Cy0=Iy0; Cw=0.5; Ch=Ih
        nCx0=Cx0/Fw; nCy0=Cy0/Fh; nCw=Cw/Fw; nCh=Ch/Fh # Normalize
        # print 'Inches: Cx0,Cy0,Cw,Ch=',Cx0,Cy0,Cw,Ch
        # print 'Normalized: nCx0,nCy0,nCw,nCh=',nCx0,nCy0,nCw,nCh

        cbaxes = fig.add_axes([nCx0,nCy0,nCw,nCh])
        cbar = plt.colorbar(im,cax=cbaxes,ticks=vclines,format='%.2f',drawedges=True,spacing='proportional')
        cbar.dividers.set_color('white')
        cbar.dividers.set_linewidth(2)
        cbar.set_label(v_name+' '+vunits,fontsize=16,fontweight='bold')
        # cbar.outline.set_color('white')
        # cbar.outline.set_linewidth(2)
        cbaxes.set_yticklabels(vlabels)
        cbaxes.tick_params(labelsize=20)
        if minmax:
            cbaxes.text(0.5,0.0,str(vmin),ha='center',va='top',style='italic',weight='bold',fontsize=16)
            cbaxes.text(0.5,1.0,str(vmax),ha='center',va='bottom',style='italic',weight='bold',fontsize=16)
    
    # Save the figure
    path_and_name = plotdir+'/'+title+'.png'
    plt.savefig(path_and_name,dpi=600,bbox_inches='tight',pad=.05)
    print '**Saved figure:',title
#===========================================================================================
'''
To try different colorbars, modify the line
      rvb = make_colormap(...)
below, and execute the script, i.e.,
      python makeColorMap.py
'''
def get_colormap(cmap):
    c = mcolors.ColorConverter().to_rgb

    if cmap == 'land_cmap':
        clrs = ['honeydew','palegreen','limegreen','seagreen','darkgreen','saddlebrown']
        # clrs = ['palegreen','darkgreen']
    if cmap == 'water_cmap':
        # clrs = ['lightblue','cornflowerblue','blue','darkblue','midnightblue']
        # clrs = ['midnightblue','blue','cornflowerblue','lightskyblue','lightblue','aliceblue','white']
        clrs = ['midnightblue','blue','lightcyan'] 
    if cmap == 'FG_cmap':
        clrs = ['white','lightyellow','yellow','gold','darkorange','red','deeppink','magenta']
        # clrs = ['yellow','red','magenta']
    if cmap == 'FG_cmap_r':
        clrs = ['white','lightyellow','yellow','gold','darkorange','red','deeppink','magenta']
        clrs.reverse()

    # print 'clrs=',clrs
    nclrs = len(clrs)
    Dcend = 1./(nclrs-1.)
    cend = np.arange(Dcend,1.+Dcend,Dcend)
    # print 'nclrs,Dcend=',nclrs,Dcend
    # print 'cend=',cend
    cv=[]
    for nclr in range(nclrs-1):
        cv.append(c(clrs[nclr]))
        cv.append(c(clrs[nclr+1]))
        cv.append(cend[nclr])
    cv.append(c(clrs[nclrs-1]))
    # print 'cv=',cv
    rvb = make_colormap(cv)
    return rvb
#===========================================================================================
'''From website
      http://stackoverflow.com/questions/16834861/
         create-own-colormap-using-matplotlib-and-plot-color-scale
    Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
'''
def make_colormap(seq):
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
#===========================================================================================
def clabels2(clabels):
    for cl in clabels:
        c0=str(cl)
        # print 'c0=',c0
        ca=c0.strip('"Text(')
        cb=ca.strip(')')
        cc=cb.split(',')
        cc0=float(cc[0])
        cc1=float(cc[1])
        cc2=cc[2]
        cc2=cc2.strip("'")
        cc2=float(cc2)
        # print 'cc2=',cc2
        bbox_props = dict(boxstyle="square,pad=0.0", fc="w", ec="k", lw=6)
        contlbl=plt.text(cc0,cc1,str(cc2),fontsize=4,weight='bold',color='k',backgroundcolor='w',\
           ha='center',va='center',bbox=bbox_props,zorder=10)
#===========================================================================================
def addstations(gridno):
    '''
    To plot Assembly Area locations on pcolor or contour plot, use this as
    # an afteraxis function:
    '''
    # Plot Assembly Areas and Critical Facilities Locations
    filename = WAcoast + SitesDir + '/_GaugeStatements/plot_CF_gauges'+str(gridno)+'.py'
    execfile(filename)
    print 'Plotted Stations:',filename
    
    # Plot Coastal Gauge Locations
    filename = WAcoast + SitesDir + '/_GaugeStatements/plot_gauges'+str(gridno)+'.py'
    execfile(filename)
    print 'Plotted Stations:',filename
    print ''
