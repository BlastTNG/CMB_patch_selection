# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:11:08 2017

@author: nlourie

@author: modified by V. Fanfani
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import astropy as ast
import astropy.coordinates as ac
import matplotlib.patheffects as PathEffects
from matplotlib import path


# SOME FUNCTIONS

def ra_dec_box(x1,x2,y1,y2,c):
    """
    This function makes a box along lines of constant RA & DEC
    connecting the 4 pts defined by x1,x2,y1,y2. Makes lines of 
    color "c". Plots in celestial coords
    """
    N=200    
    x=np.append(np.linspace(x1,x2,N),np.linspace(x2,x2,N))
    y=np.append(np.linspace(y1,y1,N),np.linspace(y1,y2,N))
    hp.projplot(x,y,'-',color=c,linewidth=2,lonlat='true')
    x=np.append(np.linspace(x2,x1,N),np.linspace(x1,x1,N))
    y=np.append(np.linspace(y2,y2,N),np.linspace(y2,y1,N))
    hp.projplot(x,y,'-',color=c,linewidth=2,lonlat='true')
    return

def box_around(x0,y0,w):
    """
    This function just outputs an array of (x,y) columns describing the four
    corners of a box  centered on x0,y0 with a half-width of w.
    """
    x = np.array([x0-w,x0-w,x0+w,x0+w])
    y = np.array([y0-w,y0+w,y0+w,y0-w])
    corners=np.column_stack((x,y))
    return corners
    
def connect_the_dots(x,y,c):
    for i in range(0,np.size(x)-1):    
        xa=x[i]
        xb=x[i+1]
        ya=y[i]
        yb=y[i+1]
        N=100
        
        lx=np.linspace(xa,xb,N)
        ly=(lx-xa)*((yb-ya)/(xb-xa))+ya
        hp.projplot(lx,ly,'-',color=c,linewidth=2,coord='C',lonlat=True)  
        
def path_mask(hp_map,nside,verts):
    """
    This function takes in a healpix map and then masks out everything
    that is not contained within a path defined by a set of vertices.
    INPUTS:
        hp_map: a healpix map
        nside:  the NSIDE of the healpix map
        verts:  a list of points given as a column of RA's and a column of DEC's (x,y)
                which define the region of interest
    OUTPUT:
        m:  the healpix map with everything masked out except for the region
    """
    r = hp.Rotator(coord=['G','C'])  # Transforms celestial to galactic coordinates
    verts[:,0],verts[:,1] = r(verts[:,0],verts[:,1],lonlat=False,inv=True)  # Apply the conversion
    hp.mollview(hp_map,coord=['G','C'])
    connect_the_dots(verts[:,0],verts[:,1],'pink')
    m=hp.ma(hp_map) #returns map as a masked array
    mask=hp_map*0+1 #initializes mask as all ones
    
    lon,lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)),lonlat=True) #gets RA/DEC of all pts in the map 
    pts=np.column_stack((lon,lat))
    p=path.Path(verts)  #generates a path from the input vertices  
    mask[p.contains_points(pts)]=0 #sets the value of all points within the boundary to zero
    m.mask=mask #uses the mask to mask out the map
    #unmasked_values=m.compressed()
    #hp.mollview(m.mask)
    #hp.mollview(m)
    return m    

    
def sensitivity_beam_pix(Nf,Am,SA,tmap_h):
    """
    INPUTS:
        Nf:     noise (in MJy/sr/sqrt(Hz)) including all detectors at that freq band
        Am:     map area in deg^2
        SA:     beam solid angle in deg^2
        tmap_h:   total scanning time in hours
    OUTPUTS:
        Sb:   sensitivity in a beam-size pixel based on Joy's Memo (JOY CALLS THIS Spix)
    """
    tmap_s=tmap_h*3600
    Sb = Nf * np.sqrt(Am/SA) * 1/(np.sqrt(tmap_s))
    return Sb

def sensitivity_smooth_pix(Sb,SA,Ap):
    """
    INPUTS:
        Sb:   sensitivity in a beam-size pixel based on Joy's Memo
        SA:     beam solid angle
        Ap:     area of the smoothed pixel in square deg
    OUTPUTS:
        Spix:   sensitivity per smoothed pixel (JOY CALLS THIS Spix')
    """
    Spix=Sb * np.sqrt(SA/Ap)
    return Spix
  
######################################################################################
"""
#Import all the nside = 512 maps from PySM
nside=512
f500='./PySM_public-master/Output/512_MJySr_cmb_freef_spinn_synch_therm_600p0_512.fits'
f350='./PySM_public-master/Output/512_MJySr_cmb_freef_spinn_synch_therm_857p0_512.fits'
f250='./PySM_public-master/Output/512_MJySr_cmb_freef_spinn_synch_therm_1199p0_512.fits'
"""

nside=2048  ## THE RESOLUTION OF THE MAP (the same of fits files from PySM)

npix= hp.nside2npix(nside)
#print('The number of total pixels on the map is:')
#print(npix)  #50331648
pixel_indices = np.arange(npix) #numpy array of integer pixel indices in RING ordering

#Import all the maps from PySM
f500='./PySM_public-master/Output/Output/2048_MJySr_cmb_freef_spinn_synch_therm_600p0_2048.fits'
f350='./PySM_public-master/Output/Output/2048_MJySr_cmb_freef_spinn_synch_therm_857p0_2048.fits'
f250='./PySM_public-master/Output/Output/2048_MJySr_cmb_freef_spinn_synch_therm_1199p0_2048.fits'

#----------------------------------------------------------------------------------
#MAKE SURE THAT ALL PLOTS WITH THE AFMHOT CMAP HAVE A WHITE BACKGROUND
cmap = plt.cm.afmhot
cmap.set_under('w')

#----------------------------------------------------------------------------------
##Import the datasets:
sim500_I,sim500_Q,sim500_U=hp.read_map(f500,0,verbose=False),hp.read_map(f500,1,verbose=False),hp.read_map(f500,2,verbose=False)
sim350_I,sim350_Q,sim350_U=hp.read_map(f350,0,verbose=False),hp.read_map(f350,1,verbose=False),hp.read_map(f350,2,verbose=False)
sim250_I,sim250_Q,sim250_U=hp.read_map(f250,0,verbose=False),hp.read_map(f250,1,verbose=False),hp.read_map(f250,2,verbose=False)

sim_I = [sim250_I,sim350_I,sim500_I]
sim_Q = [sim250_Q,sim350_Q,sim500_Q]
sim_U = [sim250_U,sim350_U,sim500_U]
#----------------------------------------------------------------------------------
#Calculate the Polarized Power
sim250_Pol=np.sqrt(sim250_Q**2+sim250_U**2)
sim350_Pol=np.sqrt(sim350_Q**2+sim350_U**2)
sim500_Pol=np.sqrt(sim500_Q**2+sim500_U**2)

sim_Pol = [sim250_Pol,sim350_Pol,sim500_Pol]
#----------------------------------------------------------------------------------
#Calculate the Polarization Fraction
sim250_PF = sim250_Pol/sim250_I
sim350_PF = sim350_Pol/sim350_I
sim500_PF = sim500_Pol/sim500_I

sim_PF = [sim250_PF,sim350_PF,sim500_PF]

channel = ['250','350','500']

#Calculate the Sensitivity Per Map Pixel Using Joy's 11/17/16 Memo

#Noise from Ian's Calculator:
Nf250=0.3856 #Total noise in 250 band in MJy/sr/sqrt(Hz)
Nf350=0.2873 #Total noise in 350 band in MJy/sr/sqrt(Hz)
Nf500=0.1547 #Total noise in 500 band in MJy/sr/sqrt(Hz)
SA250=5.45e-5
SA350=1.07e-4
SA500=2.18e-4

#Calculate the sensitivity in a beam-sized pixel for a Given Map Size and Obs Time

patch_size = 2  ## THE PATCH SIZE

Amap=patch_size*patch_size #Map size in deg^2  
tmap=96.0/2  # Observation time in hours
Sb250 = sensitivity_beam_pix(Nf250,Amap,SA250,tmap)
Sb350 = sensitivity_beam_pix(Nf350,Amap,SA350,tmap)
Sb500 = sensitivity_beam_pix(Nf500,Amap,SA500,tmap)

#Choose a pixel size to smooth your map to
#pix_size_as = 1.0 #choose pixel size in arcseconds
#Ap = pix_size_as*(1.0/60)**2 #this is the map pixel size in deg^2
#print("Pixel Size = ",np.sqrt(hp.nside2pixarea(nside,degrees=True)*3600 )," arcmin")   #1.7177432059087028 arcmin
#resolution=hp.nside2resol(nside, arcmin=True)  #1.7177432059087028 arcmin
Ap=hp.nside2pixarea(nside,degrees=True) #Pixel size in deg^2 for healpix map with nside
#print("Pixel Area = ", Ap )," square degrees")   #0.000819622700402  square degrees
#pixel_area = hp.nside2pixarea(nside, degrees=True)  #0.000819622700402  square degrees
 
 
number_of_pixels_in_a_patch = Amap/Ap
#print('The average number of pixels in a patch is:')
#print(number_of_pixels_in_a_patch)  # If patch 2x2= 4880.29430864


#Calculate the sensitivity in a smoothed pixel
Spix250 = sensitivity_smooth_pix(Sb250,SA250,Ap)
Spix350 = sensitivity_smooth_pix(Sb350,SA350,Ap)
Spix500 = sensitivity_smooth_pix(Sb500,SA500,Ap)
#print("\n \nSensitivities Per Pixel in MJY/sr:")
#print("\tSpix250 = ",Spix250)  #If patch 2x2= 0.06480196313212841#
#print("\tSpix350 = ",Spix350)  #If patch 2x2= 0.048282168070177635
#print("\tSpix500 = ",Spix500)  #If patch 2x2= 0.025998090499326422

Spix = [Spix250,Spix350,Spix500]

#----------------------------------------------------------------------------------
#Calculate the SNR refering to the Polarized Power
sim250_SNR = sim250_Pol/Spix250
sim350_SNR = sim350_Pol/Spix350
sim500_SNR = sim500_Pol/Spix500

sim_SNR = [sim250_SNR,sim350_SNR,sim500_SNR]

#----------------------------------------------------------------------------------


#10h vis for 22dec2018 
x10=np.array([910,776,1050,2566,2690,2421,910])*(360.0/4000)
y10=np.array([800,644,339,345,655,794,800])*(180.0/2000)-90
#vis10_22dec2018 = np.column_stack((x10,y10))
#1h vis for 22dec2018 
#x1=np.array([755,569,771,2757,2850,2757,755])*(360.0/4000)
#y1=np.array([887,382,257,257,335,897,887])*(180.0/2000)-90
#20h vis for 22dec2018 
x20=np.array([956,946,1003,2500,2453,956])*(360.0/4000)
y20=np.array([644,567,510,521,644,644])*(180.0/2000)-90


"""
#visibility_pixel_index= hp.ang2pix(nside, x10, y10, nest=False, lonlat=True)
#print(visibility_pixel_index)  #[32937800 38516277 46850872 46730479 38151557 33170270 32937800]

#Test for the 10h visibility area 
coord_to_vector_1= hp.ang2vec(910*(360.0/4000), 800*(180.0/2000)-90, lonlat=True)  #shape (3,)
coord_to_vector_2= hp.ang2vec(910*(360.0/4000), 339*(180.0/2000)-90, lonlat=True)  
coord_to_vector_3= hp.ang2vec(2690*(360.0/4000), 339*(180.0/2000)-90, lonlat=True)  
coord_to_vector_4= hp.ang2vec(2690*(360.0/4000), 800*(180.0/2000)-90, lonlat=True)
"""

#Patch choose to view the quality factor effect on each patch  #FTF1 (one of these with GOOD QUALITY PARAMETER VALUES (both higher than a certain threshold established below) for 250 e 350 channels)
if patch_size == 2:
	ra0_effect=  180 + (patch_size*1.0/2)
	dec0_effect= -49 

elif patch_size == 5:
	ra0_effect=  180 + (patch_size*1.0/2)
	dec0_effect= -50.5

w= patch_size*1.0/2 #box width


#selection of a sky area to see the effect of the different patch selection  #FTF2
ra0_comparison=  232.5
dec0_comparison= -40.5
w_comparison= 2.5 #box width



for v in range(0,3):

	#Map the Polarization Fraction in Cartesian Proj WITHOUT the partition (ORIGINAL RESOLUTION)
	hp.cartview(sim_PF[v],title=channel[v]+'$\mu$m Polarization Fraction',rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=0.2,cbar=True,lonra=[-112,65],latra=[-63,-16])
	connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
	connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
	#ra_dec_box(ra0_effect-w,ra0_effect+w,dec0_effect-w,dec0_effect+w,'white') #plot FTF1
	#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
	#Add some graticules            
	hp.graticule(30,30,coord='C',color='black')
	#Mark the RA
	for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    		hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
	#Mark the DEC
	for DEC in [30,60,-30,-60]:
    		hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
	plt.savefig('1_'+channel[v]+'_Polarization_Fraction_cartesian_proj_NSIDE='+str(nside)+'.png',format='png')

	#Map the Polarized SNR in Cartesian Proj WITHOUT the partition (ORIGINAL RESOLUTION)
	hp.cartview(sim_SNR[v],title=channel[v]+' $\mu$m Polarized SNR for '+str(tmap)+'h on '+str(patch_size)+'x'+str(patch_size)+' Patch',rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,lonra=[-112,65],latra=[-63,-16])
	connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
	connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
	#ra_dec_box(ra0_effect-w,ra0_effect+w,dec0_effect-w,dec0_effect+w,'white') #plot FTF1
	#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
	#Add some graticules            
	hp.graticule(30,30,coord='C',color='black')
	#Mark the RA
	for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    		hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
	#Mark the DEC
	for DEC in [30,60,-30,-60]:
    		hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
	plt.savefig('6_'+channel[v]+'_Polarized_SNR_cartesian_proj_patch_'+str(patch_size)+'x'+str(patch_size)+'for '+str(tmap)+'h_NSIDE='+str(nside)+'.png',format='png')


	#ZOOM the Polarization Fraction on a patch (choose to see the quality factor effect) WITHOUT the partition (ORIGINAL RESOLUTION)
	hp.cartview(sim_PF[v],title=channel[v]+' $\mu$m FTF1 Polarization Fraction side '+str(patch_size)+'deg',lonra=[ra0_effect-w,ra0_effect+w], latra=[dec0_effect-w,dec0_effect+w],min=0,max=0.2,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
    
	fig = plt.gcf()
	ax = plt.gca()
	image = ax.get_images()[0]
	cb = fig.colorbar(image, ax=ax)
	cb.set_label('Polarization Fraction')
	cb.set_ticks(np.linspace(0,0.2,11))
	plt.savefig('4_'+channel[v]+'_PolFrac_FTF1_side_'+str(patch_size)+'deg_NSIDE'+str(nside)+'.png',format='png')


	#ZOOM the Polarization Fraction on a patch (choose to see the effect of the different patch selection) WITHOUT the partition (ORIGINAL RESOLUTION)
	hp.cartview(sim_PF[v],title=channel[v]+' $\mu$m FTF2 Polarization Fraction side 5deg',lonra=[ra0_comparison-w_comparison,ra0_comparison+w_comparison], latra=[dec0_comparison-w_comparison,dec0_comparison+w_comparison],min=0,max=0.2,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
    
	fig = plt.gcf()
	ax = plt.gca()
	image = ax.get_images()[0]
	cb = fig.colorbar(image, ax=ax)
	cb.set_label('Polarization Fraction')
	cb.set_ticks(np.linspace(0,0.2,11))
	plt.savefig('11_'+channel[v]+'_PolFrac_FTF2_side_5deg_NSIDE'+str(nside)+'.png',format='png')

	#ZOOM the Polarized SNR on a patch (choose to see the quality factor effect) WITHOUT the partition (ORIGINAL RESOLUTION)
	hp.cartview(sim_SNR[v],title=channel[v]+' $\mu$m FTF1 Polarized SNR for '+str(tmap)+'h on '+str(patch_size)+'x'+str(patch_size)+'deg Patch',lonra=[ra0_effect-w,ra0_effect+w], latra=[dec0_effect-w,dec0_effect+w],min=0,max=10,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
    
	fig = plt.gcf()
	ax = plt.gca()
	image = ax.get_images()[0]
	cb = fig.colorbar(image, ax=ax)
	cb.set_label('Polarized SNR')
	cb.set_ticks(np.linspace(0,10,11))
	plt.savefig('9_'+channel[v]+'_PolSNR_FTF1_for_'+str(tmap)+'h_on_'+str(patch_size)+'x'+str(patch_size)+'_patch_NSIDE'+str(nside)+'.png',format='png')


	#ZOOM the Polarized SNR on a patch (choose to see the effect of the different patch selection) WITHOUT the partition (ORIGINAL RESOLUTION)
	hp.cartview(sim_SNR[v],title=channel[v]+' $\mu$m FTF2 Polarized SNR for '+str(tmap)+'h on '+str(patch_size)+'x'+str(patch_size)+'deg Patch size 5deg',lonra=[ra0_comparison-w_comparison,ra0_comparison+w_comparison], latra=[dec0_comparison-w_comparison,dec0_comparison+w_comparison],min=0,max=10,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
    
	fig = plt.gcf()
	ax = plt.gca()
	image = ax.get_images()[0]
	cb = fig.colorbar(image, ax=ax)
	cb.set_label('Polarized SNR')
	cb.set_ticks(np.linspace(0,10,11))
	plt.savefig('13_'+channel[v]+'_PolSNR_FTF2_for_'+str(tmap)+'h_on_'+str(patch_size)+'x'+str(patch_size)+'_patch_size_5deg_NSIDE'+str(nside)+'.png',format='png')


	#set empty list to save the equatorial coordinates of the pixels that have GOOD QUALITY PARAMETER VALUES (both higher than a certain threshold established below)
	ra_good = []
	dec_good = []

	#for loop that generates SQUARE PATCHES of size= patch_size within the partitioned area, that is a rectangular area which corresponds to the extended rectangle 10 h visibility area of BLAST (partitioned area's equatorial coordinates (deg): long=[70,243], lat=[-18,-61])

	for i in range(70,243,patch_size):  #longitude range of the 10 h visibility area with the step given by the pixel size
		for j in range(18,61,patch_size):  #latitude range of the 10 h visibility area with the step given by the pixel size 

			coord_to_vector_1= hp.ang2vec(i, -j, lonlat=True)  #shape (3,)
			coord_to_vector_2= hp.ang2vec(i+patch_size, -j, lonlat=True)  
			coord_to_vector_3= hp.ang2vec(i+patch_size, -(j+patch_size), lonlat=True)  
			coord_to_vector_4= hp.ang2vec(i, -(j+patch_size), lonlat=True) 

			r = hp.Rotator(coord=['G','C'],inv=True) # Transforms celestial(equatorial) to galactic coordinates

			coord_to_vector_1r = r(coord_to_vector_1)
			coord_to_vector_2r = r(coord_to_vector_2)
			coord_to_vector_3r = r(coord_to_vector_3)
			coord_to_vector_4r = r(coord_to_vector_4)

			#array containing the vertices of the polygon= the square that defines the patch [shape (N, 3)]
			vec=np.array([coord_to_vector_1r,coord_to_vector_2r,coord_to_vector_3r,coord_to_vector_4r])

			#calculates the pixel indexes inside a patch (=convex polygon generated from the vertices), without the pixels that overlap with the polygon
			ipix_patch = hp.query_polygon(nside, vertices=vec, inclusive=False)

			#calculation of the values ​​to be assigned to the patches
			mean_Q = sim_Q[v][ipix_patch].mean()
			mean_U = sim_U[v][ipix_patch].mean()
			pol_contrast = np.sqrt((sim_Q[v][ipix_patch]-mean_Q)**2 + (sim_U[v][ipix_patch]-mean_U)**2) # contrast polarization

			#QUALITY PARAMETERS definitions 
			sim_PF[v][ipix_patch] = (sim_Pol[v][ipix_patch]/sim_I[v][ipix_patch]).mean() # FACTOR G for the POLARIZATION FRACTION 
			sim_SNR[v][ipix_patch] = (pol_contrast/Spix[v]).mean()  # FACTOR G for the POLARIZED CONTRAST SNR
			# in this point the factors fill the whole patch

			#sim_SNR[v][ipix_patch] = pol_contrast/Spix[v]

			#set the threshold to obtein only good quality patches (return the equatorial coordinates)
			if (sim_SNR[v][ipix_patch].mean() >= 4.0) and (sim_PF[v][ipix_patch].mean() >= 0.08):

				ra_good.append(i+w)
				dec_good.append(-(j+w))

	ra_good = np.array(ra_good)			
	dec_good = np.array(dec_good)


	#Map the Polarization Fraction in Mollewiede Proj
	"""
	hp.mollview(sim_PF[v],coord=['G','C'],cmap='seismic',title=channel[v]+' $\mu$m Polarization Fraction Quality Factor',unit='log10(MJy/Sr)',min=0,max=0.2)
	connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
	connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
	#ra_dec_box(ra0_effect-w,ra0_effect+w,dec0_effect-w,dec0_effect+w,'green') #plot FTF1
	#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
	#Add some graticules            
	hp.graticule(30,30,coord='C',color='black')
	#Mark the RA
	for RA in [0,30,60,90,120,150,210,240,270,300,330]:
       		hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
	#Mark the DEC
	for DEC in [30,60,-30,-60]:
 		hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
	plt.savefig(channel[v]+'_Polarization_Fraction_Quality_Factor_NSIDE='+str(nside)+'.png',format='png')
	"""
	#Map the Polarized Contrast SNR in Mollewiede Proj
	"""
	hp.mollview(sim_SNR[v],coord=['G','C'],cmap='seismic',title=channel[v]+' $\mu$m Polarized Contrast SNR Quality Factor',min=0,max=10,cbar=True)
	connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
	connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
	#ra_dec_box(ra0_effect-w,ra0_effect+w,dec0_effect-w,dec0_effect+w,'white') #plot FTF1
	#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
	#Add some graticules            
	hp.graticule(30,30,coord='C',color='black')
	#Mark the RA
	for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    		hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
	#Mark the DEC
	for DEC in [30,60,-30,-60]:
    		hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
	plt.savefig(channel[v]+'_Polarized_Contrast_SNR_Quality_Factor_NSIDE='+str(nside)+'.png',format='png')
	"""

	#Map the Polarization Fraction in Cartesian Proj, without the good patches highlighted
	hp.cartview(sim_PF[v],title=channel[v]+' $\mu$m Polarization Fraction Quality Factor '+str(patch_size)+'x'+str(patch_size)+' patch',rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=0.2,cbar=True,lonra=[-112,65],latra=[-63,-16])
	connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
	connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
	#Add some graticules            
	hp.graticule(30,30,coord='C',color='black')
	#Mark the RA
	for RA in [0,30,60,90,120,150,210,240,270,300,330]:
  	  hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
	#Mark the DEC
	for DEC in [30,60,-30,-60]:
   		hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
	plt.savefig('2_'+channel[v]+'_Polarization_Fraction_Quality_Factor_cartesian_proj_patch_'+str(patch_size)+'x'+str(patch_size)+'_NSIDE='+str(nside)+'.png',format='png')

	#Map the Polarization Fraction in Cartesian Proj, with the good patches highlighted
	hp.cartview(sim_PF[v],title=channel[v]+' $\mu$m Polarization Fraction Quality Factor '+str(patch_size)+'x'+str(patch_size)+' patch with good patches',rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=0.2,cbar=True,lonra=[-112,65],latra=[-63,-16])
	connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
	connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
	for t in range(0, len(ra_good)-1):
		ra_dec_box(ra_good[t]-w,ra_good[t]+w,dec_good[t]-w,dec_good[t] + w,'black')
	#Add some graticules            
	hp.graticule(30,30,coord='C',color='black')
	#Mark the RA
	for RA in [0,30,60,90,120,150,210,240,270,300,330]:
  	  hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
	#Mark the DEC
	for DEC in [30,60,-30,-60]:
   		hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
	plt.savefig('3_'+channel[v]+'_Polarization_Fraction_Quality_Factor_cartesian_proj_patch_'+str(patch_size)+'x'+str(patch_size)+'_with_good_patches_NSIDE='+str(nside)+'.png',format='png')

	#Map the Polarized Contrast SNR in Cartesian Proj, without the good patches highlighted
	hp.cartview(sim_SNR[v],title=channel[v]+' $\mu$m Polarized Contrast SNR Quality Factor for '+str(tmap)+'h on '+str(patch_size)+'x'+str(patch_size)+' patch',rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,lonra=[-112,65],latra=[-63,-16])
	connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
	connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
	#Add some graticules            
	hp.graticule(30,30,coord='C',color='black')
	#Mark the RA
	for RA in [0,30,60,90,120,150,210,240,270,300,330]:
   		hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
	#Mark the DEC
	for DEC in [30,60,-30,-60]:
        	hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
	plt.savefig('7_'+channel[v]+'_Polarized_Contrast_SNR_Quality_Factor_cartesian_proj_patch_'+str(patch_size)+'x'+str(patch_size)+'for '+str(tmap)+'h_NSIDE='+str(nside)+'.png',format='png')

	#Map the Polarized Contrast SNR in Cartesian Proj, with the good patches highlighted
	hp.cartview(sim_SNR[v],title=channel[v]+' $\mu$m Polarized Contrast SNR Quality Factor for '+str(tmap)+'h on '+str(patch_size)+'x'+str(patch_size)+' patch with good patches',rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,lonra=[-112,65],latra=[-63,-16])
	connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
	connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
	for t in range(0, len(ra_good)-1):
		ra_dec_box(ra_good[t]-w,ra_good[t]+w,dec_good[t]-w,dec_good[t] + w,'black')
	#Add some graticules            
	hp.graticule(30,30,coord='C',color='black')
	#Mark the RA
	for RA in [0,30,60,90,120,150,210,240,270,300,330]:
   		hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
	#Mark the DEC
	for DEC in [30,60,-30,-60]:
        	hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
	plt.savefig('8_'+channel[v]+'_Polarized_Contrast_SNR_Quality_Factor_cartesian_proj_patch_'+str(patch_size)+'x'+str(patch_size)+'for '+str(tmap)+'h_with_good_patches_NSIDE='+str(nside)+'.png',format='png')

	#ZOOM the Polarization Fraction on the same patch (choose to see the quality factor effect) WITH the partition (NEW RESOLUTION)
	hp.cartview(sim_PF[v],title=channel[v]+' $\mu$m FTF1 Polarization Fraction Quality Factor '+str(patch_size)+'x'+str(patch_size)+'deg Patch',lonra=[ra0_effect-w,ra0_effect+w], latra=[dec0_effect-w,dec0_effect+w],min=0,max=0.2,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
 	fig = plt.gcf()
	ax = plt.gca()
	image = ax.get_images()[0]
	cb = fig.colorbar(image, ax=ax)
	cb.set_label('Polarization Fraction')
	cb.set_ticks(np.linspace(0,0.2,11))
	plt.savefig('5_'+channel[v]+'_PolFrac_Quality_Factor_FTF1_for_on_'+str(patch_size)+'x'+str(patch_size)+'_patch_NSIDE'+str(nside)+'.png',format='png')


	#ZOOM the Polarization Fraction on the same patch (choose to see the effect of the different patch selection) WITH the partition (NEW RESOLUTION)
	hp.cartview(sim_PF[v],title=channel[v]+' $\mu$m FTF2 Polarization Fraction Quality Factor '+str(patch_size)+'x'+str(patch_size)+'deg Patch',lonra=[ra0_comparison-w_comparison,ra0_comparison+w_comparison], latra=[dec0_comparison-w_comparison,dec0_comparison+w_comparison],min=0,max=0.2,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
 	fig = plt.gcf()
	ax = plt.gca()
	image = ax.get_images()[0]
	cb = fig.colorbar(image, ax=ax)
	cb.set_label('Polarization Fraction')
	cb.set_ticks(np.linspace(0,0.2,11))
	plt.savefig('12_'+channel[v]+'_PolFrac_Quality_Factor_FTF2_for_on_'+str(patch_size)+'x'+str(patch_size)+'_patch_NSIDE'+str(nside)+'.png',format='png')

	#ZOOM the Polarized SNR on a patch (choose to see the quality factor effect) WITH the partition (NEW RESOLUTION)  
	hp.cartview(sim_SNR[v],title=channel[v]+' $\mu$m FTF1 Polarized SNR Quality Factor for '+str(tmap)+'h on'+str(patch_size)+'x'+str(patch_size)+'deg Patch',lonra=[ra0_effect-w,ra0_effect+w], latra=[dec0_effect-w,dec0_effect+w],min=0,max=10,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
 	fig = plt.gcf()
	ax = plt.gca()
	image = ax.get_images()[0]
	cb = fig.colorbar(image, ax=ax)
	cb.set_label('Polarized SNR')
	cb.set_ticks(np.linspace(0,10,11))
	plt.savefig('10_'+channel[v]+'_PolSNR_Quality_Factor_FTF1_for_'+str(tmap)+'h_on_'+str(patch_size)+'x'+str(patch_size)+'_patch_NSIDE'+str(nside)+'.png',format='png')


	#ZOOM the Polarized SNR on a patch (choose to see the effect of the different patch selection) WITH the partition (NEW RESOLUTION)  
	hp.cartview(sim_SNR[v],title=channel[v]+' $\mu$m FTF2 Polarized SNR Quality Factor for '+str(patch_size)+'x'+str(patch_size)+'deg Patch',lonra=[ra0_comparison-w_comparison,ra0_comparison+w_comparison], latra=[dec0_comparison-w_comparison,dec0_comparison+w_comparison],min=0,max=10,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
 	fig = plt.gcf()
	ax = plt.gca()
	image = ax.get_images()[0]
	cb = fig.colorbar(image, ax=ax)
	cb.set_label('Polarized SNR')
	cb.set_ticks(np.linspace(0,10,11))
	plt.savefig('14_'+channel[v]+'_PolSNR_Quality_Factor_FTF2_for_'+str(tmap)+'h_on_'+str(patch_size)+'x'+str(patch_size)+'_patch_NSIDE'+str(nside)+'.png',format='png')







