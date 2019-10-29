# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:11:08 2017

@author: nlourie


modified by V. Fanfani & F. Nati
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
#Import all the nside = 2048 maps from PySM
nside=2048  

npix= hp.nside2npix(nside)
#print('The number of total pixels on the map is:')
#print(npix)  #50331648
pixel_indices = np.arange(npix) #numpy array of integer pixel indices in RING ordering


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
#----------------------------------------------------------------------------------
#Calculate the Polarized Power
sim250_Pol=np.sqrt(sim250_Q**2+sim250_U**2)
sim350_Pol=np.sqrt(sim350_Q**2+sim350_U**2)
sim500_Pol=np.sqrt(sim500_Q**2+sim500_U**2)
#----------------------------------------------------------------------------------
#Calculate the Polarization Fraction
sim250_PF = sim250_Pol/sim250_I
sim350_PF = sim350_Pol/sim350_I
sim500_PF = sim500_Pol/sim500_I

#Calculate the Sensitivity Per Map Pixel Using Joy's 11/17/16 Memo

#Noise from Ian's Calculator:
Nf250=0.3856 #Total noise in 250 band in MJy/sr/sqrt(Hz)
Nf350=0.2873 #Total noise in 350 band in MJy/sr/sqrt(Hz)
Nf500=0.1547 #Total noise in 500 band in MJy/sr/sqrt(Hz)
SA250=5.45e-5
SA350=1.07e-4
SA500=2.18e-4

#Calculate the sensitivity in a beam-sized pixel for a Given Map Size and Obs Time
Amap=2*2 #Map size in deg^2
tmap=96.0/2  #Observation time in hours
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
 
 
number_of_pixels_in_2x2_patch = 4/Ap
#print('The average number of pixels in a 2x2 patch is:')
#print(number_of_pixels_in_2x2_patch)  #4880.29430864


#Calculate the sensitivity in a smoothed pixel
Spix250 = sensitivity_smooth_pix(Sb250,SA250,Ap)
Spix350 = sensitivity_smooth_pix(Sb350,SA350,Ap)
Spix500 = sensitivity_smooth_pix(Sb500,SA500,Ap)
#print("\n \nSensitivities Per Pixel in MJY/sr:")
#print("\tSpix250 = ",Spix250)  #0.06480196313212841#
#print("\tSpix350 = ",Spix350)  #0.048282168070177635
#print("\tSpix500 = ",Spix500)  #0.025998090499326422


#----------------------------------------------------------------------------------
#Calculate the SNR refering to the Polarized Power
sim250_SNR = sim250_Pol/Spix250
sim350_SNR = sim350_Pol/Spix350
sim500_SNR = sim500_Pol/Spix500

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

#Patch choose for the test
ra0_1=  181
dec0_1= -53
w_1=1 #box width

"""
hp.cartview(sim500_PF,title='500 $\mu$m FTF1 Polarization Fraction for 48h on 2x2deg Patch',lonra=[ra0_1-w_1,ra0_1+w_1], latra=[dec0_1-w_1,dec0_1+w_1],min=0,max=0.2,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
    
fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cb = fig.colorbar(image, ax=ax)
cb.set_label('Polarization Fraction')
cb.set_ticks(np.linspace(0,0.2,11))
plt.savefig('YES500_tutto = '+str(nside)+'.png',format='png')
"""

"""
#Map the Polarization Fraction and the SNR refering to the Polarized Power in Cartesian Proj
hp.cartview(sim500_PF,title="500 $\mu$m Polarization Fraction",rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=0.2,cbar=True,lonra=[-112,65],latra=[-63,-16])
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('OK_500_Polarization_Fraction_cartesian_proj_NSIDE = '+str(nside)+'.png',format='png')
"""

ra_good = []
dec_good = []



#for loop that generates 2x2 patches within the extended rectangle 10 h visibility area  (area's partition)

for i in range(70,243,2):  #longitude range of the 10 h visibility area (deg)
	for j in range(18,61,2):  #latitude range of the 10 h visibility area (deg)

		coord_to_vector_1= hp.ang2vec(i, -j, lonlat=True)  #shape (3,)
		coord_to_vector_2= hp.ang2vec(i+2, -j, lonlat=True)  
		coord_to_vector_3= hp.ang2vec(i+2, -(j+2), lonlat=True)  
		coord_to_vector_4= hp.ang2vec(i, -(j+2), lonlat=True) 

		r = hp.Rotator(coord=['G','C'],inv=True) # Transforms celestial to galactic coordinates

		coord_to_vector_1r = r(coord_to_vector_1)
		coord_to_vector_2r = r(coord_to_vector_2)
		coord_to_vector_3r = r(coord_to_vector_3)
		coord_to_vector_4r = r(coord_to_vector_4)

		#array containing the vertices of the polygon, shape (N, 3)
		vec=np.array([coord_to_vector_1r,coord_to_vector_2r,coord_to_vector_3r,coord_to_vector_4r])

		#calculates the pixel indexes inside a convex polygon generated from the vertices
		ipix_patch = hp.query_polygon(nside, vertices=vec, inclusive=False)

		#calculation of the values ​​to be assigned to the patches
		mean_Q_250 = sim250_Q[ipix_patch].mean()
		mean_U_250 = sim250_U[ipix_patch].mean()
		pol_contrast_250 = np.sqrt((sim250_Q[ipix_patch]-mean_Q_250)**2 + (sim250_U[ipix_patch]-mean_U_250)**2)


		sim250_PF[ipix_patch] = (sim250_Pol[ipix_patch]/sim250_I[ipix_patch]).mean()
		sim250_SNR[ipix_patch] = (pol_contrast_250/Spix250).mean()
		#sim250_SNR[ipix_patch] = pol_contrast_250/Spix250

		mean_Q_350 = sim350_Q[ipix_patch].mean()
		mean_U_350 = sim350_U[ipix_patch].mean()
		pol_contrast_350 = np.sqrt((sim350_Q[ipix_patch]-mean_Q_350)**2 + (sim350_U[ipix_patch]-mean_U_350)**2)


		sim350_PF[ipix_patch] = (sim350_Pol[ipix_patch]/sim350_I[ipix_patch]).mean()
		sim350_SNR[ipix_patch] = (pol_contrast_350/Spix350).mean()
		#sim350_SNR[ipix_patch] = pol_contrast_350/Spix350


		mean_Q_500 = sim500_Q[ipix_patch].mean()
		mean_U_500 = sim500_U[ipix_patch].mean()
		pol_contrast_500 = np.sqrt((sim500_Q[ipix_patch]-mean_Q_500)**2 + (sim500_U[ipix_patch]-mean_U_500)**2)


		sim500_PF[ipix_patch] = (sim500_Pol[ipix_patch]/sim500_I[ipix_patch]).mean()
		sim500_SNR[ipix_patch] = (pol_contrast_500/Spix500).mean()
		#sim500_SNR[ipix_patch] = pol_contrast_500/Spix500


		#if sim250_PF[ipix_patch].all() >= 0.01 and sim250_SNR[ipix_patch].all() >= 5:



		if sim250_SNR[ipix_patch].mean() >= 5.0 and sim250_PF[ipix_patch].mean() >= 0.1:

			ra_good.append(i+1)
			dec_good.append(-(j+1))
			
ra_good = np.array(ra_good)			
dec_good = np.array(dec_good)
#print(ra_good)
#print(dec_good)		
"""
[ 77  79  79  79  85  89  91  93  93  95 103 131 131 131 133 133 133 171
 179 181 185 191 205 223 223 225 225 225 227 227 227 229 229 229 229 229
 231 231 231 233 233 235 235 235 235 237 237 237 237 237 241]
[-59 -19 -27 -29 -25 -29 -61 -23 -27 -31 -49 -25 -27 -29 -23 -27 -29 -49
 -49 -49 -51 -49 -31 -41 -47 -41 -45 -47 -37 -45 -49 -37 -39 -43 -45 -49
 -37 -43 -45 -41 -43 -31 -33 -43 -45 -33 -41 -43 -45 -59 -29]
"""
#0,1,6   47,48,49,50


#Patch choose for the test
#FTF1
ra0_1=  181
dec0_1= -53
w_1=1 #box width

#FTF2
#ra0_2=100
#dec0_2=-57.3
#w_2=1 #box width




### 250   
#Map the Polarization Fraction and the Polarized SNR in Mollewiede Proj
hp.mollview(sim250_PF,coord=['G','C'],cmap='seismic',title="250_$\mu$m Polarization Fraction",unit='log10(MJy/Sr)',min=0,max=0.2)
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'green') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('250_Polarization_Fraction_NSIDE = '+str(nside)+'.png',format='png')

hp.mollview(sim250_SNR,coord=['G','C'],cmap='seismic',title="250 $\mu$m SNR mean for 2x2deg Patch",min=0,max=10,cbar=True)
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('250_Polarized_SNR_(mean_removed_in_Q_U)_mean_NSIDE = '+str(nside)+'.png',format='png')


#Map the Polarization Fraction and the SNR refering to the Polarized Power in Cartesian Proj
hp.cartview(sim250_PF,title="250 $\mu$m Polarization Fraction",rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=0.2,cbar=True,lonra=[-112,65],latra=[-63,-16])
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra_good[0]-1,ra_good[0]+1,dec_good[0]-1,dec_good[0] + 1,'black') #plot FTF1

#ra_dec_box(ra_good[1]-1,ra_good[1]+1,dec_good[1]-1,dec_good[1] + 1,'black') #plot FTF1
ra_dec_box(ra_good[2]-1,ra_good[2]+1,dec_good[2]-1,dec_good[2] + 1,'black') #plot FTF1
ra_dec_box(ra_good[3]-1,ra_good[3]+1,dec_good[3]-1,dec_good[3] + 1,'black') #plot FTF1
ra_dec_box(ra_good[4]-1,ra_good[4]+1,dec_good[4]-1,dec_good[4] + 1,'black') #plot FTF1
ra_dec_box(ra_good[5]-1,ra_good[5]+1,dec_good[5]-1,dec_good[5] + 1,'black') #plot FTF1
#ra_dec_box(ra_good[6]-1,ra_good[6]+1,dec_good[6]-1,dec_good[6] + 1,'black') #plot FTF1
ra_dec_box(ra_good[7]-1,ra_good[7]+1,dec_good[7]-1,dec_good[7] + 1,'black') #plot FTF1
ra_dec_box(ra_good[8]-1,ra_good[8]+1,dec_good[8]-1,dec_good[8] + 1,'black') #plot FTF1
ra_dec_box(ra_good[9]-1,ra_good[9]+1,dec_good[9]-1,dec_good[9] + 1,'black') #plot FTF1
ra_dec_box(ra_good[10]-1,ra_good[10]+1,dec_good[10]-1,dec_good[10] + 1,'black') #plot FTF1
ra_dec_box(ra_good[11]-1,ra_good[11]+1,dec_good[11]-1,dec_good[11] + 1,'black') #plot FTF1
ra_dec_box(ra_good[12]-1,ra_good[12]+1,dec_good[12]-1,dec_good[12] + 1,'black') #plot FTF1
ra_dec_box(ra_good[13]-1,ra_good[13]+1,dec_good[13]-1,dec_good[13] + 1,'black') #plot FTF1
ra_dec_box(ra_good[14]-1,ra_good[14]+1,dec_good[14]-1,dec_good[14] + 1,'black') #plot FTF1
ra_dec_box(ra_good[15]-1,ra_good[15]+1,dec_good[15]-1,dec_good[15] + 1,'black') #plot FTF1
ra_dec_box(ra_good[16]-1,ra_good[16]+1,dec_good[16]-1,dec_good[16] + 1,'black') #plot FTF1
ra_dec_box(ra_good[17]-1,ra_good[17]+1,dec_good[17]-1,dec_good[17] + 1,'black') #plot FTF1
ra_dec_box(ra_good[18]-1,ra_good[18]+1,dec_good[18]-1,dec_good[18] + 1,'black') #plot FTF1
ra_dec_box(ra_good[19]-1,ra_good[19]+1,dec_good[19]-1,dec_good[19] + 1,'black') #plot FTF1
ra_dec_box(ra_good[20]-1,ra_good[20]+1,dec_good[20]-1,dec_good[20] + 1,'black') #plot FTF1
ra_dec_box(ra_good[21]-1,ra_good[21]+1,dec_good[21]-1,dec_good[21] + 1,'black') #plot FTF1
ra_dec_box(ra_good[22]-1,ra_good[22]+1,dec_good[22]-1,dec_good[22] + 1,'black') #plot FTF1
ra_dec_box(ra_good[23]-1,ra_good[23]+1,dec_good[23]-1,dec_good[23] + 1,'black') #plot FTF1
ra_dec_box(ra_good[24]-1,ra_good[24]+1,dec_good[24]-1,dec_good[24] + 1,'black') #plot FTF1
ra_dec_box(ra_good[25]-1,ra_good[25]+1,dec_good[25]-1,dec_good[25] + 1,'black') #plot FTF1
ra_dec_box(ra_good[26]-1,ra_good[26]+1,dec_good[26]-1,dec_good[26] + 1,'black') #plot FTF1
ra_dec_box(ra_good[27]-1,ra_good[27]+1,dec_good[27]-1,dec_good[27] + 1,'black') #plot FTF1
ra_dec_box(ra_good[28]-1,ra_good[28]+1,dec_good[28]-1,dec_good[28] + 1,'black') #plot FTF1
ra_dec_box(ra_good[29]-1,ra_good[29]+1,dec_good[29]-1,dec_good[29] + 1,'black') #plot FTF1
ra_dec_box(ra_good[30]-1,ra_good[30]+1,dec_good[30]-1,dec_good[30] + 1,'black') #plot FTF1
ra_dec_box(ra_good[31]-1,ra_good[31]+1,dec_good[31]-1,dec_good[31] + 1,'black') #plot FTF1
ra_dec_box(ra_good[32]-1,ra_good[32]+1,dec_good[32]-1,dec_good[32] + 1,'black') #plot FTF1
ra_dec_box(ra_good[33]-1,ra_good[33]+1,dec_good[33]-1,dec_good[33] + 1,'black') #plot FTF1
ra_dec_box(ra_good[34]-1,ra_good[34]+1,dec_good[34]-1,dec_good[34] + 1,'black') #plot FTF1
ra_dec_box(ra_good[35]-1,ra_good[35]+1,dec_good[35]-1,dec_good[35] + 1,'black') #plot FTF1
ra_dec_box(ra_good[36]-1,ra_good[36]+1,dec_good[36]-1,dec_good[36] + 1,'black') #plot FTF1
ra_dec_box(ra_good[37]-1,ra_good[37]+1,dec_good[37]-1,dec_good[37] + 1,'black') #plot FTF1
ra_dec_box(ra_good[38]-1,ra_good[38]+1,dec_good[38]-1,dec_good[38] + 1,'black') #plot FTF1
ra_dec_box(ra_good[39]-1,ra_good[39]+1,dec_good[39]-1,dec_good[39] + 1,'black') #plot FTF1
ra_dec_box(ra_good[40]-1,ra_good[40]+1,dec_good[40]-1,dec_good[40] + 1,'black') #plot FTF1
ra_dec_box(ra_good[41]-1,ra_good[41]+1,dec_good[41]-1,dec_good[41] + 1,'black') #plot FTF1
ra_dec_box(ra_good[42]-1,ra_good[42]+1,dec_good[42]-1,dec_good[42] + 1,'black') #plot FTF1
ra_dec_box(ra_good[43]-1,ra_good[43]+1,dec_good[43]-1,dec_good[43] + 1,'black') #plot FTF1
ra_dec_box(ra_good[44]-1,ra_good[44]+1,dec_good[44]-1,dec_good[44] + 1,'black') #plot FTF1
ra_dec_box(ra_good[45]-1,ra_good[45]+1,dec_good[45]-1,dec_good[45] + 1,'black') #plot FTF1
ra_dec_box(ra_good[46]-1,ra_good[46]+1,dec_good[46]-1,dec_good[46] + 1,'black') #plot FTF1
#ra_dec_box(ra_good[47]-1,ra_good[47]+1,dec_good[47]-1,dec_good[47] + 1,'black') #plot FTF1
#ra_dec_box(ra_good[48]-1,ra_good[48]+1,dec_good[48]-1,dec_good[48] + 1,'black') #plot FTF1
#ra_dec_box(ra_good[49]-1,ra_good[49]+1,dec_good[49]-1,dec_good[49] + 1,'black') #plot FTF1
#ra_dec_box(ra_good[50]-1,ra_good[50]+1,dec_good[50]-1,dec_good[50] + 1,'black') #plot FTF1



#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('PROVA250_Polarization_Fraction_cartesian_proj_NSIDE = '+str(nside)+'.png',format='png')

hp.cartview(sim250_SNR,title="250 $\mu$m SNR mean for 2x2deg Patch",rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,lonra=[-112,65],latra=[-63,-16])
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
ra_dec_box(ra_good[2]-1,ra_good[2]+1,dec_good[2]-1,dec_good[2] + 1,'black') #plot FTF1
ra_dec_box(ra_good[3]-1,ra_good[3]+1,dec_good[3]-1,dec_good[3] + 1,'black') #plot FTF1
ra_dec_box(ra_good[4]-1,ra_good[4]+1,dec_good[4]-1,dec_good[4] + 1,'black') #plot FTF1
ra_dec_box(ra_good[5]-1,ra_good[5]+1,dec_good[5]-1,dec_good[5] + 1,'black') #plot FTF1
#ra_dec_box(ra_good[6]-1,ra_good[6]+1,dec_good[6]-1,dec_good[6] + 1,'black') #plot FTF1
ra_dec_box(ra_good[7]-1,ra_good[7]+1,dec_good[7]-1,dec_good[7] + 1,'black') #plot FTF1
ra_dec_box(ra_good[8]-1,ra_good[8]+1,dec_good[8]-1,dec_good[8] + 1,'black') #plot FTF1
ra_dec_box(ra_good[9]-1,ra_good[9]+1,dec_good[9]-1,dec_good[9] + 1,'black') #plot FTF1
ra_dec_box(ra_good[10]-1,ra_good[10]+1,dec_good[10]-1,dec_good[10] + 1,'black') #plot FTF1
ra_dec_box(ra_good[11]-1,ra_good[11]+1,dec_good[11]-1,dec_good[11] + 1,'black') #plot FTF1
ra_dec_box(ra_good[12]-1,ra_good[12]+1,dec_good[12]-1,dec_good[12] + 1,'black') #plot FTF1
ra_dec_box(ra_good[13]-1,ra_good[13]+1,dec_good[13]-1,dec_good[13] + 1,'black') #plot FTF1
ra_dec_box(ra_good[14]-1,ra_good[14]+1,dec_good[14]-1,dec_good[14] + 1,'black') #plot FTF1
ra_dec_box(ra_good[15]-1,ra_good[15]+1,dec_good[15]-1,dec_good[15] + 1,'black') #plot FTF1
ra_dec_box(ra_good[16]-1,ra_good[16]+1,dec_good[16]-1,dec_good[16] + 1,'black') #plot FTF1
ra_dec_box(ra_good[17]-1,ra_good[17]+1,dec_good[17]-1,dec_good[17] + 1,'black') #plot FTF1
ra_dec_box(ra_good[18]-1,ra_good[18]+1,dec_good[18]-1,dec_good[18] + 1,'black') #plot FTF1
ra_dec_box(ra_good[19]-1,ra_good[19]+1,dec_good[19]-1,dec_good[19] + 1,'black') #plot FTF1
ra_dec_box(ra_good[20]-1,ra_good[20]+1,dec_good[20]-1,dec_good[20] + 1,'black') #plot FTF1
ra_dec_box(ra_good[21]-1,ra_good[21]+1,dec_good[21]-1,dec_good[21] + 1,'black') #plot FTF1
ra_dec_box(ra_good[22]-1,ra_good[22]+1,dec_good[22]-1,dec_good[22] + 1,'black') #plot FTF1
ra_dec_box(ra_good[23]-1,ra_good[23]+1,dec_good[23]-1,dec_good[23] + 1,'black') #plot FTF1
ra_dec_box(ra_good[24]-1,ra_good[24]+1,dec_good[24]-1,dec_good[24] + 1,'black') #plot FTF1
ra_dec_box(ra_good[25]-1,ra_good[25]+1,dec_good[25]-1,dec_good[25] + 1,'black') #plot FTF1
ra_dec_box(ra_good[26]-1,ra_good[26]+1,dec_good[26]-1,dec_good[26] + 1,'black') #plot FTF1
ra_dec_box(ra_good[27]-1,ra_good[27]+1,dec_good[27]-1,dec_good[27] + 1,'black') #plot FTF1
ra_dec_box(ra_good[28]-1,ra_good[28]+1,dec_good[28]-1,dec_good[28] + 1,'black') #plot FTF1
ra_dec_box(ra_good[29]-1,ra_good[29]+1,dec_good[29]-1,dec_good[29] + 1,'black') #plot FTF1
ra_dec_box(ra_good[30]-1,ra_good[30]+1,dec_good[30]-1,dec_good[30] + 1,'black') #plot FTF1
ra_dec_box(ra_good[31]-1,ra_good[31]+1,dec_good[31]-1,dec_good[31] + 1,'black') #plot FTF1
ra_dec_box(ra_good[32]-1,ra_good[32]+1,dec_good[32]-1,dec_good[32] + 1,'black') #plot FTF1
ra_dec_box(ra_good[33]-1,ra_good[33]+1,dec_good[33]-1,dec_good[33] + 1,'black') #plot FTF1
ra_dec_box(ra_good[34]-1,ra_good[34]+1,dec_good[34]-1,dec_good[34] + 1,'black') #plot FTF1
ra_dec_box(ra_good[35]-1,ra_good[35]+1,dec_good[35]-1,dec_good[35] + 1,'black') #plot FTF1
ra_dec_box(ra_good[36]-1,ra_good[36]+1,dec_good[36]-1,dec_good[36] + 1,'black') #plot FTF1
ra_dec_box(ra_good[37]-1,ra_good[37]+1,dec_good[37]-1,dec_good[37] + 1,'black') #plot FTF1
ra_dec_box(ra_good[38]-1,ra_good[38]+1,dec_good[38]-1,dec_good[38] + 1,'black') #plot FTF1
ra_dec_box(ra_good[39]-1,ra_good[39]+1,dec_good[39]-1,dec_good[39] + 1,'black') #plot FTF1
ra_dec_box(ra_good[40]-1,ra_good[40]+1,dec_good[40]-1,dec_good[40] + 1,'black') #plot FTF1
ra_dec_box(ra_good[41]-1,ra_good[41]+1,dec_good[41]-1,dec_good[41] + 1,'black') #plot FTF1
ra_dec_box(ra_good[42]-1,ra_good[42]+1,dec_good[42]-1,dec_good[42] + 1,'black') #plot FTF1
ra_dec_box(ra_good[43]-1,ra_good[43]+1,dec_good[43]-1,dec_good[43] + 1,'black') #plot FTF1
ra_dec_box(ra_good[44]-1,ra_good[44]+1,dec_good[44]-1,dec_good[44] + 1,'black') #plot FTF1
ra_dec_box(ra_good[45]-1,ra_good[45]+1,dec_good[45]-1,dec_good[45] + 1,'black') #plot FTF1
ra_dec_box(ra_good[46]-1,ra_good[46]+1,dec_good[46]-1,dec_good[46] + 1,'black') #plot FTF1
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('PROVA250_Polarized_SNR_(mean_removed_in_Q_U)_mean_cartesian_proj_NSIDE = '+str(nside)+'.png',format='png')




### 350
#Map the Polarization Fraction and the SNR refering to the Polarized Power in Mollewiede Proj
hp.mollview(sim350_PF,coord=['G','C'],cmap='seismic',title="350_$\mu$m Polarization Fraction",unit='log10(MJy/Sr)',min=0,max=0.2)
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('350_Polarization_Fraction_NSIDE = '+str(nside)+'.png',format='png')

hp.mollview(sim350_SNR,coord=['G','C'],cmap='seismic',title="350 $\mu$m SNR mean for 2x2deg Patch",min=0,max=10,cbar=True)
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('350_Polarized_SNR_(mean_removed_in_Q_U)_mean_NSIDE = '+str(nside)+'.png',format='png')

#Map the Polarization Fraction and the SNR refering to the Polarized Power in Cartesian Proj
hp.cartview(sim350_PF,title="350 $\mu$m Polarization Fraction",rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=0.2,cbar=True,lonra=[-112,65],latra=[-63,-16])
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('350_Polarization_Fraction_cartesian_proj_NSIDE = '+str(nside)+'.png',format='png')

hp.cartview(sim350_SNR,title="350 $\mu$m SNR mean for 2x2deg Patch",rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,lonra=[-112,65],latra=[-63,-16])
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('350_Polarized_SNR_(mean_removed_in_Q_U)_mean_cartesian_proj_NSIDE = '+str(nside)+'.png',format='png')




### 500
#Map the Polarization Fraction and the SNR refering to the Polarized Power in Mollewiede Proj
hp.mollview(sim500_PF,coord=['G','C'],cmap='seismic',title="500_$\mu$m Polarization Fraction",unit='log10(MJy/Sr)',min=0,max=0.2)
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('500_Polarization_Fraction_NSIDE = '+str(nside)+'.png',format='png')

hp.mollview(sim500_SNR,coord=['G','C'],cmap='seismic',title="500 $\mu$m SNR mean for 2x2deg Patch",min=0,max=10,cbar=True)
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('500_Polarized_SNR_(mean_removed_in_Q_U)_mean_NSIDE = '+str(nside)+'.png',format='png')

#Map the Polarization Fraction and the SNR refering to the Polarized Power in Cartesian Proj
hp.cartview(sim500_PF,title="500 $\mu$m Polarization Fraction",rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=0.2,cbar=True,lonra=[-112,65],latra=[-63,-16])
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('500_Polarization_Fraction_cartesian_proj_NSIDE = '+str(nside)+'.png',format='png')

hp.cartview(sim500_SNR,title="500 $\mu$m SNR mean for 2x2deg Patch",rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,lonra=[-112,65],latra=[-63,-16])
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
#ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
#ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')
#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('500_Polarized_SNR_(mean_removed_in_Q_U)_mean_cartesian_proj_NSIDE = '+str(nside)+'.png',format='png')

"""
hp.cartview(sim250_PF,title='250 $\mu$m FTF1 Polarization Fraction for 48h on 2x2deg Patch',lonra=[ra0_1-w_1,ra0_1+w_1], latra=[dec0_1-w_1,dec0_1+w_1],min=0,max=0.2,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
    
fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cb = fig.colorbar(image, ax=ax)
cb.set_label('SNR')
cb.set_ticks(np.linspace(0,10,11))
plt.savefig('YES250_mean = '+str(nside)+'.png',format='png')
"""    
    
hp.cartview(sim500_PF,title='500 $\mu$m FTF1 Polarization Fraction for 48h on 2x2deg Patch',lonra=[ra0_1-w_1,ra0_1+w_1], latra=[dec0_1-w_1,dec0_1+w_1],min=0,max=0.2,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
    
fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cb = fig.colorbar(image, ax=ax)
cb.set_label('Polarization Fraction')
cb.set_ticks(np.linspace(0,0.2,11))
plt.savefig('YES500_mean = '+str(nside)+'.png',format='png')



#----------------------------------------------------------------------------------
"""
data_name=['250','350','500']
sim_I=[sim250_I,sim350_I,sim500_I]
sim_Pol=[sim250_Pol,sim350_Pol,sim500_Pol]
Spix=[Spix250,Spix350,Spix500]

for i in range(0,3):
    
    #250 SNR Plots
    #Map the Intensity in Mollewiede Proj
    hp.mollview(np.log10(sim_I[i]),coord=['G','C'],cmap='seismic',title=data_name[i]+"$\mu$m Intensity",\
        unit='log10(MJy/Sr)',min=-1,max=2)
    #connect_the_dots(x1,y1,'white')  #plot the 1hr/day contour
    connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
    connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
    ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
    ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF2
    #Add some graticules            
    hp.graticule(30,30,coord='C',color='black')
    
    #Mark the RA
    for RA in [0,30,60,90,120,150,210,240,270,300,330]:
        hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
    #Mark the DEC
    for DEC in [30,60,-30,-60]:
        hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
    plt.savefig('VF'+data_name[i]+'_LOG_Intensity_NSIDE = '+str(nside)+'.png',format='png')
    
    
    
    #Map the Signal in Mollewiede Proj
    hp.mollview(np.log10(sim_Pol[i]),coord=['G','C'],cmap='seismic',title=data_name[i]+"$\mu$m Polarized Power",\
        unit='log10(MJy/Sr)',min=-3,max=2)
    #connect_the_dots(x1,y1,'white')  #plot the 1hr/day contour
    connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
    connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
    ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
    ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
    #Add some graticules            
    hp.graticule(30,30,coord='C',color='black')
    
    #Mark the RA
    for RA in [0,30,60,90,120,150,210,240,270,300,330]:
        hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
    #Mark the DEC
    for DEC in [30,60,-30,-60]:
        hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
    plt.savefig('VF'+data_name[i]+'_LOG_PolarizedPower_NSIDE = '+str(nside)+'.png',format='png')
    
    #Map the PolFrac in Mollewiede Proj
    hp.mollview(sim_Pol[i]/sim_I[i],coord=['G','C'],cmap='seismic',title=data_name[i]+"$\mu$m Polarization Fraction",\
        unit='log10(MJy/Sr)',min=0,max=0.2)
    #connect_the_dots(x1,y1,'white')  #plot the 1hr/day contour
    connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
    connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
    ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
    ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
    #Add some graticules            
    hp.graticule(30,30,coord='C',color='black')
    #Mark the RA
    for RA in [0,30,60,90,120,150,210,240,270,300,330]:
        hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
    #Mark the DEC
    for DEC in [30,60,-30,-60]:
        hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
    plt.savefig('VF'+data_name[i]+'_PolFrac_NSIDE = '+str(nside)+'.png',format='png')

    
    #Map the SNR in Mollewiede Proj
    hp.mollview(sim_Pol[i]/Spix[i],coord=['G','C'],cmap='seismic',title=data_name[i]+"$\mu$m SNR for 2x2deg Patch",\
        min=0,max=10,cbar=True)
    #hp.cartview(sim_Pol[i]/Spix[i],title=data_name[i]+"$\mu$m SNR for 2x2deg Patch",\
        #rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,\
        #lonra=[-135,120],latra=[-90,0])
    #connect_the_dots(x1,y1,'white')  #plot the 1hr/day contour
    connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
    connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
    ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
    ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
    #Add some graticules            
    hp.graticule(30,30,coord='C',color='w')
    #Mark the RA
    for RA in [0,30,60,90,120,150,210,240,270,300,330]:
        hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='w',fontsize=6)
    #Mark the DEC
    for DEC in [30,60,-30,-60]:
        hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='w',fontsize=6)
    plt.savefig('VF'+data_name[i]+'_SNR_NSIDE = '+str(nside)+'.png',format='png')




    #Plot the SNR over the whole BLAST region as a big cartview
    hp.cartview(sim_Pol[i]/Spix[i],title=data_name[i]+"$\mu$m SNR for 2x2deg Patch",\
        rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,\
        lonra=[-135,120],latra=[-90,0])
    #connect_the_dots(x1,y1,'white')  #plot the 1hr/day contour
    connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
    connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
    ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
    ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
    #Add some graticules            
    hp.graticule(30,30,coord='C',color='w')
    #Mark the RA
    for RA in [0,30,60,90,120,150,210,240,270,300,330]:
        hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='w',fontsize=6)
    #Mark the DEC
    for DEC in [30,60,-30,-60]:
        hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='w',fontsize=6)
    plt.savefig('VF'+data_name[i]+'_SNR_BLAST_REGION_NSIDE = '+str(nside)+'.png',format='png')
    
    #Plot the Polarization Fraction over the whole BLAST region as a big cartview
    hp.cartview(sim_Pol[i]/sim_I[i],title=data_name[i]+"$\mu$m Polarization Fraction",\
        rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=0.2,cbar=True,\
        lonra=[-135,120],latra=[-90,0])
    #connect_the_dots(x1,y1,'white')  #plot the 1hr/day contour
    connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
    connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
    ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
    ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
    #Add some graticules            
    hp.graticule(30,30,coord='C',color='w')
    #Mark the RA
    for RA in [0,30,60,90,120,150,210,240,270,300,330]:
        hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='w',fontsize=6)
    #Mark the DEC
    for DEC in [30,60,-30,-60]:
        hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='w',fontsize=6)
    plt.savefig('VF'+data_name[i]+'_PolFrac_BLAST_REGION_NSIDE = '+str(nside)+'.png',format='png')
    

    #Plot the I Map  over the whole BLAST region as a big cartview
    hp.cartview(np.log10(sim_I[i]),title=data_name[i]+"$\mu$m I",\
        rot=(0,180,180,),coord=['G','C'],cmap='seismic',cbar=True,\
        lonra=[-135,120],latra=[-90,0],unit='log10(MJy/sr)',min=-1,max=3)
    #connect_the_dots(x1,y1,'white')  #plot the 1hr/day contour
    connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
    connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
    #ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
    #ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
    #Add some graticules            
    hp.graticule(30,30,coord='C',color='w')
    #Mark the RA
    for RA in [0,30,60,90,120,150,210,240,270,300,330]:
        hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='w',fontsize=6)
    #Mark the DEC
    for DEC in [30,60,-30,-60]:
        hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='w',fontsize=6)
    #plt.savefig('VF'+'250_I_BLAST_REGION_NSIDE = '+str(nside)+'.png',format='png')




    
    #Zoom on FTF1
    #FTF1
    hp.cartview(sim_Pol[i]/Spix[i],title=data_name[i]+'$\mu$m FTF1 SNR for 48h on 2x2deg Patch',lonra=[ra0_1-w_1,ra0_1+w_1],\
        latra=[dec0_1-w_1,dec0_1+w_1],min=0,max=10,cmap='seismic',cbar=True,coord=['G','C'],xsize=2048)
    
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    cb = fig.colorbar(image, ax=ax)
    cb.set_label('SNR')
    cb.set_ticks(np.linspace(0,10,11))
    plt.savefig('VF'+data_name[i]+'_SNR_FTF1_NSIDE = '+str(nside)+'.png',format='png')
    
    #Zoom on FTF2
    hp.cartview(sim_Pol[i]/Spix[i],title=data_name[i]+' FTF2 SNR for 48h on 2x2deg Patch',lonra=[ra0_2-w_2,ra0_2+w_2],\
        latra=[dec0_2-w_2,dec0_2+w_2],min=0,max=10,cmap='seismic',cbar=False,coord=['G','C'], return_projected_mapbool=True)
    
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    cb = fig.colorbar(image, ax=ax)
    cb.set_label('SNR')
    cb.set_ticks(np.linspace(0,10,11))
    plt.savefig('VF'+data_name[i]+'_SNR_FTF2_NSIDE = '+str(nside)+'.png',format='png')
    ####
    #Zoom on FTF1
    #FTF1
    hp.cartview(sim_Pol[i]/sim_I[i],title=data_name[i]+'$\mu$m FTF1 PolFrac',lonra=[ra0_1-w_1,ra0_1+w_1],\
        latra=[dec0_1-w_1,dec0_1+w_1],min=0,max=0.2,cmap='seismic',cbar=False,coord=['G','C'],xsize=2048)
    
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    cb = fig.colorbar(image, ax=ax)
    cb.set_label('PolFrac')
    cb.set_ticks(np.linspace(0,0.2,5))
    plt.savefig('VF'+data_name[i]+'_PolFrac_FTF1_NSIDE = '+str(nside)+'.png',format='png')
    
    #Zoom on FTF2
    hp.cartview(sim_Pol[i]/sim_I[i],title=data_name[i]+' FTF2 PolFrac',lonra=[ra0_2-w_2,ra0_2+w_2],\
        latra=[dec0_2-w_2,dec0_2+w_2],min=0,max=0.2,cmap='seismic',cbar=False,coord=['G','C'])
    
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    cb = fig.colorbar(image, ax=ax)
    cb.set_label('PolFrac')
    cb.set_ticks(np.linspace(0,0.2,5))
    plt.savefig('VF'+data_name[i]+'_PolFrac_FTF2_NSIDE = '+str(nside)+'.png',format='png')
    #----------------------------------------------------------------------------------



#350 SNR Plots
#Map the Signal in Mollewiede Proj
hp.mollview(np.log(sim350_Pol),coord=['G','C'],cmap='seismic',title="350$\mu$m Polarized Power",\
    min=-5,max=2,unit='Log(MJy/Sr)')
connect_the_dots(x1,y10,'white')  #plot the 1hr/day contour
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')

#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('VF'+'350_LOG_PolarizedPower_NSIDE = '+str(nside)+'.png',format='png')

#Plot the SNR over the whole BLAST region as a big cartview
hp.cartview(sim350_Pol/Spix350,title="350$\mu$m Polarized Power SNR for 5x5deg Patch",\
    rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,\
    lonra=[-135,120],latra=[-90,0])
connect_the_dots(x1,y10,'white')  #plot the 1hr/day contour
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
#Add some graticules            
hp.graticule(30,30,coord='C',color='w')

#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='w',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='w',fontsize=6)
#fig = plt.gcf()
#ax = plt.gca()
#image = ax.get_images()[0]
#cb = fig.colorbar(image, ax=ax)
#cb.set_label('SNR')
#cb.set_ticks(np.linspace(0,10,11))
#txt=hp.projtext(ra0-2,dec0+2,'FTF1',lonlat='true',color='green',fontsize=8,fontweight='bold')
#txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
plt.savefig('VF'+'350_SNR_BLAST_REGION_NSIDE = '+str(nside)+'.pdf',format='pdf')

#Zoom on FTF1
#FTF1
hp.cartview(sim350_Pol/Spix350,title='350$\mu$m FTF1 SNR',lonra=[ra0_1-w_1,ra0_1+w_1],\
    latra=[dec0_1-w_1,dec0_1+w_1],min=0,max=10,cmap='seismic',cbar=False,coord=['G','C'])

fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cb = fig.colorbar(image, ax=ax)
cb.set_label('SNR')
cb.set_ticks(np.linspace(0,10,11))
plt.savefig('VF'+'350_SNR_FDF1_NSIDE = '+str(nside)+'.png',format='png')

#Zoom on FTF2
hp.cartview(sim350_Pol/Spix350,title='350 FTF2 SNR',lonra=[ra0_2-w_2,ra0_2+w_2],\
    latra=[dec0_2-w_2,dec0_2+w_2],min=0,max=10,cmap='seismic',cbar=False,coord=['G','C'])

fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cb = fig.colorbar(image, ax=ax)
cb.set_label('SNR')
cb.set_ticks(np.linspace(0,10,11))
plt.savefig('VF'+'350_SNR_FTF2_NSIDE = '+str(nside)+'.png',format='png')

#----------------------------------------------------------------------------------

#500 SNR Plots
#Map the Signal in Mollewiede Proj
hp.mollview(np.log(sim500_Pol),coord=['G','C'],cmap='seismic',title="500$\mu$m Polarized Power",\
    min=-5,max=2,unit='Log(MJy/Sr)')
connect_the_dots(x1,y10,'white')  #plot the 1hr/day contour
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
#Add some graticules            
hp.graticule(30,30,coord='C',color='black')

#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='black',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='black',fontsize=6)
plt.savefig('VF'+'500_LOG_PolarizedPower_NSIDE = '+str(nside)+'.png',format='png')

#Plot the SNR over the whole BLAST region as a big cartview
hp.cartview(sim500_Pol/Spix500,title="500$\mu$m Polarized Power SNR for 5x5deg Patch",\
    rot=(0,180,180,),coord=['G','C'],cmap='seismic',min=0,max=10,cbar=True,\
    lonra=[-135,120],latra=[-90,0])
connect_the_dots(x1,y10,'white')  #plot the 1hr/day contour
connect_the_dots(x10,y10,'white') #plot the 10hr/day contour
connect_the_dots(x20,y20,'white') #plot the 20hr/day contour
ra_dec_box(ra0_1-w_1,ra0_1+w_1,dec0_1-w_1,dec0_1+w_1,'white') #plot FTF1
ra_dec_box(ra0_2-w_2,ra0_2+w_2,dec0_2-w_2,dec0_2+w_2,'white') #plot FTF1
#Add some graticules            
hp.graticule(30,30,coord='C',color='w')

#Mark the RA
for RA in [0,30,60,90,120,150,210,240,270,300,330]:
    hp.projtext(RA,1,np.str(RA),lonlat=True,coord='C',color='w',fontsize=6)
#Mark the DEC
for DEC in [30,60,-30,-60]:
    hp.projtext(0,DEC,np.str(DEC),lonlat=True,coord='C',color='w',fontsize=6)
#fig = plt.gcf()
#ax = plt.gca()
#image = ax.get_images()[0]
#cb = fig.colorbar(image, ax=ax)
#cb.set_label('SNR')
#cb.set_ticks(np.linspace(0,10,11))
#txt=hp.projtext(ra0-2,dec0+2,'FTF1',lonlat='true',color='green',fontsize=8,fontweight='bold')
#txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
plt.savefig('VF'+'500_SNR_BLAST_REGION_NSIDE = '+str(nside)+'.pdf',format='pdf')

#Zoom on FTF1
#FTF1
hp.cartview(sim500_Pol/Spix500,title='500$\mu$m FTF1 SNR',lonra=[ra0_1-w_1,ra0_1+w_1],\
    latra=[dec0_1-w_1,dec0_1+w_1],min=0,max=10,cmap='seismic',cbar=False,coord=['G','C'])

fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cb = fig.colorbar(image, ax=ax)
cb.set_label('SNR')
cb.set_ticks(np.linspace(0,10,11))
plt.savefig('VF'+'500_SNR_FDF1_NSIDE = '+str(nside)+'.png',format='png')

#Zoom on FTF2
hp.cartview(sim500_Pol/Spix500,title='500 FTF2 SNR',lonra=[ra0_2-w_2,ra0_2+w_2],\
    latra=[dec0_2-w_2,dec0_2+w_2],min=0,max=10,cmap='seismic',cbar=False,coord=['G','C'])

fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cb = fig.colorbar(image, ax=ax)
cb.set_label('SNR')
cb.set_ticks(np.linspace(0,10,11))
plt.savefig('VF'+'500_SNR_FTF2_NSIDE = '+str(nside)+'.png',format='png')
"""
