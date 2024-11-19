# 	Utility functions for free-fre:2e radiative transfer calculation through 
# 	a spherically symmetric stellar wind (Parker's model)
# 	Author: H. K. Vedantham
# 	ASTRON, Oct 2023
#
import numpy as np
import matplotlib.pyplot as plt
from const import *
from scipy.interpolate import RegularGridInterpolator as interp2
#from numba import jit # Could be used if speed becomes an issue
#
#
#@jit
def give_alpha_ff(n,T,nu,Z2=1.16):
   # Compute the free free absorption coefficient for a thermal plasma
   # n = charge density in cm^-3
   # T = temperature in Kelvin
   # nu = Frequency in Hz
   # Z2 = avg value of ion charge squared (default value is solar)
   #
   gff = give_gff(T,nu) 	# Free free Guant factor 
   nup = give_plasma_freq(n)    # Electron plasma freqeuncy in Hz
   ref_ind = (1-nup**2/nu**2)**0.5   # Refractive index for an isotropic plasma
   # Use equation 5.19b from Rybicki and Lightman to calculate the abs. coeff
   # divide by ref index to correct for group velocity effects. 
   # See eqn 11.2.1 of Benz's plasma astrophysics book
   return 0.01772033630889372*T**-1.5*Z2*n**2*nu**-2*gff/ref_ind
#
#@jit
def give_gff_approx(T,nu):
   # return the Free free Guant factor
   # From Condon and Ransom eqn 4.59
   return 1.5*np.log(T)+ np.log(4.955e-2*1e9/nu)
#
def give_gff(T,nu):
   # Return the free-free Gaunt factor using tables from 
   # van Hoof et al https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..420V/abstract
   # T is temp in K
   # nu is frequeny in Hz
   #
   log_g2 = np.log10(C_Ry/C_k/T) # Gamma^2 defined in van Hoof et al. 
   log_u = np.log10(C_h*nu/C_k/T) # u defined in van Hoof et al.
   tp = np.loadtxt("gauntff.dat",comments="#") # Read in the tables
   ugrid = np.arange(-16,13.2,0.2)	# u vales used in table
   g2grid = np.arange(-6,10.2,0.2)	# gamma^2 values used in table
   f = interp2(points=(ugrid,g2grid),values=tp) # Interpolant
   return f([log_u,log_g2])[0]
#
#
#@jit
def give_plasma_freq(n):
   # Return plasma freq given electron density
   # Note the use of Gaussian units formula
   return (4*np.pi*n*C_e**2/C_me)**0.5 / (2*np.pi)
#
#@jit
def give_group_vel(n,nu):
   # Return group velocity given density and freq
   # Assume an isotropic plasma (i.e. no B field)
   nup = give_plasma_freq(n);
   if nu<nup:
      return 0.0; # If freq < plasma freq then return 0 (non propagating modes) 
   else:
      return C_c*(1-nup**2/nu**2)**0.5   
#
#@jit
def ff_int(r,n,T,nu,Z2=1.16):
   # Calculate the emergent optical depth and specific intensity of free-free emission
   # Given:
   #	r = radius vector values (code assumes a regular grid for now)
   #        Maybe update later for any grid and use Gaussian quadrature integration
   #	n = Density vector
   #	T = Temp. vector (or scalar)
   #	nu = Frequency (scalar)
   #
   dr = np.absolute(r[2]-r[1])		# distance increment
   alpha = give_alpha_ff(n,T,nu,Z2)	# Absorption coefficient
   nup = give_plasma_freq(n)		# Plasma freq
   ref_ind = (1-nup**2/nu**2)**0.5	# Ref index
   tau_tot = np.sum(alpha*dr)		# Total optical depth
   source_func = 2*C_k*T/C_c**2*nu**2 	# Source function in vaccum (Rayleigh-Jeans)

   if tau_tot>=5:			# Optically thick regime
					# Same computation time by assuming blackbody
      I = source_func

   elif tau_tot<1e-2:			# Optically thin regime
					# Save time by assuming exp(-tau) = 1-tau

      opt_depth = np.array([dr*np.sum(alpha[i:]) for i in range(len(alpha))])
      I = source_func*dr*np.sum(alpha*(1-opt_depth))

   else:				# Intermediate case (need full calculation) 
      opt_depth = np.array([dr*np.sum(alpha[i:]) for i in range(len(alpha))])
      I = source_func*np.sum(alpha*np.exp(-opt_depth))*dr

   return tau_tot,I
#
#
#@jit
def give_density_profile_parker(mstar,rstar,T,rvec,Mdot,mu=0.6):
   # Return the Parker wind solution
   # mstar = Stellar mass in g
   # rstar = Stellar radius in cm
   # T = Coronal temp (isothermal) in K
   # rvec = Radius vector grid on which solution is needed
   # Mdot = Mass loss rate to scale base density (g/s)
   # mu = Mean atomic weight
   #
   tp = np.load("parker_wind_vel_sol_lin.npz")
   cs2 = C_k*T/C_mp/mu
   rc = C_G*mstar/(2*cs2)
   rnorm = rvec/rc
   cs=cs2**0.5
   v = cs*np.interp(x=rnorm,xp=tp['r'],fp=tp['v'])
   rho = Mdot / (v*4*np.pi*rvec**2)
   return rho/(mu*C_mp)
   #
#@jit
def scale_height(M,R,T): # Density scale height
   # M = mass of the star in gm
   # R = radius of the star in cm
   # T = plasma temperature
   return 2*C_k*T*(R)**2/(C_G*M*C_mp)
#
#@jit
def ion_thermal_speed(T):
        return (C_k*T/C_mp)**0.5
#
#@jit
def give_density_profile_r2(mstar,rstar,T,n0,rmax=None):
   # Return 1/R^2 density profile (only for testing; not used otherwise)
   hp = scale_height(mstar,rstar,T)
   dr = min(rstar/5,hp/5)
   print ("rstar = %.2e, scale height = %.2e"%(rstar,hp))
   if rmax is None:
      rmax = max(10*rstar,10*hp)
   r = np.arange(rstar,rmax,dr)
   n = n0*(r/rstar)**-2
   return r,n
#
#@jit
def int_image(mstar,rstar,T,nu,Mdot,rmax_rstar,dist):
   # 2D ray trace to compute the free-free flux
   # mstar = Stellar mass in grams
   # rstar = stellar radius in cm
   # T = coronal temp in Kelvin
   # nu = freq in Hz
   # n0 = base density in cm^-3
   #
   print (" ----- 2D trace nu = %.2e MHz ------- \n"%(nu/1e6))
   hp = scale_height(mstar,rstar,T)	# Scale height
   dr = min(rstar/5,hp/5)		# numerical grid resolution
   rmax = rstar*rmax_rstar		# numerical grid extent
   r = np.arange(rstar,rmax,dr)		
   n = give_density_profile_parker( mstar,rstar,T,r,Mdot)

   # Compute the Zone 1 radial size
   nup_list = give_plasma_freq(n)
   if nu > nup_list[0]:
      xmin_plasma = rstar
   else:
      xmin_plasma = r[np.where(nu>nup_list)[0][0]]

   # Compute dh and hint: the grid for lateral (plane of sky) distance from stellar centre
   dh = r[1]-r[0]
   hint = np.arange(0.0,rmax,dh)
   # Initialize flux and optical depth vectors (functions of lateral distance)
   F = []
   tau = []
   
   norm = dist**2/1e23	# To convert to Jy from c.g.s with 1/D^2 scaling
   #
   #print ("nu = %.1f GHz, rmin_plasma = %.2f r*"%(nu/1e9,xmin_plasma/rstar))
   for h in hint:
      if h<xmin_plasma: # If the ray ends on the surface of the Zone 1 sphere
         xmin = (xmin_plasma**2-h**2)**0.5
         xmax = (rmax**2-h**2)**0.5
         xvec=np.arange(xmin,xmax,dh)
         rvec = (h**2+xvec**2)**0.5
         nvec = np.interp(rvec,r,n)
         tp1,tp2=ff_int(xvec,nvec,T,nu)
         F.append(tp2 *2*np.pi*dh*h/norm)
         tau.append(tp1) 
         #print ("Ray (I) y=%.2e, tau = %.2e, F=%.2e"%(h/rstar,tau[-1],F[-1]))
      else:		# If ray misses the surface of the Zone 1 sphere
         xmax = (rmax**2-h**2)**0.5 
         xvec=np.arange(-xmax,xmax,dh)
         rvec = (h**2+xvec**2)**0.5
         nvec = np.interp(rvec,r,n)
         tp1,tp2=ff_int(xvec,nvec,T,nu)
         F.append(tp2 *2*np.pi*dh*h/norm)
         tau.append(tp1) 
         #print ("Ray (O) y=%.2e, tau = %.2e, F=%.2e"%(h/rstar,tau[-1],F[-1]))
   return np.array(hint),np.array(tau),np.array(F)


def int_1d_approx_parker(mstar,rstar,T,nu,Mdot,dist=C_pc*10,mu=0.6):
   # Calculate the approx flux density in a 1D model (radial ray)
   # Based on a 3-zone model
   # Zone-1: Wave freq less than plasma freq (no emission contribuion)
   # Zone-2: Optically thick (blackbody contribution)
   # Zone-3: Optically thin (intensity = Radial integraion of emissivity)
   # Also determine the Zone-3 outer boundary given some tolerance
   # to form an rmax input for full 2D ray trace calculations
   #
   #

   print (" ----- Aprox 1D radial trace and Zone sectioning ----- ")
   # First compute the Parker density profile
   tp = np.load("parker_wind_vel_sol_lin.npz")
   cs2 = C_k*T/C_mp		# Sound speed squared
   rc = C_G*mstar/(2*cs2)	# Critical radius
   rvec = tp['r']*rc
   cs=cs2**0.5
   v = cs*tp['v']
   # Cut the inner boundary at rstar and outer boundary at 200R*
   I = np.logical_and(rvec>rstar,rvec<200*rstar)
   v=v[I]
   rvec=rvec[I]
   rho = Mdot / (v*4*np.pi*rvec**2)
   n = rho/(mu*C_mp)
   assert len(v) > 100, "Increase outer limit in int_1d_approx_parker"
   source_func = 2*C_k*T*(nu/C_c)**2 # Source function (ignoring ref index effects)
   alpha = np.zeros(n.shape); # Refractive index array initiaization


   # Compute the Zone-1 outer boundary
   nup_list = give_plasma_freq(n)
   tp = np.where(nu>nup_list)[0]
   assert len(tp), "nu=%.2e Hz: Zone 1 boundary is beyond Parker's numerical model. Code fails!"%nu
   I1_outer = tp[0]
   r1_outer = rvec[I1_outer] 
   print ("nu=%.2e MHz: Zone 1 outer boundary at %.2f R*"%(nu/1e6,r1_outer/rstar))

   # Compute refractive index for all points beyond Zone - 1 for future use
   alpha[I1_outer:] = give_alpha_ff(n[I1_outer:],T,nu,Z2=1.16);  # Absorption coefficient 
   
   # Compute the Zone-2 outer boundary
   dr = rvec[1]-rvec[0]
   tau_frac = np.array([np.sum(alpha[i+I1_outer:])*dr for i in range(len(rvec[I1_outer:]))])
   tp = np.where(tau_frac>=1)[0]
   if len(tp):
      I2_outer = tp[-1]+I1_outer
      r2_outer = rvec[I2_outer]
      print ("nu=%.2e MHz: Zone 2 outer boundary at %.2f R*"%(nu/1e6,r2_outer/rstar) )
      # Compute the flux density from Zone 2
      F2 = source_func*np.pi*(r2_outer/dist)**2
   else:
      print ("nu=%.2e MHz: No optically thick Zone-2"%(nu/1e6))
      I2_outer = I1_outer
      r2_outer=r1_outer 
      F2=0
   # Compute the emissivity, Intensity and optically thin Flux contribution per dr in Zone 3
   dI = (alpha[I2_outer:] * source_func) * dr
   dF = (dI*2*np.pi*rvec[I2_outer:]*dr) / (dist**2)
   F_cum = np.array([np.sum(dF[:i]) for i in range(len(dF))])
   #plt.plot(F_cum); plt.show(); plt.close()
   I3_outer = np.where(F_cum/F_cum[-1]>0.99)[0][0] + I2_outer
   r3_outer = rvec[I3_outer]
   print ("nu=%.2e MHz: Zone 3 outer boundary is at %.2f R*"%(nu/1e6,r3_outer/rstar))
   print (" ----------------------------------------------------- ")
   #
   # return the total flux and Zone 3 outer boundary
   return (F2+F_cum[-1])*1e23,r3_outer
