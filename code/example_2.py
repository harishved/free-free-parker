# Compute free-free flux density for given stellar parameters
# and freq as a function of Mdot
# Author: H. K. Vedantham
# ASTRON, Oct 2023
#
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from utils import ff_int, int_image,int_1d_approx_parker
from const import *

###########################
#    INPUT PARAMETERS #
# ------------------------
Mdot_msun_list = np.logspace(1,4,10)
dist_pc = 9.714
rstar_rsun = 0.82
mstar_msun = 0.6
T_MK = 7
nu = 20e9 	# Freq in Hz
#
Mdot_list = Mdot_msun_list*2e-14*C_Msun/(365*24*3600) # Mass loss rate in g/s
rstar=rstar_rsun*C_Rsun			# Stellar radius in cm
mstar=mstar_msun*C_Msun			# Stellar mass in g
T = T_MK * 1e6				# Coronal temp in K
#
# Title string for plots
parstr = r"$\nu=%d\,{\rm GHz},\,R_\ast=%.2f\,R_\odot,\,M_\ast=%.2f\,M_\odot,\,T=%.1f\,{\rm MK}$"%(nu/1e9,rstar_rsun,mstar_msun,T_MK)
#
#
taumax = np.zeros(Mdot_msun_list.shape)	# Peak optical depth during raytrace
flux_uJy = np.zeros(taumax.shape)	# Obs flux in micro Jy full 2D raytrace
flux_approx_uJy = np.zeros(taumax.shape)	# 1D flux in micro Jy 1D simple raytrace
#
#
pp = PdfPages('y_profile.pdf')

for i in range(len(Mdot_list)): # For each frequency
   Mdot = Mdot_list[i]
   # Do approx 1D radial ray trace to get 
   # (a) approx flux and
   # (b) size of the numerical grid for 2D ray trace
   tp1,tp2=int_1d_approx_parker(mstar,rstar,T,nu,Mdot,dist=C_pc*dist_pc) 
   flux_approx_uJy[i]=tp1*1e6
   #
   # Do 2D ray trace with proper geometry
   h,tau,F=int_image(mstar=mstar,rstar=rstar,T=T,nu=nu,Mdot=Mdot,rmax_rstar=1.5*tp2/rstar,dist=C_pc*dist_pc)
   plt.subplot(211)
   plt.plot(h/C_Rsun,F*1e6)
   plt.xlabel(r"$y/R_\ast$")
   plt.ylabel("Fractional flux [micro Jy]")
   plt.title(parstr,fontsize=12)
   plt.subplot(212)
   #
   plt.plot(h/C_Rsun,tau)
   plt.yscale("log")
   plt.ylabel("Fractional flux [micro Jy]")
   plt.ylabel(r"$\tau$")
   plt.suptitle(r"$\nu = $%d MHz"%(nu/1e6),fontsize=12)
   plt.tight_layout()
   #plt.show()
   pp.savefig()
   plt.close()

   taumax[i] = np.nanmax(tau)
   flux_uJy[i] = np.nansum(F*1e6)
#
pp.close()

# PLot the peak optical depth from 2D ray trace and the total flux density v/s freq
plt.subplot(211)
plt.loglog(Mdot_msun_list,taumax)
plt.xlabel("Mass-loss rate / Solar")
plt.ylabel("Peak free-free optical depth")
plt.title(parstr,fontsize=12)

plt.subplot(212)
plt.loglog(Mdot_msun_list,flux_uJy,label="This work")
plt.xlabel("Mass-loss rate/ Solar")
plt.ylabel("Free free flux / micro Jy")

# Comparison line eqn 24 from Panagia and Felli 1975
cs = (C_k*T/C_mp)**0.5
tp = np.load("parker_wind_vel_sol_lin.npz")
vexp = np.amax(tp['v'])*cs
Snu_uJy = 5.12e3 * (nu/1e10)**0.6 * (T/1e4)**0.1 * (Mdot_list*365*24*3600/C_Msun/1e-5)**(4./3)*(vexp/1e8)**(-4./3)*(dist_pc/1e3)**-2
plt.semilogx(Mdot_msun_list,Snu_uJy,'k--',label="PF75")
plt.legend()
plt.tight_layout()
plt.savefig("tau_flux.pdf")
#plt.show()
plt.close()
#
#END

'''
# Comparison line eqn 24 from Panagia and Felli 1975
cs = (C_k*T/C_mp)**0.5
tp = np.load("parker_wind_vel_sol_lin.npz")
vexp = np.amax(tp['v'])*cs
Snu_uJy = 5.12e3 * (nulist/1e10)**0.6 * (T/1e4)**0.1 * (Mdot*365*24*3600/C_Msun/1e-5)**(4./3)*(vexp/1e8)**(-4./3)*(10/1e3)**-2
plt.semilogx(nulist/1e9,Snu*1e3)
plt.show()
plt.close()
exit(1)
'''

