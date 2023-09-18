#2023.3.9
import sys
import os
import numpy as np
import h5py


import hmf
from hmf import MassFunction
from hmf.halos import mass_definitions as md
#from hmf import cosmo

from astropy.modeling import models, fitting

from colossus.lss import bias
from colossus.lss import mass_function
from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.halo import mass_so, profile_nfw


import MAS_library as MASL
import Pk_library as PKL

from dynesty import NestedSampler
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot

import camb
from camb import model, initialpower

a=0
class halo_model():
    def __init__(self, redshift = 0,lgk_min = -3,lgk_max = 2,lgk_nbin = 100,
                 lgM_min = 8,lgM_max = 16,lgM_nbin = 100,
                 lgx_min = -3,lgx_max = 3,lgx_nbin = 2000,R_cut=10, 
                 mdef = "500c",bias_model = 'tinker10',hmf_model = 'tinker08',gnfw_model = "Planck13", cosmo_model = "planck15",
                 mass_bias=0, alpha_p=0.12,
                 pk_lin = 0, dndlgM=0):
        cosmo = cosmology.setCosmology(cosmo_model);
        #lgM_bin, x_bin, k_bin
        k_bins = 10**np.linspace(lgk_min, lgk_max, lgk_nbin) # h/Mpc comoving
        lgM_bins = np.linspace(lgM_min, lgM_max, lgM_nbin)
       
        dlgM = lgM_bins[1]-lgM_bins[0]
        x = 10**np.linspace(lgx_min, lgx_max, lgx_nbin)
        
        #bh len(M)
        bh = bias.haloBias(10**lgM_bins, model = bias_model, z = redshift, mdef = mdef)
        #nM len(M) Mpc/h^-3
        if type(dndlgM)==int:
            dndlgM=mass_function.massFunction(10**lgM_bins, z = redshift, mdef = mdef,
                                                  model = hmf_model, q_out = 'dndlnM')*np.log(10) #len(M)
        
        #pk_lin  len(k)
        if type(pk_lin)==int:
            pk_lin = self.cal_pk_lin(redshift, lgk_max, lgk_min, lgk_nbin)
        
        #wk'
        wk_pe = np.zeros((lgM_nbin, lgk_nbin))
        wk_m  = np.zeros((lgM_nbin, lgk_nbin))
        for i in (range(lgM_nbin)):

            R500 = mass_so.M_to_R(M = 10**lgM_bins[i], z = redshift, mdef = mdef) # kpc/h physical
            r = x*R500/1000*(1+redshift)  # Mpc/h comoving
            
            rho_pe = self.pressure_profile_gNFW(x=x, M500=10**lgM_bins[i], z=redshift, model=gnfw_model, mass_bias=mass_bias, alpha_p=alpha_p, R_cut=R_cut)
            rho_m  = self.dm_profile_NFW(r=r, M=10**lgM_bins[i], z = redshift, mdef = mdef) #M_sun/h / (Mpc/h)^3
            
            
            
            wk_pe[i,:] = self.rho_r_2_W_k(r=r, rho=rho_pe, k=k_bins) #mev/cm^3 Mpc/h^3
            wk_m[i,:]  = self.rho_r_2_W_k(r=r, rho=rho_m,  k=k_bins) #M_sun/h
        
        
        #rho_M 
        rho_m_mean = cosmo.rho_m(z=redshift)*10**9/(1+redshift)**3  #M_sun/h / (Mpc/h)^3 como
        extra_term = 1-np.sum(wk_m[:,0]*dndlgM*bh, axis=0)*dlgM/rho_m_mean
        
        Pk_mm_2h = pk_lin*(np.sum(wk_m*(dndlgM*bh).reshape(-1,1), axis=0)*dlgM/rho_m_mean+extra_term)**2
        Pk_mm_1h = np.sum(wk_m**2*dndlgM.reshape(-1,1), axis=0)*dlgM/rho_m_mean**2
        
     
        #Pe_mean
        Pe_sum=wk_pe[:,0] 
        Pe_mean = np.sum(Pe_sum*dndlgM)*dlgM #mev/cm3
            
        
        byk = np.sum(wk_pe*(dndlgM*bh).reshape(-1,1), axis=0)*dlgM/Pe_mean
        
        Pk_pepe_2h =byk**2*pk_lin
        Pk_pepe_1h = np.sum(wk_pe**2*dndlgM.reshape(-1,1), axis=0)*dlgM/Pe_mean**2
        
        #cross
        Pk_mpe_1h = np.sum(wk_pe*wk_m*dndlgM.reshape(-1,1), axis=0)*dlgM/Pe_mean/rho_m_mean
        Pk_mpe_2h = pk_lin*(np.sum(wk_m*(dndlgM*bh).reshape(-1,1), axis=0)*dlgM/rho_m_mean+extra_term)*byk
            
        
        self.k = k_bins
        self.Pk_pepe_1h = Pk_pepe_1h
        self.Pk_pepe_2h = Pk_pepe_2h
        self.Pk_pepe = Pk_pepe_1h+Pk_pepe_2h
        self.Pe_mean = Pe_mean
        self.pk_lin = pk_lin
        
        self.Pk_mm_1h = Pk_mm_1h
        self.Pk_mm_2h = Pk_mm_2h
        self.Pk_mm = Pk_mm_1h+Pk_mm_2h
        
        self.Pk_mpe_1h = Pk_mpe_1h
        self.Pk_mpe_2h = Pk_mpe_2h
        self.Pk_mpe = Pk_mpe_1h+Pk_mpe_2h
            
    def cal_pk_lin(self, redshift, lgk_max, lgk_min, lgk_nbin):
        #pk_lin  len(k)
        cosmo = cosmology.setCosmology('planck15');
        # linear matter power spectrum from camb
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=cosmo.h*100, ombh2=cosmo.h**2*cosmo.Ob0, omch2=cosmo.h**2*(cosmo.Om0-cosmo.Ob0))
        pars.InitPower.set_params(ns=0.965)
        pars.set_matter_power(redshifts=[redshift], kmax=10**lgk_max+1)
        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        kh, z, pk_lin = results.get_matter_power_spectrum(minkh=10**lgk_min, maxkh=10**lgk_max, npoints = lgk_nbin)
        pk_lin=pk_lin[0]
        return pk_lin
    
    def dm_profile_NFW(self, r, M, z, mdef = "500c"): #r Mpc/h comoving
        '''
        r: Mpc/h comoving

        return p_nfw 𝑀⊙ℎ2/Mpc3
        '''
        #concentration
        c = concentration.concentration(M = M, mdef=mdef, z=z)
        p = profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef)
        p_nfw = p.density(r*1000/(1+z))*10**9/(1+z)**3 #M_sun/h / (Mpc/h)^3 como


        R500 = mass_so.M_to_R(M = M, z = z, mdef = mdef)/1000*(1+z) # Mpc/h como

        p_nfw[np.where(r>R500)]=0

        return p_nfw 
    
    def pressure_profile_gNFW(self, x, M500, z, model, mass_bias, alpha_p, R_cut):
        '''
        x = r/R500 np.array
        M500: M_sun/h 
        z: redshift 
        mass_bias: def=0
        model: "Planck13", "A10", "D23"

        return 
        Pe: meV/cm^3, same dimensions as x
        '''
        cosmo = cosmology.setCosmology('planck15');
        h_70 = cosmo.H0/70
        if model == "Planck13": # 62 SZ clusters
            P0=6.41
            gamma=0.31
            c500=1.81
            alpha=1.33
            beta=4.13
        if model == "A10": # arnaud, 2010; 33 X-ray clusters
            P0=8.403*h_70**-3/2
            gamma=0.3081
            c500=1.177
            alpha=1.051
            beta=5.4905
        if model == "D23": #denis, 2023; SZ stacking
            P0=5.9
            gamma=0.31
            c500=2
            alpha=1.8
            beta=4.9

        #P_500
        E_z = (cosmo.Om0*(1+z)**3+1-cosmo.Om0)**(1./2)
        P_500 = (1-mass_bias)*1.65*10**-3*E_z**(8/3.)*(M500/(3*10**14*0.7))**(2/3+alpha_p)*h_70**2   #keV/cm^3


        #P_x
        P_x = P0/((c500*x)**gamma * (1+(c500*x)**alpha)**((beta-gamma)/alpha))
        
        P_x[np.where(x>R_cut)]=0


        return P_500 * P_x *10**6 #mev/cm3      
    def rho_r_2_W_k(self, r, rho, k):
        '''
        r: Mpc/h comoving lg-space
        rho: as a function of r, of mass M
        k: h/Mpc coming

        return W'    w*rho
        '''
        dlnr = np.log(r)[1]-np.log(r)[0]
        kr=r.reshape(-1,1)*k.reshape(1,-1) #len(r), len(k)
        r=r.reshape(-1,1)
        rho = rho.reshape(-1,1)
        W=4*np.pi*np.sum(r**3*np.sin(kr)/kr*rho,axis=0)*dlnr

        return W #len(k)
 
    
