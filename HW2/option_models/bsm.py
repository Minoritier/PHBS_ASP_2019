    # -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt

def bsm_formula(strike, spot, vol, texp, intr=0.0, divr=0.0, cp_sign=1):
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac

    if( texp<0 or vol*np.sqrt(texp)<1e-8 ):
        return disc_fac * np.fmax( cp_sign*(forward-strike), 0 )
    
    vol_std = vol*np.sqrt(texp)
    d1 = np.log(forward/strike)/vol_std + 0.5*vol_std
    d2 = d1 - vol_std

    price = cp_sign * disc_fac \
        * ( forward * ss.norm.cdf(cp_sign*d1) - strike * ss.norm.cdf(cp_sign*d2) )
    return price

class BsmModel:
    vol, intr, divr = None, None, None
    
    def __init__(self, vol, intr=0, divr=0):
        self.vol = vol
        self.intr = intr
        self.divr = divr
    
    def price(self, strike, spot, texp, cp_sign=1):
        return bsm_formula(strike, spot, self.vol, texp, intr=self.intr, divr=self.divr, cp_sign=cp_sign)
    
    def delta(self, strike, spot, texp, cp_sign=1):
        ''' 
        Formula: delta = e^(-qT)*cp_sign*N(cp_sign*d1), d1 = ln(F0/K)/(σ*T^0.5)+0.5*σ*T^0.5, F0 = S0*e^[(r-q)T]
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot/disc_fac*div_fac
        vol_std = self.vol*np.sqrt(texp)    
        d1 = np.log(forward/strike)/vol_std+0.5*vol_std
        delta_ana = div_fac*(cp_sign*ss.norm().cdf(cp_sign*d1))
        # numerical check
        # spot1,spot2 = spot*0.999,spot*1.001
        # P1 = self.price(strike, spot1, texp, cp_sign=cp_sign)
        # P2 = self.price(strike, spot2, texp, cp_sign=cp_sign)
        # delta_num = (P1-P2)/(spot1-spot2)
        # print(delta_ana,delta_num,delta_ana-delta_num)
        return delta_ana

    def vega(self, strike, spot, texp, cp_sign=1):
        ''' 
        Formula: vega = e^(-qT)*S0*T^0.5*n(d1), d1 = ln(F0/K)/(σ*T^0.5)+0.5*σ*T^0.5, F0 = S0*e^[(r-q)T]
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot/disc_fac*div_fac
        vol_std = self.vol*np.sqrt(texp)    
        d1 = np.log(forward/strike)/vol_std+0.5*vol_std
        vega_ana = div_fac*spot*np.sqrt(texp)*ss.norm().pdf(d1)
        # numerical check
        # vol1,vol2,vol = self.vol*0.999,self.vol*1.001,self.vol
        # self.vol = vol1
        # P1 = self.price(strike, spot, texp, cp_sign=cp_sign)
        # self.vol = vol2
        # P2 = self.price(strike, spot, texp, cp_sign=cp_sign)
        # self.vol = vol
        # vega_num = (P1-P2)/(vol1-vol2)
        # print(vega_ana,vega_num,vega_ana-vega_num)
        return vega_ana

    def gamma(self, strike, spot, texp, cp_sign=1):
        ''' 
        Formula: gamma = e^(-qT)*n(d1)/(S0*σ*T^0.5), d1 = ln(F0/K)/(σ*T^0.5)+0.5*σ*T^0.5, F0 = S0*e^[(r-q)T]
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot/disc_fac*div_fac
        vol_std = self.vol*np.sqrt(texp)    
        d1 = np.log(forward/strike)/vol_std+0.5*vol_std
        gamma_ana = div_fac*ss.norm().pdf(d1)/(spot*vol_std)
        # numerical check
        # spot1,spot2 = spot*0.999,spot*1.001
        # P1 = self.price(strike, spot1, texp, cp_sign=cp_sign)
        # P2 = self.price(strike, spot2, texp, cp_sign=cp_sign)
        # P3 = self.price(strike, spot, texp, cp_sign=cp_sign)
        # delta_num_1 = (P1-P3)/(spot1-spot)
        # delta_num_2 = (P3-P2)/(spot-spot2)
        # gamma_num = (delta_num_1-delta_num_2)/((spot1-spot2)/2)
        # print(gamma_ana,gamma_num,gamma_ana-gamma_num)
        return gamma_ana

    def impvol(self, price, strike, spot, texp, cp_sign=1):
        iv_func = lambda _vol: \
            bsm_formula(strike, spot, _vol, texp, self.intr, self.divr, cp_sign) - price
        vol = sopt.brentq(iv_func, 0, 10)
        return vol
