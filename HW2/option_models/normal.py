# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017

@author: jaehyuk
"""
import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt

def normal_formula(strike, spot, vol, texp, intr=0.0, divr=0.0, cp_sign=1):
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac

    if( texp<0 or vol*np.sqrt(texp)<1e-8 ):
        return disc_fac * np.fmax( cp_sign*(forward-strike), 0 )

    vol_std = np.fmax(vol * np.sqrt(texp), 1.0e-16)
    d = (forward - strike) / vol_std

    price = disc_fac * (cp_sign * (forward - strike) * ss.norm.cdf(cp_sign * d) + vol_std * ss.norm.pdf(d))
    return price

class NormalModel:
    
    vol, intr, divr = None, None, None
    
    def __init__(self, vol, intr=0, divr=0):
        self.vol = vol
        self.intr = intr
        self.divr = divr
    
    def price(self, strike, spot, texp, cp_sign=1):
        return normal_formula(strike, spot, self.vol, texp, intr=self.intr, divr=self.divr, cp_sign=cp_sign)
    
    def delta(self, strike, spot, texp, cp_sign=1):
        ''' 
        c_price = cp_sign*[S0*e^(-qT)-K*e^(-rT)]*N(cp_sign*d)+e^(-rT)*σT^0.5*n(d), d = (F0-K)/(σT^0.5), F0 = S0*e^[(r-q)T]
        Formula: delta = cp_sign*e^(-qT)*N(cp_sign*d), d = (F0-K)/(σT^0.5), F0 = S0*e^[(r-q)T]
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot/disc_fac*div_fac
        vol_std = self.vol*np.sqrt(texp)
        d = (forward-strike)/vol_std  #N(d)+dn(d)
        delta_ana = div_fac*cp_sign*ss.norm().cdf(cp_sign*d)
        # numerical check
        # spot1,spot2 = spot*0.999,spot*1.001
        # P1 = self.price(strike, spot1, texp, cp_sign=cp_sign)
        # P2 = self.price(strike, spot2, texp, cp_sign=cp_sign)
        # delta_num = (P1-P2)/(spot1-spot2)
        # print(delta_ana,delta_num,delta_ana-delta_num)
        return delta_ana

    def vega(self, strike, spot, texp, cp_sign=1):
        ''' 
        Formula: vega = n(d)*T^0.5*e^(-rT), d = (F0-K)/(σT^0.5), F0 = S0*e^[(r-q)T]
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot/disc_fac*div_fac
        vol_std = self.vol*np.sqrt(texp)
        d = (forward-strike)/vol_std
        vega_ana = ss.norm().pdf(d)*np.sqrt(texp)*disc_fac
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
        Formula: gamma = n(d)*e^[(r-2q)T)]/(σ*T^0.5), d = (F0-K)/(σT^0.5), F0 = S0*e^[(r-q)T]
        '''
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot/disc_fac*div_fac
        vol_std = self.vol*np.sqrt(texp)
        d = (forward-strike)/vol_std
        gamma_ana = ss.norm().pdf(d)*np.square(div_fac)/(disc_fac*vol_std)
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
        iv_func = lambda _vol: normal_formula(strike, spot, _vol, texp, self.intr, self.divr, cp_sign) - price
        vol = sopt.brentq(iv_func, 0, 10)
        return vol