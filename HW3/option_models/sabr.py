
    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm

'''
Asymptotic approximation for 0<beta<=1 by Hagan
'''
def bsm_vol(strike, forward, texp, sigma, alpha=0, rho=0, beta=1):
    if(texp<=0.0):
        return( 0.0 )

    powFwdStrk = (forward*strike)**((1-beta)/2)
    logFwdStrk = np.log(forward/strike)
    logFwdStrk2 = logFwdStrk**2
    rho2 = rho*rho

    pre1 = powFwdStrk*( 1 + (1-beta)**2/24 * logFwdStrk2*(1 + (1-beta)**2/80 * logFwdStrk2) )
  
    pre2alp0 = (2-3*rho2)*alpha**2/24
    pre2alp1 = alpha*rho*beta/4/powFwdStrk
    pre2alp2 = (1-beta)**2/24/powFwdStrk**2

    pre2 = 1 + texp*( pre2alp0 + sigma*(pre2alp1 + pre2alp2*sigma) )

    zz = powFwdStrk*logFwdStrk*alpha/np.fmax(sigma, 1e-32)  # need to make sure sig > 0
    if isinstance(zz, float):
        zz = np.array([zz])
    yy = np.sqrt(1 + zz*(zz-2*rho))

    xx_zz = np.zeros(zz.size)

    ind = np.where(abs(zz) < 1e-5)
    xx_zz[ind] = 1 + (rho/2)*zz[ind] + (1/2*rho2-1/6)*zz[ind]**2 + 1/8*(5*rho2-3)*rho*zz[ind]**3
    ind = np.where(zz >= 1e-5)
    xx_zz[ind] = np.log( (yy[[ind]] + (zz[ind]-rho))/(1-rho) ) / zz[ind]
    ind = np.where(zz <= -1e-5)
    xx_zz[ind] = np.log( (1+rho)/(yy[ind] - (zz[ind]-rho)) ) / zz[ind]

    bsmvol = sigma*pre2/(pre1*xx_zz) # bsm vol
    return(bsmvol[0] if bsmvol.size==1 else bsmvol)

'''
Asymptotic approximation for beta=0 by Hagan
'''
def norm_vol(strike, forward, texp, sigma, alpha=0, rho=0):
    # forward, spot, sigma may be either scalar or np.array. 
    # texp, alpha, rho, beta should be scholar values

    if(texp<=0.0):
        return( 0.0 )
    
    zeta = (forward - strike)*alpha/np.fmax(sigma, 1e-32)
    # explicitly make np.array even if args are all scalar or list
    if isinstance(zeta, float):
        zeta = np.array([zeta])
        
    yy = np.sqrt(1 + zeta*(zeta - 2*rho))
    chi_zeta = np.zeros(zeta.size)
    
    rho2 = rho*rho
    ind = np.where(abs(zeta) < 1e-5)
    chi_zeta[ind] = 1 + 0.5*rho*zeta[ind] + (0.5*rho2 - 1/6)*zeta[ind]**2 + 1/8*(5*rho2-3)*rho*zeta[ind]**3

    ind = np.where(zeta >= 1e-5)
    chi_zeta[ind] = np.log( (yy[ind] + (zeta[ind] - rho))/(1-rho) ) / zeta[ind]

    ind = np.where(zeta <= -1e-5)
    chi_zeta[ind] = np.log( (1+rho)/(yy[ind] - (zeta[ind] - rho)) ) / zeta[ind]

    nvol = sigma * (1 + (2-3*rho2)/24*alpha**2*texp) / chi_zeta
 
    return(nvol[0] if nvol.size==1 else nvol)

'''
Hagan model class for 0<beta<=1
'''
class ModelHagan:
    alpha, beta, rho = 0.0, 1.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.beta = beta
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return bsm_vol(strike, forward, texp, sigma, alpha=self.alpha, beta=self.beta, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        bsm_vol = self.bsm_vol(strike, spot, texp, sigma)
        return self.bsm_model.price(strike, spot, texp, bsm_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho, beta=self.beta) - vol  # A BUG HERE!!
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        return sigma
    
    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or bsm vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp = self.texp if (texp==None) else texp
        if is_vol == False:
            vol3 = [self.bsm_model.impvol(price_or_vol3[i],strike3[i],spot,texp,cp_sign) for i in range(3)]
        else:
            vol3 = price_or_vol3
        forward = spot*np.exp(texp*(self.intr-self.divr))
        
        _func = lambda _sar: bsm_vol(strike3,forward,texp,np.exp(_sar[0]),np.exp(_sar[1]),np.tanh(_sar[2]),self.beta)-vol3
        # the initial guess of sigma and alpha should be large and that of rho should be 0 to have large gradient
        sar = sopt.root(_func,[10,10,0]).x
        sigma,alpha,rho = np.exp(sar[0]),np.exp(sar[1]),np.tanh(sar[2])
        if setval == True:
            self.sigma,self.alpha,self.rho = sigma,alpha,rho
        return sigma,alpha,rho # sigma, alpha, rho

'''
Hagan model class for beta=0
'''
class ModelNormalHagan:
    alpha, beta, rho = 0.0, 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.beta = 0.0 # not used but put it here
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return norm_vol(strike, forward, texp, sigma, alpha=self.alpha, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        n_vol = self.norm_vol(strike, spot, texp, sigma)
        return self.normal_model.price(strike, spot, texp, n_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            norm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 50)
        if(setval):
            self.sigma = sigma
        return sigma

    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or normal vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp = self.texp if (texp==None) else texp
        if is_vol == False:
            vol3 = [self.normal_model.impvol(price_or_vol3[i],strike3[i],spot,texp,cp_sign) for i in range(3)]
        else:
            vol3 = price_or_vol3
        forward = spot*np.exp(texp*(self.intr-self.divr))

        _func = lambda _sar: norm_vol(strike3,forward,texp,np.exp(_sar[0]),np.exp(_sar[1]),np.tanh(_sar[2]))-vol3
        # initial guess of sigma and alpha should be large and that of rho should be 0 to have large gradient
        sar = sopt.root(_func,[10,10,0]).x
        sigma,alpha,rho = np.exp(sar[0]),np.exp(sar[1]),np.tanh(sar[2])
        if setval == True:
            self.sigma,self.alpha,self.rho = sigma,alpha,rho
        return sigma,alpha,rho # sigma, alpha, rho

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    step_num = 100
    path_num = 10000
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0, step_num=100, path_num=10000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.step_num = step_num
        self.path_num = path_num
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        if type(strike) in {float,int}:
        	price = self.price(strike,spot,texp,sigma,cp_sign=1)
        	imp_vol = self.bsm_model.impvol(price,strike,spot,texp,cp_sign=1)
        elif type(strike) == np.ndarray and len(strike.shape)==1:
        	imp_vol = []
        	for _strike in strike:
        		_price = self.price(_strike,spot,texp,sigma,cp_sign=1)
        		_imp_vol = self.bsm_model.impvol(_price,_strike,spot,texp,cp_sign=1)
        		imp_vol.append(_imp_vol)
        	imp_vol = np.array(imp_vol)
        else:
            print("Strike must be an 1-d array or a float or an int")
            return 0
        return imp_vol
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1, rdm_sd=12345):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if rdm_sd != None:
	        np.random.seed(rdm_sd)
        texp = self.texp if (texp==None) else texp
        sigma = self.sigma if (sigma==None) else sigma
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot*div_fac/disc_fac
        path_num = self.path_num
        step_num = self.step_num
        step_len = texp/step_num

        Z = np.random.randn(path_num,step_num)
        Z_k = np.concatenate([np.zeros((path_num,1)),Z.cumsum(axis=1)],axis=1)
        k = np.array([np.arange(step_num+1)]*path_num)
        sigma_k = sigma*np.exp(self.alpha*np.sqrt(step_len)*Z_k-0.5*np.square(self.alpha)*step_len*k)
        W = self.rho*Z+np.sqrt(1-self.rho**2)*np.random.randn(path_num,step_num)
        ln_F_k = (sigma_k[:,:-1]*W*np.sqrt(step_len)-0.5*np.square(sigma_k[:,:-1])*step_len).cumsum(axis=1)
        ln_F_k = np.concatenate([np.zeros((path_num,1)),ln_F_k],axis=1)+np.log(forward)
        F_k = np.exp(ln_F_k)

        prices = np.fmax((F_k[:,-1].reshape(-1,1)-strike)*cp_sign,0).mean(axis=0)

        return prices*disc_fac

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    step_num = 100
    path_num = 10000
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0, step_num=100, path_num=10000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.step_num = step_num
        self.path_num = path_num
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        if type(strike) in {float,int}:
        	price = self.price(strike,spot,texp,sigma,cp_sign=1)
        	imp_vol = self.normal_model.impvol(price,strike,spot,texp,cp_sign=1)
        elif type(strike) == np.ndarray and len(strike.shape)==1:
        	imp_vol = []
        	for _strike in strike:
        		_price = self.price(_strike,spot,texp,sigma,cp_sign=1)
        		_imp_vol = self.normal_model.impvol(_price,_strike,spot,texp,cp_sign=1)
        		imp_vol.append(_imp_vol)
        	imp_vol = np.array(imp_vol)
        else:
            print("Strike must be an 1-d array or a float or an int")
            return 0
        return imp_vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1, rdm_sd=12345):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if rdm_sd != None:
	        np.random.seed(rdm_sd)
        texp = self.texp if (texp==None) else texp
        sigma = self.sigma if (sigma==None) else sigma
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot*div_fac/disc_fac
        path_num = self.path_num
        step_num = self.step_num
        step_len = texp/step_num

        Z = np.random.randn(path_num,step_num)
        Z_k = np.concatenate([np.zeros((path_num,1)),Z.cumsum(axis=1)],axis=1)
        k = np.array([np.arange(step_num+1)]*path_num)
        sigma_k = sigma*np.exp(self.alpha*np.sqrt(step_len)*Z_k-0.5*np.square(self.alpha)*step_len*k)
        W = self.rho*Z+np.sqrt(1-self.rho**2)*np.random.randn(path_num,step_num)
        F_k = (sigma_k[:,:-1]*W*np.sqrt(step_len)).cumsum(axis=1)
        F_k = np.concatenate([np.zeros((path_num,1)),F_k],axis=1)+forward

        prices = np.fmax((F_k[:,-1].reshape(-1,1)-strike)*cp_sign,0).mean(axis=0)

        return prices*disc_fac

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    step_num = 100
    path_num = 10000
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0, step_num=100, path_num=10000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.step_num = step_num
        self.path_num = path_num
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        if type(strike) in {float,int}:
        	price = self.price(strike,spot,texp,sigma,cp_sign=1)
        	imp_vol = self.bsm_model.impvol(price,strike,spot,texp,cp_sign=1)
        elif type(strike) == np.ndarray and len(strike.shape)==1:
        	imp_vol = []
        	for _strike in strike:
        		_price = self.price(_strike,spot,texp,sigma,cp_sign=1)
        		_imp_vol = self.bsm_model.impvol(_price,_strike,spot,texp,cp_sign=1)
        		imp_vol.append(_imp_vol)
        	imp_vol = np.array(imp_vol)
        else:
            print("Strike must be an 1-d array or a float or an int")
            return 0
        return imp_vol
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1, rule="S", rdm_sd=12345):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if rdm_sd != None:
	        np.random.seed(rdm_sd)
        texp = self.texp if (texp==None) else texp
        sigma = self.sigma if (sigma==None) else sigma
        path_num = self.path_num
        step_num = self.step_num+1 if (rule == "S" and self.step_num%2==1) else self.step_num
        step_len = texp/step_num

        Z_k = np.random.randn(path_num,step_num).cumsum(axis=1)
        Z_k = np.concatenate([np.zeros((path_num,1)),Z_k],axis=1)
        k = np.array([np.arange(step_num+1)]*path_num)
        sigma_k = sigma*np.exp(self.alpha*np.sqrt(step_len)*Z_k-0.5*np.square(self.alpha)*step_len*k)

        if rule == "T": # trapezoidal rule 
            _coef = np.ones((path_num,step_num+1))*2
            _coef[:,0],_coef[:,-1] = 1,1
            I_T = (_coef*np.square(sigma_k)).sum(axis=1)*(step_len/2)
        if rule == "S": # Simpson’s rule
            _coef = (k%2)*2+2
            _coef[:,0],_coef[:,-1] = 1,1
            I_T = (_coef*np.square(sigma_k)).sum(axis=1)*(step_len/3)
        else:
            I_T = (np.square(sigma_k)[:,1:]).sum(axis=1)*step_len
            # I_T = (np.square(sigma_k)[:,:-1]).sum(axis=1)*step_len

        spot_n = spot*np.exp(self.rho/self.alpha*(sigma_k[:,-1]-sigma)-0.5*np.square(self.rho)*I_T).reshape(-1,1)
        vol_n = np.sqrt((1-self.rho**2)*I_T/texp).reshape(-1,1) # with shape (num_path,1)
        prices = self.bsm_model.price(strike,spot_n,texp,vol_n,cp_sign).mean(axis=0)

        return prices

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    step_num = 100
    path_num = 10000
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0, step_num=100, path_num=10000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.step_num = step_num
        self.path_num = path_num
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        if type(strike) in {float,int}:
        	price = self.price(strike,spot,texp,sigma,cp_sign=1)
        	imp_vol = self.normal_model.impvol(price,strike,spot,texp,cp_sign=1)
        elif type(strike) == np.ndarray and len(strike.shape)==1:
        	imp_vol = []
        	for _strike in strike:
        		_price = self.price(_strike,spot,texp,sigma,cp_sign=1)
        		_imp_vol = self.normal_model.impvol(_price,_strike,spot,texp,cp_sign=1)
        		imp_vol.append(_imp_vol)
        	imp_vol = np.array(imp_vol)
        else:
            print("Strike must be an 1-d array or a float or an int")
            return 0
        return imp_vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1, rule="S", rdm_sd=12345):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        if rdm_sd != None:
	        np.random.seed(rdm_sd)
        texp = self.texp if (texp==None) else texp
        sigma = self.sigma if (sigma==None) else sigma
        path_num = self.path_num
        step_num = self.step_num+1 if (rule == "S" and self.step_num%2==1) else self.step_num
        step_len = texp/step_num

        Z_k = np.random.randn(path_num,step_num).cumsum(axis=1)
        Z_k = np.concatenate([np.zeros((path_num,1)),Z_k],axis=1)
        k = np.array([np.arange(step_num+1)]*path_num)
        sigma_k = sigma*np.exp(self.alpha*np.sqrt(step_len)*Z_k-0.5*np.square(self.alpha)*step_len*k)

        if rule == "T": # trapezoidal rule 
            _coef = np.ones((path_num,step_num+1))*2
            _coef[:,0],_coef[:,-1] = 1,1
            I_T = (_coef*np.square(sigma_k)).sum(axis=1)*(step_len/2)
        if rule == "S": # Simpson’s rule
            _coef = (k%2)*2+2
            _coef[:,0],_coef[:,-1] = 1,1
            I_T = (_coef*np.square(sigma_k)).sum(axis=1)*(step_len/3)
        else:
            I_T = (np.square(sigma_k)[:,1:]).sum(axis=1)*step_len
            # I_T = (np.square(sigma_k)[:,:-1]).sum(axis=1)*step_len

        spot_n = spot+self.rho/self.alpha*(sigma_k[:,-1]-sigma).reshape(-1,1)  # the only change
        vol_n = np.sqrt((1-self.rho**2)*I_T/texp).reshape(-1,1)  # with shape (num_path,1)
        prices = self.normal_model.price(strike,spot_n,texp,vol_n,cp_sign).mean(axis=0)

        return prices