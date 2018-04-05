import os
import numpy as np

from . import __file__ as filepath

__all__ = ["Inoue14"]

class Inoue14(object):
    def __init__(self):
        """
        IGM absorption from Inoue et al. (2014)
        """
        self._load_data()
        
    def _load_data(self):
        path = os.path.join(os.path.dirname(filepath),'data')
        #print path
    
        LAF_file = os.path.join(path, 'LAFcoeff.txt')
        DLA_file = os.path.join(path, 'DLAcoeff.txt')
    
        data = np.loadtxt(LAF_file, unpack=True)
        ix, self.lam, self.ALAF1, self.ALAF2, self.ALAF3 = data
        
        data = np.loadtxt(DLA_file, unpack=True)
        ix, self.lam, self.ADLA1, self.ADLA2 = data
        
        self.NA = len(self.lam)
        
        return True
    
    def tLSLAF(self, zS, lobs):
        z1LAF = 1.2
        z2LAF = 4.7

        l2 = np.dot(self.lam[:, np.newaxis], np.ones((1, lobs.shape[0])))
        tLSLAF_value = l2.T*0
        
        match0 = (lobs < l2*(1+zS))
        match1 = lobs < l2*(1+z1LAF)
        match2 = (lobs >= l2*(1+z1LAF)) & (lobs < l2*(1+z2LAF))
        match3 = lobs >= l2*(1+z2LAF)
        
        tLSLAF_value += self.ALAF1*(((lobs/l2)*(match0 & match1))**1.2).T
        tLSLAF_value += self.ALAF2*(((lobs/l2)*(match0 & match2))**3.7).T
        tLSLAF_value += self.ALAF3*(((lobs/l2)*(match0 & match3))**5.5).T
        
        return tLSLAF_value.sum(axis=1)
    
    def tLSDLA(self, zS, lobs):
        """
        Lyman Series, DLA
        """
        z1DLA = 2.0
        
        l2 = np.dot(self.lam[:, np.newaxis], np.ones((1, lobs.shape[0])))
        tLSDLA_value = l2.T*0
        
        match0 = (lobs < l2*(1+zS)) 
        match1 = lobs < l2*(1.+z1DLA)
        
        tLSDLA_value += self.ADLA1*((lobs/l2*(match0 & match1))**2.0).T
        tLSDLA_value += self.ADLA2*((lobs/l2*(match0 & ~match1))**3.0).T
                
        return tLSDLA_value.sum(axis=1)
    
    def _tLSLAF(self, zS, lobs):
        """
        Lyman series, Lyman-alpha forest
        """
        z1LAF = 1.2
        z2LAF = 4.7
        
        tLSLAF_value = lobs*0.
                
        for j in range(self.NA):
            match0 = (lobs < self.lam[j]*(1+zS)) #& (lobs > self.lam[j])
            match1 = lobs < self.lam[j]*(1+z1LAF)
            match2 = (lobs >= self.lam[j]*(1+z1LAF)) & (lobs < self.lam[j]*(1+z2LAF))
            match3 = lobs >= self.lam[j]*(1+z2LAF)
            
            tLSLAF_value[match0 & match1] += self.ALAF1[j]*(lobs[match0 & match1]/self.lam[j])**1.2
            tLSLAF_value[match0 & match2] += self.ALAF2[j]*(lobs[match0 & match2]/self.lam[j])**3.7
            tLSLAF_value[match0 & match3] += self.ALAF3[j]*(lobs[match0 & match3]/self.lam[j])**5.5
        
        return tLSLAF_value
        
    def _tLSDLA(self, zS, lobs):
        """
        Lyman Series, DLA
        """
        z1DLA = 2.0
        tLSDLA_value = lobs*0.
        
        for j in range(self.NA):
            match0 = (lobs < self.lam[j]*(1+zS)) #& (lobs > self.lam[j])
            match1 = lobs < self.lam[j]*(1.+z1DLA)
            tLSDLA_value[match0 & match1] += self.ADLA1[j]*(lobs[match0 & match1]/self.lam[j])**2.0
            tLSDLA_value[match0 & ~match1] += self.ADLA2[j]*(lobs[match0 & ~match1]/self.lam[j])**3.0
        
        return tLSDLA_value
    
    def tLCDLA(self, zS, lobs):
        """
        Lyman continuum, DLA
        """
        z1DLA = 2.0
        lamL = 911.8
        
        tLCDLA_value = lobs*0.
        
        match0 = lobs < lamL*(1.+zS)
        if zS < z1DLA:
            tLCDLA_value[match0] = 0.2113 * _pow(1.0+zS, 2) - 0.07661 * _pow(1.0+zS, 2.3) * _pow(lobs[match0]/lamL, (-3e-1)) - 0.1347 * _pow(lobs[match0]/lamL, 2)
        else:
            match1 = lobs >= lamL*(1.+z1DLA)
            
            tLCDLA_value[match0 & match1] = 0.04696 * _pow(1.0+zS, 3) - 0.01779 * _pow(1.0+zS, 3.3) * _pow(lobs[match0 & match1]/lamL, (-3e-1)) - 0.02916 * _pow(lobs[match0 & match1]/lamL, 3)
            tLCDLA_value[match0 & ~match1] =0.6340 + 0.04696 * _pow(1.0+zS, 3) - 0.01779 * _pow(1.0+zS, 3.3) * _pow(lobs[match0 & ~match1]/lamL, (-3e-1)) - 0.1347 * _pow(lobs[match0 & ~match1]/lamL, 2) - 0.2905 * _pow(lobs[match0 & ~match1]/lamL, (-3e-1))
        
        return tLCDLA_value
        
    def tLCLAF(self, zS, lobs):
        """
        Lyman continuum, LAF
        """
        z1LAF = 1.2
        z2LAF = 4.7
        lamL = 911.8

        tLCLAF_value = lobs*0.
        
        match0 = lobs < lamL*(1.+zS)
        
        if zS < z1LAF:
            tLCLAF_value[match0] = 0.3248 * (_pow(lobs[match0]/lamL, 1.2) - _pow(1.0+zS, -9e-1) * _pow(lobs[match0]/lamL, 2.1))
        elif zS < z2LAF:
            match1 = lobs >= lamL*(1+z1LAF)
            tLCLAF_value[match0 & match1] = 2.545e-2 * (_pow(1.0+zS, 1.6) * _pow(lobs[match0 & match1]/lamL, 2.1) - _pow(lobs[match0 & match1]/lamL, 3.7))
            tLCLAF_value[match0 & ~match1] = 2.545e-2 * _pow(1.0+zS, 1.6) * _pow(lobs[match0 & ~match1]/lamL, 2.1) + 0.3248 * _pow(lobs[match0 & ~match1]/lamL, 1.2) - 0.2496 * _pow(lobs[match0 & ~match1]/lamL, 2.1)
        else:
            match1 = lobs > lamL*(1.+z2LAF)
            match2 = (lobs >= lamL*(1.+z1LAF)) & (lobs < lamL*(1.+z2LAF))
            match3 = lobs < lamL*(1.+z1LAF)
            
            tLCLAF_value[match0 & match1] = 5.221e-4 * (_pow(1.0+zS, 3.4) * _pow(lobs[match0 & match1]/lamL, 2.1) - _pow(lobs[match0 & match1]/lamL, 5.5))
            tLCLAF_value[match0 & match2] = 5.221e-4 * _pow(1.0+zS, 3.4) * _pow(lobs[match0 & match2]/lamL, 2.1) + 0.2182 * _pow(lobs[match0 & match2]/lamL, 2.1) - 2.545e-2 * _pow(lobs[match0 & match2]/lamL, 3.7)
            tLCLAF_value[match0 & match3] = 5.221e-4 * _pow(1.0+zS, 3.4) * _pow(lobs[match0 & match3]/lamL, 2.1) + 0.3248 * _pow(lobs[match0 & match3]/lamL, 1.2) - 3.140e-2 * _pow(lobs[match0 & match3]/lamL, 2.1)
            
        return tLCLAF_value
        
    def full_IGM(self, z, lobs):
        """Get full Inoue IGM absorption
        
        Parameters
        ----------
        z : float
            Redshift to evaluate IGM absorption
        
        lobs : array-like
            Observed-frame wavelength(s).
        
        Returns
        -------
        abs : IGM absorption
        
        """
        tau_LS = self.tLSLAF(z, lobs) + self.tLSDLA(z, lobs)
        tau_LC = self.tLCLAF(z, lobs) + self.tLCDLA(z, lobs)
        
        ### Upturn at short wavelengths, low-z
        #k = 1./100
        #l0 = 600-6/k
        #clip = lobs/(1+z) < 600.
        #tau_clip = 100*(1-1./(1+np.exp(-k*(lobs/(1+z)-l0))))
        tau_clip = 0.
        
        return np.exp(-(tau_LC + tau_LS + tau_clip))
    
    def build_grid(self, zgrid, lrest):
        """Build a spline interpolation object for fast IGM models
        
        Returns: self.interpolate
        """
        
        from scipy.interpolate import CubicSpline
        igm_grid = np.zeros((len(zgrid), len(lrest)))
        for iz in range(len(zgrid)):
            igm_grid[iz,:] = self.full_IGM(zgrid[iz], lrest*(1+zgrid[iz]))
        
        self.interpolate = CubicSpline(zgrid, igm_grid)
        
def _pow(a, b):
    """C-like power, a**b
    """
    return a**b
    
