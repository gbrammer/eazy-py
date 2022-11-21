"""
Tools for making FSPS templates
"""
import os
from collections import OrderedDict

import numpy as np
import astropy.units as u
from astropy.cosmology import WMAP9

FLAM_CGS = u.erg/u.second/u.cm**2/u.Angstrom
LINE_CGS = 1.e-17*u.erg/u.second/u.cm**2

try:
    from dust_attenuation.baseclasses import BaseAttAvModel
except:
    BaseAttAvModel = object

from astropy.modeling import Parameter
import astropy.units as u

try:
    from fsps import StellarPopulation
except:
    # Broken, but imports
    StellarPopulation = object

from . import utils
from . import templates

DEFAULT_LABEL = 'fsps_tau{tau:3.1f}_logz{logzsol:4.2f}_tage{tage:4.2f}_av{Av:4.2f}'

WG00_DEFAULTS = dict(geometry='shell', dust_type='mw', 
                   dust_distribution='homogeneous')

__all__ = ["Zafar15", "ExtinctionModel", "SMC", "Reddy15", "KC13",
           "ParameterizedWG00", "ExtendedFsps", "fsps_line_info", 
           "wuyts_line_Av"]


class ArrayExtCurve(BaseAttAvModel):
    """
    Alam interpolated from arrays
    """
    name = 'Array'
    #bump_ampl = 1.
        
    Rv = 2.21 # err 0.22
    
    xarray = np.arange(0.09, 2.2, 0.01)
    yarray = xarray*0.+1
    left=None
    right=None
    
    def Alam(self, mu):
        """
        klam, eq. 1
        """
        Alam = np.interp(mu, self.xarray, self.yarray, 
                         left=self.left, right=self.right)
        return Alam
            
    def evaluate(self, x, Av):
       
        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.micron
        else:
            xin = np.atleast_1d(x)
        
        mu = xin.to(u.micron).value

        alam = self.Alam(mu) #*self.Rv
        return np.maximum(alam*Av, 0.)


class Zafar15(BaseAttAvModel):
    """
    Quasar extinction curve from Zafar et al. (2015)
        
    https://ui.adsabs.harvard.edu/abs/2015A%26A...584A.100Z/abstract
    """
    name = 'Zafar+15'
    #bump_ampl = 1.
        
    Rv = 2.21 # err 0.22
    
    @staticmethod
    def Alam(mu, Rv):
        """
        klam, eq. 1
        """
        x = 1/mu
                
        # My fit
        coeffs = np.array([0.05694421, 0.57778243, -0.12417444])
        Alam = np.polyval(coeffs, x)*2.21/Rv
        
        # Only above x > 5.90
        fuv = x > 5.90
        if fuv.sum() > 0:
            Afuv = 1/Rv*(-4.678+2.355*x + 0.622*(x-5.90)**2) + 1.
            Alam[fuv] = Afuv[fuv]
            
        return Alam
            
    def evaluate(self, x, Av):
       
        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.micron
        else:
            xin = np.atleast_1d(x)
        
        mu = xin.to(u.micron).value

        alam = self.Alam(mu, self.Rv) #*self.Rv

        # Rv = Av/EBV
        # EBV=Av/Rv
        # Ax = Alam/Av
        # 
        # klam = Alam/EBV
        # Alam = klam*EBV = klam*Av/Rv
        return np.maximum(alam*Av, 0.)

class ExtinctionModel(BaseAttAvModel):
    """
    Modify `dust_extinction.averages.G03_SMCBar` to work as Att
    """     
    #from dust_extinction.averages import G03_SMCBar
    #SMCBar = G03_SMCBar()
    curve_type = 'smc'
    init_curve = None
    
    #@property
    def _curve_model(self):
        
        if self.init_curve == self.curve_type:
            return 0
            
        if self.curve_type.upper() == 'SMC':
            from dust_extinction.averages import G03_SMCBar as curve
        elif self.curve_type.upper() == 'LMC':
            from dust_extinction.averages import G03_LMCAvg as curve
        elif self.curve_type.upper() in ['MW','F99']:
            from dust_extinction.parameter_averages import F99 as curve
        else:
            raise ValueError(f'curve_type {self.curve_type} not recognized')
            
        self.curve = curve()
        self.init_curve = self.curve_type
        
    def evaluate(self, x, Av):
        
        self._curve_model()
        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.Angstrom
        else:
            xin = np.atleast_1d(x)
        
        xinv = 1./xin.to(u.micron)

        curve = self.curve
        xr = [x for x in curve.x_range]
        xr[0] *= 1.001
        xr[1] *= 0.999
        print('xxx', xr)
        
        if 'Rv' in curve.param_names:
            klam = curve.evaluate(1/np.clip(xinv, 
                                    xr[0]/u.micron, xr[1]/u.micron), 
                                    Rv=curve.Rv)
        else:
            klam = curve.evaluate(1/np.clip(xinv, 
                                    xr[0]/u.micron, xr[1]/u.micron))
            
        return klam*Av

class SMC(BaseAttAvModel):
    """
    Modify `dust_extinction.averages.G03_SMCBar` to work as Att
    """     
    from dust_extinction.averages import G03_SMCBar
    SMCBar = G03_SMCBar()
    
    bump_ampl = Parameter(description="Amplitude of UV bump",
                      default=0., min=0., max=10.)
    
    bump_gamma = 0.04
    bump_x0 = 0.2175
    
    
    def uv_bump(self, mu, bump_ampl):
        """
        Drude profile for computing the UV bump.

        Parameters
        ----------
        x: np array (float)
           expects wavelengths in [micron]

        x0: float
           Central wavelength of the UV bump (in microns).

        gamma: float
           Width (FWHM) of the UV bump (in microns).

        ampl: float
           Amplitude of the UV bump.

        Returns
        -------
        np array (float)
           lorentzian-like Drude profile

        Raises
        ------
        ValueError
           Input x values outside of defined range

        """
        return bump_ampl * (mu**2 * self.bump_gamma**2 /
                       ((mu**2 - self.bump_x0**2)**2 + 
                         mu**2 * self.bump_gamma**2))
    
    
    def evaluate(self, x, Av, bump_ampl):

        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.Angstrom
        else:
            xin = np.atleast_1d(x)

        xinv = 1./xin.to(u.micron)

        klam = self.SMCBar.evaluate(1/np.clip(xinv, 
                                    0.301/u.micron, 9.99/u.micron))
        
        if bump_ampl > 0:
            klam += self.uv_bump(xin.to(u.micron).value, bump_ampl)
            
        return klam*Av
                

class Reddy15(BaseAttAvModel):
    """
    Attenuation curve from Reddy et al. (2015)
    
    With optional UV bump
    
    https://ui.adsabs.harvard.edu/abs/2015ApJ...806..259R/abstract
    """
    name = 'Reddy+15'
    #bump_ampl = 1.
    
    bump_ampl = Parameter(description="Amplitude of UV bump",
                      default=2., min=0., max=10.)
    
    bump_gamma = 0.04
    bump_x0 = 0.2175
    
    Rv = 2.505
    
    @staticmethod
    def _left(mu):
        """
        klam, mu < 0.6 micron
        """
        return -5.726 + 4.004/mu - 0.525/mu**2 + 0.029/mu**3 + 2.505
    
    
    @staticmethod
    def _right(mu):
        """
        klam, mu > 0.6 micron
        """
        return -2.672 - 0.010/mu + 1.532/mu**2 - 0.412/mu**3 + 2.505
    
    @property
    def koffset(self):
        """
        Force smooth transition at 0.6 micron
        """
        return self._left(0.6) - self._right(0.6)
        
    def evaluate(self, x, Av, bump_ampl):
       
        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.Angstrom
        else:
            xin = np.atleast_1d(x)
        
        mu = xin.to(u.micron).value
        left =  mu < 0.6
        klam = mu*0.
        # Reddy Eq. 8
        kleft = self._left(mu)
        kright = self._right(mu)
        
        klam[left] = self._left(mu[left])
        klam[~left] = self._right(mu[~left]) + self.koffset

        # Rv = Av/EBV
        # EBV=Av/Rv
        # klam = Alam/EBV
        # Alam = klam*EBV = klam*Av/Rv
        return np.maximum((klam + self.uv_bump(mu, bump_ampl))*Av/self.Rv, 0.)
    
    def uv_bump(self, mu, bump_ampl):
        """
        Drude profile for computing the UV bump.

        Parameters
        ----------
        x: np array (float)
           expects wavelengths in [micron]

        x0: float
           Central wavelength of the UV bump (in microns).

        gamma: float
           Width (FWHM) of the UV bump (in microns).

        ampl: float
           Amplitude of the UV bump.

        Returns
        -------
        np array (float)
           lorentzian-like Drude profile

        Raises
        ------
        ValueError
           Input x values outside of defined range

        """
        return bump_ampl * (mu**2 * self.bump_gamma**2 /
                       ((mu**2 - self.bump_x0**2)**2 + 
                         mu**2 * self.bump_gamma**2))

class KC13(BaseAttAvModel):
    """
    Kriek & Conroy (2013) attenuation model, extends Noll 2009 with UV bump 
    amplitude correlated with the slope, delta.
    
    Slightly different from KC13 since the N09 model uses Leitherer (2002) 
    below 1500 Angstroms.
    
    """
    name = 'Kriek+Conroy2013'
    
    delta = Parameter(description="delta: slope of the power law",
                      default=0., min=-3., max=3.)
    
    #extra_bump = 1.
    extra_params = {'extra_bump':1., 'beta':-3.2, 'extra_uv':-0.4}
    
    # Big range for use with FSPS
    x_range = [0.9e-4, 2.e8]
    
    def _init_N09(self):
        from dust_attenuation import averages, shapes, radiative_transfer

        # Allow extrapolation
        #shapes.x_range_N09 = [0.9e-4, 2.e8] 
        #averages.x_range_C00 = [0.9e-4, 2.e8]
        #averages.x_range_L02 = [0.9e-4, 0.18]
        shapes.C00.x_range = self.x_range
        shapes.N09.x_range = self.x_range 
        if self.x_range[0] < 0.18:
            shapes.L02.x_range = [self.x_range[0], 0.18] 
        else:
            shapes.L02.x_range = [0.097, 0.18] 
                    
        self.N09 = shapes.N09()
                
    def evaluate(self, x, Av, delta):
        import dust_attenuation
        
        if not hasattr(self, 'N09'):
            self._init_N09()
                    
        #Av = np.polyval(self.coeffs['Av'], tau_V)
        x0 = 0.2175
        gamma = 0.0350
        ampl = (0.85 - 1.9*delta)*self.extra_params['extra_bump']
        
        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.Angstrom
        else:
            xin = x
        
        wred = np.array([2.199e4])*u.Angstrom
        
        if self.N09.param_names[0] == 'x0':                
            Alam = self.N09.evaluate(xin, x0, gamma, ampl, delta, Av)
            Ared = self.N09.evaluate(wred, x0, gamma, ampl, delta, Av)[0]
        else:
            Alam = self.N09.evaluate(xin, Av, x0, gamma, ampl, delta)
            Ared = self.N09.evaluate(wred, Av, x0, gamma, ampl, delta)[0]
            
        
        # Extrapolate with beta slope
        red = xin > wred[0]
        if red.sum() > 0:
            Alam[red] = Ared*(xin[red]/wred[0])**self.extra_params['beta']
        
        blue = xin < 1500*u.Angstrom
        if blue.sum() > 0:
            plblue = np.ones(len(xin))
            wb = xin[blue].to(u.Angstrom).value/1500
            plblue[blue] = wb**self.extra_params['extra_uv']
            Alam *= plblue
            
        return Alam
        
class ParameterizedWG00(BaseAttAvModel):
    
    coeffs = {'Av': np.array([-0.001,  0.026,  0.643, -0.016]),
              'x0': np.array([ 3.067e-19, -7.401e-18,  6.421e-17, -2.370e-16,  
                               3.132e-16, 2.175e-01]),
              'gamma': np.array([ 2.101e-06, -4.135e-05,  2.719e-04, 
                                 -7.178e-04, 3.376e-04, 4.270e-02]),
              'ampl': np.array([-1.906e-03,  4.374e-02, -3.501e-01, 
                                 1.228e+00, -2.151e+00, 8.880e+00]),
              'slope': np.array([-4.084e-05,  9.984e-04, -8.893e-03,  
                                 3.670e-02, -7.325e-02, 5.891e-02])}
    
    # Turn off bump
    include_bump = 0.25
    
    wg00_coeffs = {'geometry': 'shell', 
                   'dust_type': 'mw',
                   'dust_distribution': 'homogeneous'}
    
    name = 'ParameterizedWG00'
        
    # def __init__(self, Av=1.0, **kwargs):
    #     """
    #     Version of the N09 curves fit to the WG00 curves up to tauV=10
    #     """        
    #     from dust_attenuation import averages, shapes, radiative_transfer
    #     
    #     # Allow extrapolation
    #     shapes.x_range_N09 = [0.01, 1000] 
    #     averages.x_range_C00 = [0.01, 1000]
    #     averages.x_range_L02 = [0.01, 0.18]
    #     
    #     self.N09 = shapes.N09()
    
    def _init_N09(self):
        from dust_attenuation import averages, shapes, radiative_transfer

        # Allow extrapolation
        shapes.x_range_N09 = [0.009, 2.e8] 
        averages.x_range_C00 = [0.009, 2.e8]
        averages.x_range_L02 = [0.009, 0.18]

        self.N09 = shapes.N09()
        
    def get_tau(self, Av):
        """
        Get the WG00 tau_V for a given Av
        """
        tau_grid = np.arange(0, 10, 0.01)
        av_grid = np.polyval(self.coeffs['Av'], tau_grid)
        return np.interp(Av, av_grid, tau_grid, left=0., right=tau_grid[-1])
        
    def evaluate(self, x, Av):
        
        import dust_attenuation
        
        if not hasattr(self, 'N09'):
            self._init_N09()
            
        tau_V = self.get_tau(Av)
        
        #Av = np.polyval(self.coeffs['Av'], tau_V)
        x0 = np.polyval(self.coeffs['x0'], tau_V)
        gamma = np.polyval(self.coeffs['gamma'], tau_V)
        if self.include_bump:
            ampl = np.polyval(self.coeffs['ampl'], tau_V)*self.include_bump
        else:
            ampl = 0.
            
        slope = np.polyval(self.coeffs['slope'], tau_V)
        
        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.Angstrom
        else:
            xin = x
        
        if self.N09.param_names[0] == 'x0':                
            return self.N09.evaluate(xin, x0, gamma, ampl, slope, Av)
        else:
            return self.N09.evaluate(xin, Av, x0, gamma, ampl, slope)


def fsps_line_info(wlimits=None):
    """
    Read FSPS line list
    """
    try:
        info_file = os.path.join(os.getenv('SPS_HOME'), 'data/emlines_info.dat')
        with open(info_file, 'r') as f:
            lines = f.readlines()
    except:
        return [], []
        
    waves = np.array([float(l.split(',')[0]) for l in lines])
    names = np.array([l.strip().split(',')[1].replace(' ','') for l in lines])
    if wlimits is not None:
        clip = (waves > wlimits[0]) & (waves < wlimits[1])
        waves = waves[clip]
        names = names[clip]
        
    return waves, names


DEFAULT_LINES = fsps_line_info(wlimits=[1200, 1.9e4])[0]

BOUNDS = {}
BOUNDS['tage'] = [0.03, 12, 0.05]
BOUNDS['tau'] = [0.03, 2, 0.05]
BOUNDS['zred'] = [0.0, 13, 1.e-4]
BOUNDS['Av'] = [0.0, 15, 0.05]
BOUNDS['gas_logu'] = [-4, 0, 0.05]
BOUNDS['gas_logz'] = [-2, 0.3, 0.05]
BOUNDS['logzsol'] = [-2, 0.3, 0.05]
BOUNDS['sigma_smooth'] = [100, 500, 0.05]


def wuyts_line_Av(Acont):
    """
    Wuyts prescription for extra extinction towards nebular emission
    """
    return Acont + 0.9*Acont - 0.15*Acont**2


class ExtendedFsps(StellarPopulation):
    """
    Extended functionality for the `fsps.StellarPopulation` object
    """
    
    lognorm_center = 0.
    lognorm_logwidth = 0.05
    is_lognorm_sfh = False
    lognorm_fburst = -30
    
    cosmology = WMAP9
    scale_lyman_series = 0.1
    scale_lines = OrderedDict()
    
    line_av_func = None
    
    #_meta_bands = ['v']
    
    @property
    def izmet(self):
        """
        Get zmet index for nearest ``self.zlegend`` value to ``loggzsol``.
        """
        NZ = len(self.zlegend)
        logzsol = self.params['logzsol']
        zi = np.interp(logzsol, np.log10(self.zlegend/0.019), np.arange(NZ))
        return np.clip(np.cast[int](np.round(zi)), 0, NZ-1)
    
    @property
    def fsps_ages(self):
        """
        (linear) ages of the FSPS SSP age grid, Gyr
        """
        if hasattr(self, '_fsps_ages'):
            return self._fsps_ages
        
        _ = self.get_spectrum()  
        fsps_ages = 10**(self.log_age-9)
        self._fsps_ages = fsps_ages
        return fsps_ages
        
    def set_lognormal_sfh(self, min_sigma=3, verbose=False, **kwargs):
        """
        Set lognormal tabular SFH
        """
        try:
            from grizli.utils_c.interp import interp_conserve_c as interp_func
        except:
            interp_func = utils.interp_conserve
        
        if 'lognorm_center' in kwargs:
            self.lognorm_center = kwargs['lognorm_center']
        
        if 'lognorm_logwidth' in kwargs:
            self.lognorm_logwidth = kwargs['lognorm_logwidth']
                
        if self.is_lognorm_sfh:
            self.params['sfh'] = 3
        
        if verbose:
            msg = 'lognormal SFH ({0}, {1}) [sfh3={2}]'
            print(msg.format(self.lognorm_center, self.lognorm_logwidth, 
                             self.is_lognorm_sfh))
                
        xages = np.logspace(np.log10(self.fsps_ages[0]), 
                            np.log10(self.fsps_ages[-1]), 2048)

        mu = self.lognorm_center#*np.log(10)
        # sfh = 1./t*exp(-(log(t)-mu)**2/2/sig**2)
        logn_sfh = 10**(-(np.log10(xages)-mu)**2/2/self.lognorm_logwidth**2)
        logn_sfh *= 1./xages 
        
        # Normalize
        logn_sfh *= 1.e-9/(self.lognorm_logwidth*np.sqrt(2*np.pi*np.log(10)))
        
        self.set_tabular_sfh(xages, logn_sfh) 
        self._lognorm_sfh = (xages, logn_sfh)
    
    def lognormal_integral(self, tage=0.1, **kwargs):
        """
        Integral of lognormal SFH up to t=tage
        """
        from scipy.special import erfc
        mu = self.lognorm_center*np.log(10)
        sig = self.lognorm_logwidth*np.sqrt(np.log(10))
        cdf = 0.5*erfc(-(np.log(tage)-mu)/sig/np.sqrt(2)) 
        return cdf
        
    def _set_extend_attrs(self, line_sigma=50, lya_sigma=200, **kwargs):
        """
        Set attributes on `fsps.StellarPopulation` object used by `narrow_lines`.

        sigma : line width (FWHM/2.35), km/s.
        lya_sigma : width for Lyman-alpha

        Sets `emline_dlam`, `emline_sigma` attributes.

        """
        
        # Line widths, native FSPS and new 
        wave, line = self.get_spectrum(tage=1., peraa=True)
        dlam = np.diff(wave)
        self.emline_dlam = [np.interp(w, wave[1:], dlam) 
                            for w in self.emline_wavelengths] # Angstrom
        self.emline_sigma = [line_sigma for w in self.emline_wavelengths] #kms
        
        # Separate Ly-alpha
        lya_ix = np.argmin(np.abs(self.emline_wavelengths - 1216.8))
        self.emline_sigma[lya_ix] = lya_sigma
        
        # Line EWs computed in `narrow_emission_lines`
        self.emline_eqw = [-1e10 for w in self.emline_wavelengths]
        
        # Emission line names
        waves, names = fsps_line_info()
        if np.allclose(self.emline_wavelengths, waves, 0.5):
            self.emline_names = names
        else:
            self.emline_names = ['?'] * len(self.emline_wavelengths)
            for w, n in zip(waves, names):
                dl = np.abs(self.emline_wavelengths - w)
                if dl.min() < 0.5:
                    self.emline_names[np.argmin(dl)] = n
        
        for l in self.emline_names:
            self.scale_lines[l] = 1.
            
        # Precomputed arrays for WG00 reddening defined between 0.1..3 um
        self.wg00lim = (self.wavelengths > 1000) & (self.wavelengths < 3.e4)
        self.wg00red = (self.wavelengths > 1000)*1.
        
        self.exec_params = None
        self.narrow = None
        
    def narrow_emission_lines(self, tage=0.1, emwave=DEFAULT_LINES, line_sigma=100, oversample=5, clip_sigma=10, verbose=False, get_eqw=True, scale_lyman_series=None, scale_lines={}, force_recompute=False, use_sigma_smooth=True, lorentz=False, **kwargs):
        """
        Replace broad FSPS lines with specified line widths
    
        tage : age in Gyr of FSPS model
        FSPS sigma: line width in A in FSPS models
        emwave : (approx) wavelength of line to replace
        line_sigma : line width in km/s of new line
        oversample : factor by which to sample the Gaussian profiles
        clip_sigma : sigmas from line center to use for the line
        scale_lyman_series : scaling to apply to Lyman-series emission lines
        scale_lines : scaling to apply to other emission lines, by name
        
        Returns: `dict` with keys
            wave_full, flux_full, line_full = wave and flux with fine lines
            wave, flux_line, flux_clean = original model + removed lines
            ymin, ymax = range of new line useful for plotting
        
        """
        if not hasattr(self, 'emline_dlam'):
            self._set_extend_attrs(line_sigma=line_sigma, **kwargs)
        
        self.params['add_neb_emission'] = True
        
        if scale_lyman_series is None:
            scale_lyman_series = self.scale_lyman_series
        else:
            self.scale_lyman_series = scale_lyman_series
        
        if scale_lines is None:
            scale_lines = self.scale_lines
        else:
            for k in scale_lines:
                if k in self.scale_lines:
                    self.scale_lines[k] = scale_lines[k]
                else:
                    print(f'Line "{k}" not found in `self.scale_lines`')
        
        # Avoid recomputing if all parameters are the same (i.e., change Av)
        call_params = np.hstack([self.param_floats(params=None), emwave, 
                                 list(self.scale_lines.values()), 
                        [tage, oversample, clip_sigma, scale_lyman_series]])
        try:
            is_close = np.allclose(call_params, self.exec_params)
        except:
            is_close = False
            
        if is_close & (not force_recompute):
             if verbose:
                 print('use stored')
             return self.narrow
        
        self.exec_params = call_params
        wave, line = self.get_spectrum(tage=tage, peraa=True)
        
        line_ix = [np.argmin(np.abs(self.emline_wavelengths - w)) 
                   for w in emwave]
                   
        line_lum = [self.emline_luminosity[i] for i in line_ix]
        line_wave = [self.emline_wavelengths[i] for i in line_ix]
        fsps_sigma = [np.sqrt((2*self.emline_dlam[i])**2 + 
             (self.params['sigma_smooth']/3.e5*self.emline_wavelengths[i])**2)
                              for i in line_ix]
        
        if line_sigma < 0:
            lines_sigma = [-line_sigma for ix in line_ix]
        elif (self.params['sigma_smooth'] > 0) & (use_sigma_smooth):
            lines_sigma = [self.params['sigma_smooth'] for ix in line_ix]
        else:
            lines_sigma = [self.emline_sigma[ix] for ix in line_ix]
            
        line_dlam = [sig/3.e5*lwave 
                     for sig, lwave in zip(lines_sigma, line_wave)]
    
        clean = line*1
        wlimits = [np.min(emwave), np.max(emwave)]
        wlimits = [2./3*wlimits[0], 4.3*wlimits[1]]
    
        wfine = utils.log_zgrid(wlimits, np.min(lines_sigma)/oversample/3.e5)
        qfine = wfine < 0
    
        if verbose:
            msg = 'Matched line: {0} [{1}], lum={2}'
            for i, ix in enumerate(line_ix):
                print(msg.format(line_wave[i], ix, line_lum[i]))
    
        ######### 
        # Remove lines from FSPS
        # line width seems to be 2*dlam at the line wavelength
        for i, ix in enumerate(line_ix):
            if self.params['nebemlineinspec']:
                gauss = 1/np.sqrt(2*np.pi*fsps_sigma[i]**2)
                gauss *= np.exp(-(wave - line_wave[i])**2/2/fsps_sigma[i]**2)
                clean -= gauss*line_lum[i]
            
            # indices of fine array where new lines defined
            qfine |= np.abs(wfine - line_wave[i]) < clip_sigma*line_dlam[i]
                
        # Linear interpolate cleaned spectrum on fine grid
        iclean = np.interp(wfine[qfine], wave, clean)
        
        # Append original and fine sampled arrays
        wfull = np.append(wave, wfine[qfine])
        cfull = np.append(clean, iclean)
        so = np.argsort(wfull)
        wfull, uniq = np.unique(wfull, return_index=True)
        cfull = cfull[uniq]
    
        gfull = cfull*0.
        for i in range(len(line_wave)):
                    
            if lorentz:
                # astropy.modeling.functional_models.Lorentz1D.html
                # gamma is FWHM/2., integral is gamma*pi
                gam = 2.35482*line_dlam[i]/2.
                gline = gam**2/(gam**2 + (wfull-line_wave[i])**2) 
                norm = line_lum[i]/(gam*np.pi)
            else:
                # Gaussian
                gline = np.exp(-(wfull - line_wave[i])**2/2/line_dlam[i]**2)
                norm = line_lum[i]/np.sqrt(2*np.pi*line_dlam[i]**2)
                
            if self.emline_names[line_ix[i]].startswith('Ly'):
                norm *= scale_lyman_series
            
            if self.emline_names[line_ix[i]] in self.scale_lines:
                norm *= self.scale_lines[self.emline_names[line_ix[i]]]
                    
            gfull += gline*norm
            
            if get_eqw:
                clip = np.abs(wfull - line_wave[i]) < clip_sigma*line_dlam[i]
                eqw = np.trapz(gline[clip]*norm/cfull[clip], wfull[clip])
                self.emline_eqw[line_ix[i]] = eqw
                
        cfull += gfull
    
        # For plot range
        ymin = iclean.min()
        line_peak = [1/np.sqrt(2*np.pi*dlam**2)*lum for dlam,lum in zip(line_dlam, line_lum)]
        ymax = iclean.max()+np.max(line_peak)
        
        data = OrderedDict()
        data['wave_full'] =  wfull
        data['flux_full'] =  cfull
        data['line_full'] =  gfull
        data['wave' ] =  wave
        data['flux_line' ] =  line
        data['flux_clean'] =  clean
        data['ymin' ] =  ymin
        data['ymax' ] =  ymax
        self.narrow = data
        
        return data


    def set_fir_template(self, arrays=None, file='templates/magdis/magdis_09.txt', verbose=True, unset=False, scale_pah3=0.5):
        """
        Set the far-IR template for reprocessed dust emission
        """
        
        if unset:
            if verbose:
                print('Unset FIR template attributes')
            
            for attr in ['fir_template', 'fir_name', 'fir_arrays']:
                if hasattr(self, attr):
                    delattr(self, attr)
            
            return True
            
        if os.path.exists(file):
            if verbose:
                print(f'Set FIR dust template from {file}')
            _ = np.loadtxt(file, unpack=True)
            wave, flux = _[0], _[1]
            
            self.fir_name = file

        elif arrays is not None:
            if verbose:
                print(f'Set FIR dust template from input arrays')
            wave, flux = arrays
            self.fir_name = 'user-supplied'
        else:
            if verbose:
                print(f'Set FIR dust template from FSPS (DL07)')

            # Set with fsps
            self.params['dust1'] = 0
            self.params['dust2'] = 1.
            
            self.params['add_dust_emission'] = True
            wave, flux = self.get_spectrum(tage=1., peraa=True)
            self.params['add_dust_emission'] = False
            wave, flux_nodust = self.get_spectrum(tage=1., peraa=True)
            flux -= flux_nodust

            self.fir_name = 'fsps-dl07'
        
        if scale_pah3 is not None:
            if verbose:
                print(f'Scale 3.3um PAH: {scale_pah3:.2f}')
                
            ran = np.abs(wave-3.3e4) < 0.5e4
            line = np.abs(wave-3.3e4) < 0.3e4
            px = np.polyfit(wave[ran & ~line], flux[ran & ~line], 3)
            
            scaled_line = (flux[ran] - np.polyval(px, wave[ran]))*scale_pah3
            
            flux[ran] = np.polyval(px, wave[ran]) + scaled_line
                
        fir_flux = np.interp(self.wavelengths, wave, flux, left=0, right=0)
        self.fir_template = fir_flux/np.trapz(fir_flux, self.wavelengths)
        self.fir_arrays = arrays
        return True


    def set_dust(self, Av=0., dust_obj_type='WG00x', wg00_kwargs=WG00_DEFAULTS):
        """
        Set `dust_obj` attribute
        
        dust_obj_type: 
        
            'WG00'  = `dust_attenuation.radiative_transfer.WG00`
            'C00'   = `dust_attenuation.averages.C00`
            'WG00x' = `ParameterizedWG00`
            'KC13'  = Kriek & Conroy (2013) with dust_index parameter
            'R15'  = Reddy et al. (2015) with dust bump parameter
            
        ir_template: (wave, flux)
            Template to use for re-emitted IR light
            
        """
        from dust_attenuation import averages, radiative_transfer
        
        self.params['dust1'] = 0.
        self.params['dust2'] = 0.
                
        needs_init = False
        if not hasattr(self, 'dust_obj'):
            needs_init = True
        
        if hasattr(self, 'dust_obj_type'):
            if self.dust_obj_type != dust_obj_type:
                needs_init = True
        
        if needs_init:
            self.dust_obj_type = dust_obj_type
            if dust_obj_type == 'WG00':
                self.dust_obj = radiative_transfer.WG00(tau_V=1.0,
                                                        **WG00_DEFAULTS) 
            elif dust_obj_type == 'WG00x':
                self.dust_obj = ParameterizedWG00(Av=Av)             
            elif dust_obj_type == 'C00':
                self.dust_obj = averages.C00(Av=Av)
            elif dust_obj_type == 'R15':
                self.dust_obj = Reddy15(Av=Av, bump_ampl=2.)
            elif hasattr(dust_obj_type, 'extinguish'):
                self.dust_obj = dust_obj_type
            else:
                self.dust_obj = KC13(Av=Av)
                
            print('Init dust_obj: {0} {1}'.format(dust_obj_type, self.dust_obj.param_names))
            
        self.Av = Av
        
        if dust_obj_type == 'WG00':
            Avs = np.array([0.151, 0.298, 0.44 , 0.574, 0.825, 1.05 , 1.252, 1.428, 1.584, 1.726, 1.853, 1.961, 2.065, 2.154, 2.318, 2.454, 2.573, 2.686, 3.11 , 3.447, 3.758, 4.049, 4.317, 4.59 , 4.868, 5.148])
            taus = np.array([ 0.25,  0.5 ,  0.75,  1.  ,  1.5 ,  2.  ,  2.5 ,  3.  ,  3.5 , 4.  ,  4.5 ,  5.  ,  5.5 ,  6.  ,  7.  ,  8.  ,  9.  , 10.  , 15.  , 20.  , 25.  , 30.  , 35.  , 40.  , 45.  , 50.  ])
            tau_V = np.interp(Av, Avs, taus, left=0.25, right=50)
            self.dust_obj.tau_V = tau_V
            self.Av = self.dust_obj(5500*u.Angstrom)
        elif dust_obj_type == 'KC13':
            self.dust_obj.Av = Av
            self.dust_obj.delta = self.params['dust_index']
        else:
            self.dust_obj.Av = Av
    
    def redden(self, wave):
        if hasattr(self.dust_obj, 'extinguish'):
            return self.dust_obj.extinguish(wave, Av=self.Av)
        else:
            return 10**(-0.4*self.dust_obj(wave))
            
    def get_full_spectrum(self, tage=1.0, Av=0., get_template=True, set_all_templates=False, z=None, tie_metallicity=True, **kwargs):
        """
        Get full spectrum with reprocessed emission lines and dust emission
        
        dust_fraction: Fraction of the SED that sees the specified Av
        
        """
        
        # Set the dust model
        if Av is None:
            Av = self.Av
            
        if 'dust_obj_type' in kwargs:
            self.set_dust(Av=Av, dust_obj_type=kwargs['dust_obj_type'])
        elif hasattr(self, 'dust_obj'):
            self.set_dust(Av=Av, dust_obj_type=self.dust_obj_type)
        else:
            self.set_dust(Av=Av, dust_obj_type='WG00x')
        
        # Lognormal SFH?
        if ('lognorm_center' in kwargs) | ('lognorm_logwidth' in kwargs):
            self.set_lognormal_sfh(**kwargs)
        
        if 'lognorm_fburst' in kwargs:
            self.lognorm_fburst = kwargs['lognorm_fburst']
        
        # Burst fraction for lognormal SFH
        if self.is_lognorm_sfh:
            if not hasattr(self, '_lognorm_sfh'):
                self.set_lognormal_sfh()
        
            t1 = self.lognormal_integral(tage)
            dt = (tage-self._lognorm_sfh[0])
            t100 = (dt <= 0.1) & (dt >= 0)
            
            sfhy = self._lognorm_sfh[1]*1.
            sfhy += t1*10**self.lognorm_fburst/100e6*t100
            self.set_tabular_sfh(self._lognorm_sfh[0], sfhy) 
                             
        # Set FSPS parameters
        for k in kwargs:
            if k in self.params.all_params:
                self.params[k] = kwargs[k]
        
        if 'zmet' not in kwargs:
            self.params['zmet'] = self.izmet
        
        if ('gas_logz' not in kwargs) & tie_metallicity:
            self.params['gas_logz'] = self.params['logzsol']
            
        # Run the emission line function
        if tage is None:
            tage = self.params['tage']
            
        _ = self.narrow_emission_lines(tage=tage, **kwargs)

        wave = _['wave_full']
        flux = _['flux_full']
        lines = _['line_full']
        contin = flux - lines
        
        #self.sfr100 = self.sfr_avg(dt=np.minimum(tage, 0.1))
        
        # Apply dust
        if self.dust_obj_type == 'WG00':
            
            # To template
            red = (wave > 1000)*1.
            wlim = (wave > 1000) & (wave < 3.e4)
            red[wlim] = self.redden(wave[wlim]*u.Angstrom)
            
            # To lines
            red_lines = (self.emline_wavelengths > 1000)*1.
            wlim = (self.emline_wavelengths > 1000) 
            wlim &= (self.emline_wavelengths < 3.e4)
            line_wave = self.emline_wavelengths[wlim]*u.Angstrom
            red_lines[wlim] = self.redden(line_wave)
            
        else:
            red = self.redden(wave*u.Angstrom)
            
            if self.line_av_func is None:
                self.Av_line = self.Av*1.
                red_lines_full = red
                line_wave = self.emline_wavelengths*u.Angstrom
                red_lines = self.redden(line_wave)
            else:
                # Differential reddening towards nebular lines
                self.Av_line = self.line_av_func(Av)
                self.set_dust(Av=self.Av_line,
                              dust_obj_type=self.dust_obj_type)
                
                red_lines_full = self.redden(wave*u.Angstrom)
                line_wave = self.emline_wavelengths*u.Angstrom
                red_lines = self.redden(line_wave)
                
                # Reset for continuum
                self.set_dust(Av=Av, dust_obj_type=self.dust_obj_type)
                
        # Apply dust to line luminosities
        lred = [llum*lr for llum, lr in 
                        zip(self.emline_luminosity, red_lines)]
        self.emline_reddened = np.array(lred)
        
        # Total energy
        e0 = np.trapz(flux, wave)
        # Energy of reddened template
        
        reddened = contin*red+lines*red_lines_full
        e1 = np.trapz(reddened, wave)
        self.energy_absorbed = (e0 - e1)
                
        # Add dust emission
        if hasattr(self, 'fir_template') & self.params['add_dust_emission']:
            dust_em = np.interp(wave, self.wavelengths, self.fir_template)
            dust_em *= self.energy_absorbed
        else:
            dust_em = 0.
        
        meta0 = self.meta
        self.templ = self.as_template(wave, reddened+dust_em, meta=meta0)
        
        # Set template attributes
        if set_all_templates:
            
            # Original wavelength grid
            # owave = self.wavelengths
            # owave = owave[self.wg00lim]*u.Angstrom
            # self.wg00red[self.wg00lim] = self.redden(owave)
            # ofir = self.fir_template*self.energy_absorbed
            # fl_orig = _['flux_line']*self.wg00red + ofir
            # self.templ_orig = self.as_template(owave, fl_orig, meta=meta0)
            
            # No lines
            meta = meta0.copy()
            meta['add_neb_emission'] = False
            fl_cont = contin*red + dust_em
            #ocont = _['flux_clean']*self.wg00red + ofir
            self.templ_cont = self.as_template(wave, fl_cont, meta=meta)
            #self.templ_cont_orig = self.as_template(owave, ocont, meta=meta)
            
            # No dust
            meta = meta0.copy()
            meta['add_neb_emission'] = True
            meta['Av'] = 0
            self.templ_unred = self.as_template(wave, flux, meta=meta)
            #self.templ_unred_orig = self.as_template(owave, _['flux_clean'],
            #                                         meta=meta)
            
        if get_template:
            return self.templ
        else:
            return self.templ.wave, self.templ.flux
            
    def as_template(self, wave, flux, label=DEFAULT_LABEL, meta=None):
        """
        Return a `eazy.templates.Template` object with metadata
        """
        if meta is None:
            meta = self.meta
            
        templ = templates.Template(arrays=(wave, flux), meta=meta,
                                   name=label.format(**meta))
        return templ
    
    def lognorm_avg_sfr(self, tage=None, dt=0.1):
        """
        Analytic average SFR for lognorm SFH
        """
        if tage is None:
            tage = self.params['tage']
            
        t1 = self.lognormal_integral(tage)
        t0 = self.lognormal_integral(np.maximum(tage-dt, 0))
        sfr_avg = (t1*(1+10**self.lognorm_fburst)-t0)/(dt*1.e9)
        return sfr_avg
        
    @property 
    def sfr100(self):
        """
        SFR averaged over maximum(tage, 100 Myr) from `sfr_avg`
        """
        if self.params['sfh'] == 0:
            sfr_avg = 0.
        elif self.params['sfh'] == 3:
            # Try to integrate SFH arrays if attribute set
            if self.is_lognorm_sfh:
                sfr_avg = self.lognorm_avg_sfr(tage=None, dt=0.1)
                
            elif hasattr(self, '_sfh_tab'):
                try:
                    fwd = self.params['tabsfh_forward']
                except:
                    fwd = 1

                if fwd == 1:
                    age_lb = self.params['tage'] - self._sfh_tab[0]
                    step = -1
                else:
                    age_lb = self._sfh_tab[0]
                    step = 1
                    
                age100 = (age_lb <= 0.1) & (age_lb >= 0)
                if age100.sum() < 2:
                    sfr_avg = 0.
                else:
                    sfr_avg = np.trapz(self._sfh_tab[1][age100][::step],
                                       age_lb[age100][::step])/0.1
                
            else:
                sfr_avg = 0.
        else:
            sfr_avg = self.sfr_avg(dt=np.minimum(self.params['tage'], 0.1))
            
        return sfr_avg
    
    @property 
    def sfr10(self):
        """
        SFR averaged over last MAXIMUM(tage, 10 Myr) from `sfr_avg`
        """
        if self.params['sfh'] == 0:
            sfr_avg = 0.
        elif self.params['sfh'] == 3:
            # Try to integrate SFH arrays if attribute set
            if self.is_lognorm_sfh:
                sfr_avg = self.lognorm_avg_sfr(tage=None, dt=0.01)
                
            elif hasattr(self, '_sfh_tab'):
                try:
                    fwd = self.params['tabsfh_forward']
                except:
                    fwd = 1
                    
                if fwd == 1:
                    age_lb = self.params['tage'] - self._sfh_tab[0]
                    step = -1
                else:
                    age_lb = self._sfh_tab[0]
                    step = 1

                age10 = (age_lb < 0.01) & (age_lb >= 0)
                if age10.sum() < 2:
                    sfr_avg = 0.
                else:
                    sfr_avg = np.trapz(self._sfh_tab[1][age10][::step],
                                       age_lb[age10][::step])/0.1
                
            else:
                sfr_avg = 0.
        else:
            sfr_avg = self.sfr_avg(dt=np.minimum(self.params['tage'], 0.01))
            
        return sfr_avg
        
    @property 
    def meta(self):
        """
        Full metadata, including line properties
        """
        import fsps
        meta = self.param_dict
        
        if self._zcontinuous:
            meta['metallicity'] = 10**self.params['logzsol']*0.019
        else:
            meta['metallicity'] = self.zlegend[self.params['zmet']]
            
        for k in ['log_age','stellar_mass', 'formed_mass', 'log_lbol', 
                  'sfr', 'sfr100', 'dust_obj_type','Av','energy_absorbed', 
                  'fir_name', '_zcontinuous', 'scale_lyman_series',
                  'lognorm_center', 'lognorm_logwidth', 'is_lognorm_sfh', 
                  'lognorm_fburst']:
            if hasattr(self, k):
                meta[k] = self.__getattribute__(k)
        
        if hasattr(self, 'emline_names'):
            has_red = hasattr(self, 'emline_reddened')
            
            if self.emline_luminosity.ndim == 1:
                for i in range(len(self.emline_wavelengths)):
                    n = self.emline_names[i]
                    if n in self.scale_lines:
                        kscl = self.scale_lines[n]
                    else:
                        kscl = 1.0
                
                    meta[f'scale {n}'] = kscl
                    meta[f'line {n}'] = self.emline_luminosity[i]*kscl
                    if has_red:
                        meta[f'rline {n}'] = self.emline_reddened[i]*kscl
                    
                    meta[f'eqw {n}'] = self.emline_eqw[i]
                    meta[f'sigma {n}'] = self.emline_sigma[i]

        # Band information
        if hasattr(self, '_meta_bands'):
            light_ages = self.light_age_band(self._meta_bands, flat=False)
            band_flux = self.get_mags(tage=self.params['tage'], zmet=None,
                                      bands=self._meta_bands, units='flam')
            
            band_waves = [fsps.get_filter(b).lambda_eff*u.Angstrom 
                          for b in self._meta_bands]
            band_lum = [f*w for f, w in zip(band_flux, band_waves)]
            
            for i, b in enumerate(self._meta_bands):
                meta[f'lwage_{b}'] = light_ages[i]
                meta[f'lum_{b}'] = band_lum[i].value
        
        try:
            meta['libraries'] = ';'.join([s.decode() for s in self.libraries])  
        except:
            try:
                meta['libraries'] = ';'.join([s for s in self.libraries])  
            except:
                meta['libraries'] = '[error]'
                
        return meta
        
    @property 
    def param_dict(self):
        """
        `dict` version of `self.params`
        """
        d = OrderedDict()
        for p in self.params.all_params:
            d[p] = self.params[p]
        
        return d
    
    def light_age_band(self, bands=['v'], flat=True):
        """
        Get light-weighted age of current model
        """
        self.params['compute_light_ages'] = True
        band_ages = self.get_mags(tage=self.params['tage'], zmet=None, 
                                  bands=bands)
        self.params['compute_light_ages'] = False
        if flat & (band_ages.shape == (1,)):
            return band_ages[0]
        else:
            return band_ages
        
    def pset(self, params):
        """
        Return a subset dictionary of `self.meta`
        """
        d = OrderedDict()
        for p in params:
            if p in self.meta:
                d[p] = self.meta[p]
            else:
                d[p] = np.nan
                
        return d
        
    def param_floats(self, params=None):
        """
        Return a list of parameter values.  If `params` is None, then use 
        full list in `self.params.all_params`. 
        """
        
        if params is None:
            params = self.params.all_params
            
        d = []
        for p in params:
            d.append(self.params[p]*1)
        
        return np.array(d)
    
    def parameter_bounds(self, params, limit_age=False):
        """
        Parameter bounds for `scipy.optimize.least_squares`
        
        """
        blo = []
        bhi = []
        steps = []
        for p in params:
            if p in BOUNDS:
                blo.append(BOUNDS[p][0])
                bhi.append(BOUNDS[p][1])
                steps.append(BOUNDS[p][2])
            else:
                blo.append(-np.inf)
                bhi.append(np.inf)
                steps.append(0.05)
                
        return (blo, bhi), steps
            
    def fit_spec(self, wave_obs, flux_obs, err_obs, mask=None, plist=['tage', 'Av', 'gas_logu', 'sigma_smooth'], func_kwargs={'lorentz':False}, verbose=True, bspl_kwargs=None, cheb_kwargs=None, lsq_kwargs={'method':'trf', 'max_nfev':200, 'loss':'huber', 'x_scale':1.0, 'verbose':True}, show=False):
        """
        Fit models to observed spectrum
        """
        from scipy.optimize import least_squares
        import matplotlib.pyplot as plt
        
        import grizli.utils
        
        sys_err = 0.015
        
        if wave_obs is None:
            # mpdaf muse spectrum
            _ = None # muse spectrum object
            spec = _[0].spectra['MUSE_TOT_SKYSUB']
            wave_obs = spec.wave.coord()
            flux_obs = spec.data.filled(fill_value=np.nan)
            err_obs = np.sqrt(spec.var.filled(fill_value=np.nan))
            err_obs = np.sqrt(err_obs**2+(sys_err*flux_obs)**2) 
            
            mask = np.isfinite(flux_obs+err_obs) & (err_obs > 0) 
            omask = mask
            
            #mask = omask & (wave_obs/(1+0.0342) > 6520) & (wave_obs/(1+0.0342) < 6780)
            mask = omask & (wave_obs/(1+0.0342) > 4800) & (wave_obs/(1+0.0342) < 5050)
            
        theta0 = np.array([self.meta[p] for p in plist])
        
        if bspl_kwargs is not None:
            bspl = grizli.utils.bspline_templates(wave_obs, get_matrix=True, 
                                                  **bspl_kwargs)
        elif cheb_kwargs is not None:
            bspl = grizli.utils.cheb_templates(wave_obs, get_matrix=True, 
                                                  **cheb_kwargs)
        else:
            bspl = None
            
        kwargs = func_kwargs.copy()
        for i, p in enumerate(plist):
            kwargs[p] = theta0[i]
        
        # Model test
        margs = (self, plist, wave_obs, flux_obs, err_obs, mask, bspl, kwargs, 'model')        
        flux_model, Anorm, chi2_init = self.objfun_fitspec(theta0, *margs)
        
        if show:                
            mask &= np.isfinite(flux_model+flux_obs+err_obs) & (err_obs > 0) 

            plt.close('all')
            
            fig = plt.figure(figsize=(12, 6))
            plt.errorbar(wave_obs[mask], flux_obs[mask], err_obs[mask], color='k', alpha=0.5, linestyle='None', marker='.')
            plt.plot(wave_obs, flux_model, color='pink', linewidth=2, alpha=0.8)
        else:
            fig = None
            
        bounds, steps = self.parameter_bounds(plist)
        #lsq_kwargs['diff_step'] = np.array(steps)/2.
        #lsq_kwargs['diff_step'] = 0.05
        lsq_kwargs['diff_step'] = steps
        lmargs = (self, plist, wave_obs, flux_obs, err_obs, mask, bspl, kwargs, 'least_squares verbose')        
        _res = least_squares(self.objfun_fitspec, theta0, bounds=bounds, args=lmargs, **lsq_kwargs)
        
        fit_model, Anorm, chi2_fit = self.objfun_fitspec(_res.x, *margs)
        
        result = {'fit_model':fit_model, 'Anorm':Anorm, 'chi2_fit':chi2_fit, 
                  'least_squares':_res, 'bounds':bounds, 'plist':plist, 
                  'lsq_kwargs':lsq_kwargs, 'bspl':bspl}
        
        return result


    @staticmethod
    def objfun_fitspec(theta, self, plist, wave_obs, flux_obs, err_obs, mask, bspl, kwargs, ret_type):
        """
        Objective function for fitting spectra
        """
        import scipy.stats
        
        try:
            from grizli.utils_c.interp import interp_conserve_c as interp_func
        except:
            interp_func = utils.interp_conserve
        
        err_scale = 1.
            
        for i, p in enumerate(plist):
            if p == 'err_scale':
                err_scale = theta[i]
                continue
                
            kwargs[p] = theta[i]
                    
        templ = self.get_full_spectrum(**kwargs)
        flux_model = templ.resample(wave_obs, z=self.params['zred'],
                                   in_place=False,
                                   return_array=True, interp_func=interp_func)
        
        flux_model = flux_model.flatten()
        
        if mask is None:
            mask = np.isfinite(flux_model+flux_obs+err_obs) & (err_obs > 0)
        
        if bspl is not None:
            _A = (bspl.T*flux_model)
            _yx = (flux_obs / err_obs)[mask]
            _c = np.linalg.lstsq((_A/err_obs).T[mask,:], _yx, rcond=-1)
            Anorm = np.mean(bspl.dot(_c[0]))
            flux_model = _A.T.dot(_c[0])
            
        else:
            lsq_num = (flux_obs*flux_model/err_obs**2)[mask].sum()
            lsq_den = (flux_model**2/err_obs**2)[mask].sum()
            Anorm = lsq_num/lsq_den
            flux_model *= Anorm
        
        chi = ((flux_model - flux_obs)/err_obs)[mask]
        chi2 = (chi**2).sum()
                                       
        if 'verbose' in ret_type:
            print('{0} {1:.4f}'.format(theta, chi2))
            
        if 'model' in ret_type:
            return flux_model, Anorm, chi2
        elif 'least_squares' in ret_type:
            return chi
        elif 'logpdf' in ret_type:
            #return -chi2/2.
            lnp = scipy.stats.norm.logpdf((flux_model-flux_obs)[mask], 
                                           loc=0, 
                                           scale=(err_obs*flux_model)[mask])
            
            return lnp
        else:
            return chi2
    
    def line_to_obsframe(self, zred=None, cosmology=None, verbose=False, unit=LINE_CGS, target_stellar_mass=None, target_sfr=None, target_lir=None):
        """
        Scale factor to convert internal line luminosities (L_sun) to observed frame
        
        If ``target_stellar_mass``, ``target_sfr``, or ``target_lir`` 
        specified, then scale the output to the desired value using the 
        intrinsic properties.  Units are linear ``Msun``, ``Msun/yr``, 
        and ``Lsun``, respectively.
        
        """
        from astropy.constants import L_sun
        
        if zred == None:
            zred = self.params['zred']
            if verbose:
                msg = 'continuum_to_obsframe: Use params[zred] = {0:.3f}'
                print(msg.format(zred))
                    
        if cosmology is None:
            cosmology = self.cosmology
        else:
            self.cosmology = cosmology
        
        if zred <= 0:
            dL = 1*u.cm
        else:
            dL = cosmology.luminosity_distance(zred).to(u.cm)

        to_cgs = (1*L_sun/(4*np.pi*dL**2)).to(unit)
        
        if target_stellar_mass is not None:
            to_cgs *= target_stellar_mass / self.stellar_mass
        elif target_sfr is not None:
            to_cgs *= target_sfr / self.sfr100
        elif target_lir is not None:
            to_cgs *= target_lir / self.energy_absorbed
        
        return to_cgs.value
        
    def continuum_to_obsframe(self, zred=None, cosmology=None, unit=u.microJansky, verbose=False, target_stellar_mass=None, target_sfr=None, target_lir=None):
        """
        Compute a normalization factor to scale input FSPS model flux density 
        units of (L_sun / Hz) or (L_sun / \AA) to observed-frame `unit`.
        
        If ``target_stellar_mass``, ``target_sfr``, or ``target_lir`` 
        specified, then scale the output to the desired value using the 
        intrinsic properties.  Units are linear ``Msun``, ``Msun/yr``, 
        and ``Lsun``, respectively.
        """
        from astropy.constants import L_sun

        if zred == None:
            zred = self.params['zred']
            if verbose:
                msg = 'continuum_to_obsframe: Use params[zred] = {0:.3f}'
                print(msg.format(zred))
                    
        if cosmology is None:
            cosmology = self.cosmology
        else:
            self.cosmology = cosmology
        
        if zred <= 0:
            dL = 1*u.cm
        else:
            dL = cosmology.luminosity_distance(zred).to(u.cm)
        
        # FSPS L_sun / Hz to observed-frame
        try:
            # Unit is like f-lambda
            _x = (1*unit).to(u.erg/u.second/u.cm**2/u.Angstrom) 
            is_flam = True
            obs_unit = (1*L_sun/u.Angstrom/(4*np.pi*dL**2)).to(unit)/(1+zred)
        except:
            # Unit is like f-nu
            is_flam = False
            obs_unit = (1*L_sun/u.Hz/(4*np.pi*dL**2)).to(unit)*(1+zred)
        
        if target_stellar_mass is not None:
            obs_unit *= target_stellar_mass / self.stellar_mass
        elif target_sfr is not None:
            obs_unit *= target_sfr / self.sfr100
        elif target_lir is not None:
            obs_unit *= target_lir / self.energy_absorbed
            
        return obs_unit.value
        
    def fit_phot(self, phot_dict, filters=None, flux_unit=u.microJansky, plist=['tage', 'Av', 'gas_logu', 'sigma_smooth'], func_kwargs={'lorentz':False}, verbose=True, lsq_kwargs={'method':'trf', 'max_nfev':200, 'loss':'huber', 'x_scale':1.0, 'verbose':True}, show=False, TEF=None, photoz_obj=None):
        """
        Fit models to observed spectrum
        """
        from scipy.optimize import least_squares
        import matplotlib.pyplot as plt
        import grizli.utils
        
        sys_err = 0.02
        
        flux = phot_dict['fobs']
        err = phot_dict['efobs']
        if 'flux_unit' in phot_dict:
            flux_unit = phot_dict['flux_unit']
            
        x0 = np.array([self.meta[p] for p in plist])
        
        # Are input fluxes f-lambda or f-nu?
        try:
            _x = (1*flux_unit).to(u.erg/u.second/u.cm**2/u.Angstrom) 
            is_flam = True
        except:
            is_flam = False
         
        # Initialize keywords   
        kwargs = func_kwargs.copy()
        for i_p, p in enumerate(plist):
            kwargs[p] = x0[i_p]
        
        # Initial model
        margs = (self, plist, flux, err, is_flam, filters, TEF, kwargs, 'model')        
        flux_model, Anorm, chi2_init, templ = self.objfun_fitphot(x0, *margs)
        
        if show:                
            
            if hasattr(show, 'plot'):
                ax = show
            else:
                plt.close('all')
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                
            mask = err > 0
            pivot = np.array([f.pivot for f in filters])
            so = np.argsort(pivot)
            
            ax.errorbar(pivot[mask]/1.e4, flux[mask], err[mask],
                        color='k', alpha=0.5, linestyle='None', marker='.')
            ax.scatter(pivot[so]/1.e4, flux_model[so], color='pink', 
                       alpha=0.8, zorder=100)
        
        # Parameter bounds    
        bounds, steps = self.parameter_bounds(plist)
        lsq_kwargs['diff_step'] = steps
        
        # Run the optimization
        lmargs = (self, plist, flux, err, is_flam, filters, TEF, kwargs, 'least_squares verbose')        
        _res = least_squares(self.objfun_fitphot, x0, bounds=bounds,
                             args=lmargs, **lsq_kwargs)
        
        _out = self.objfun_fitphot(_res.x, *margs)
        
        xtempl = _out[3]
        xscale = _out[1]
        
        _fit = {}
        _fit['fmodel'] = _out[0]
        _fit['scale'] = xscale
        _fit['chi2'] = _out[2]
        _fit['templ'] = xtempl
        _fit['plist'] = plist
        _fit['theta'] = _res.x
        _fit['res'] = _res
        
        # Stellar mass
        #fit_model, Anorm, chi2_fit, templ = _phot
        
        # Parameter scaling to observed frame.  
        # e.g., stellar mass = self.stellar_mass * scale / to_obsframe
        z = self.params['zred']
        _obsfr = self.continuum_to_obsframe(zred=z, unit=flux_unit)
        _fit['to_obsframe'] = _obsfr
        
        scl = _fit['scale']/_fit['to_obsframe']
        _fit['log_mass'] = np.log10(self.stellar_mass*scl)
        _fit['sfr'] = self.sfr*scl
        _fit['sfr10'] = self.sfr10*scl
        _fit['sfr100'] = self.sfr100*scl
        age_bands = ['i1500','v']
        ages = self.light_age_band(bands=age_bands)
        for i, b in enumerate(age_bands):
            _fit['age_'+b] = ages[i]
            
        if show:
            ax.scatter(pivot[so]/1.e4, _fit['fmodel'][so], color='r', 
                      alpha=0.8, zorder=101)
            
            iz = templ.zindex(z)
            igm = templ.igm_absorption(z, scale_tau=1.4)
            
            if is_flam:
                ax.plot(xtempl.wave*(1+z)/1.e4, xtempl.flux[iz,:]*xscale*igm, 
                    color='r', alpha=0.3, zorder=10000)
            else:
                ax.plot(xtempl.wave*(1+z)/1.e4, xtempl.flux_fnu(iz)*xscale*igm, 
                    color='r', alpha=0.3, zorder=10000)
        
        return _fit
        
    @staticmethod
    def objfun_fitphot(theta, self, plist, flux_fnu, err_fnu, is_flam, filters, TEF, kwargs, ret_type):
        """
        Objective function for fitting spectra
        """
        try:
            from grizli.utils_c.interp import interp_conserve_c as interp_func
        except:
            interp_func = utils.interp_conserve
            
        for i, p in enumerate(plist):
            kwargs[p] = theta[i]
                    
        templ = self.get_full_spectrum(**kwargs)
        model_fnu = templ.integrate_filter_list(filters,
                                     z=self.params['zred'], flam=is_flam, 
                                     include_igm=True)
        
        mask = (err_fnu > 0)
        full_var = err_fnu**2
        
        if TEF is not None:
            tefz = TEF(self.params['zred'])
            full_var += (flux_fnu*tefz)**2
            
        lsq_num = (flux_fnu*model_fnu/full_var)[mask].sum()
        lsq_den = (model_fnu**2/full_var)[mask].sum()
        Anorm = lsq_num/lsq_den
        model_fnu *= Anorm
        
        chi = ((model_fnu - flux_fnu)/np.sqrt(full_var))[mask]
        chi2 = (chi**2).sum()
        
        if 'verbose' in ret_type:
            print('{0} {1:.4f}'.format(theta, (chi**2).sum()))
            
        if 'model' in ret_type:
            return model_fnu, Anorm, chi2, templ
        elif 'least_squares' in ret_type:
            return chi
        elif 'logpdf' in ret_type:
            return -chi2/2
        else:
            return chi2


    def fit_grism(self, mb, plist=['tage', 'zred'], func_kwargs={'lorentz':False}, verbose=True, lsq_kwargs={'method':'trf', 'max_nfev':200, 'loss':'huber', 'x_scale':1.0, 'verbose':True}, show=False, TEF=None, photoz_obj=None, flux_unit=u.erg/u.second/u.cm**2/u.Angstrom):
        """
        Fit models to grism spectrum
        """
        from scipy.optimize import least_squares
        import matplotlib.pyplot as plt
        import grizli.utils

        x0 = np.array([self.meta[p] for p in plist])

        # Initialize keywords   
        kwargs = func_kwargs.copy()
        for i_p, p in enumerate(plist):
            kwargs[p] = x0[i_p]

        # Initial model
        margs = (self, plist, mb, TEF, kwargs, 'model')        
        tfit, Anorm, chi2_init, templ = self.objfun_fitgrism(x0, *margs)

        if show:                
            _init_fig = mb.oned_figure(tfit=tfit, show_individual_templates=True)

        # Parameter bounds    
        bounds, steps = self.parameter_bounds(plist)
        lsq_kwargs['diff_step'] = steps

        # Run the optimization
        lmargs = (self, plist, mb, TEF, kwargs, 'least_squares verbose')        
        _res = least_squares(self.objfun_fitgrism, x0, bounds=bounds,
                             args=lmargs, **lsq_kwargs)

        _out = self.objfun_fitgrism(_res.x, *margs)

        xtempl = _out[3]
        xscale = _out[1]

        _fit = {}
        _fit['fmodel'] = _out[0]
        _fit['scale'] = xscale
        _fit['chi2'] = _out[2]
        _fit['templ'] = xtempl
        _fit['plist'] = plist
        _fit['theta'] = _res.x
        _fit['res'] = _res

        # Stellar mass
        #fit_model, Anorm, chi2_fit, templ = _phot

        # Parameter scaling to observed frame.  
        # e.g., stellar mass = self.stellar_mass * scale / to_obsframe
        z = self.params['zred']
        _obsfr = self.continuum_to_obsframe(zred=z, unit=flux_unit)
        _fit['to_obsframe'] = _obsfr

        scl = _fit['scale']/_fit['to_obsframe']
        _fit['log_mass'] = np.log10(self.stellar_mass*scl)
        _fit['sfr'] = self.sfr*scl
        _fit['sfr10'] = self.sfr10*scl
        _fit['sfr100'] = self.sfr100*scl
        age_bands = ['i1500','v']
        ages = self.light_age_band(bands=age_bands)
        for i, b in enumerate(age_bands):
            _fit['age_'+b] = ages[i]

        _tfit = _out[0]

        if show:
            _fit_fig = mb.oned_figure(tfit=_tfit, show_individual_templates=True)
            _figs = (_init_fig, _fit_fig)
        else:
            _figs = None

        return _fit, _tfit, _figs

    @staticmethod
    def objfun_fitgrism(theta, self, plist, mb, TEF, kwargs, ret_type):
        """
        Objective function for fitting spectra
        """
        try:
            from grizli.utils_c.interp import interp_conserve_c as interp_func
        except:
            interp_func = utils.interp_conserve

        from grizli import utils

        for i, p in enumerate(plist):
            kwargs[p] = theta[i]

        templ = self.get_full_spectrum(**kwargs)
        tsp = utils.SpectrumTemplate(wave=templ.wave, flux=templ.flux_flam())
        tx = {'fsps':tsp}

        if 'least_squares' in ret_type:
            out = mb.xfit_at_z(z=self.params['zred'],
                               templates=tx, fitter='lstsq',
                               fit_background=True, get_uncertainties=False,
                               include_photometry=False, get_residuals=True,
                               use_cached_templates=False)

            chi, _coeffs, _, _ = out
            chi2 = (chi**2).sum()
            Anorm = _coeffs[-1]
        else:
            tfit = mb.template_at_z(z=self.params['zred'], templates=tx)
            chi2 = tfit['chi2']
            Anorm = tfit['cfit']['fsps'][0]

        if 'verbose' in ret_type:
            print('{0} {1:.4f}'.format(theta, chi2))

        if 'model' in ret_type:
            return tfit, Anorm, chi2, templ
        elif 'least_squares' in ret_type:
            return chi
        elif 'logpdf' in ret_type:
            return -chi2/2
        else:
            return chi2
