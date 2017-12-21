import os
import numpy as np

#import unicorn

__all__ = ["TemplateError", "Template"]

class TemplateError():
    """
    Make an easy (spline) interpolator for the template error function
    """
    def __init__(self, file='templates/TEMPLATE_ERROR.eazy_v1.0', arrays=None, lc=[5500.], scale=1.):
        from scipy import interpolate
        self.file = file
        if arrays is None:
            self.te_x, self.te_y = np.loadtxt(file, unpack=True)
        else:
            self.te_x, self.te_y = arrays
            
        self.scale = scale
        self._spline = interpolate.CubicSpline(self.te_x, self.te_y)
        self.lc = lc
        
    def interpolate(self, filter_wavelength=5500., z=1.):
        """
        observed_wavelength is observed wavelength of photometric filters.  But 
        these sample the *rest* wavelength of the template error function at lam/(1+z)
        """
        return self._spline(filter_wavelength/(1+z))*self.scale
    
    def __call__(self, z):
        return self._spline(self.lc/(1+z))*self.scale
        
class Template():
    def __init__(self, sp=None, file=None, name=None, arrays=None):
        self.wave = None
        self.flux = None
        self.flux_fnu = None
        self.name = 'None'
        if name is None:
            if file is not None:
                self.name = os.path.basename(file)
        else:
            self.name = name
                
        if sp is not None:
            self.wave = np.cast[np.double](sp.wave)
            self.flux = np.cast[np.double](sp.flux)
            self.flux_fnu = self.flux
            
        if file is not None:
            self.wave, self.flux = np.loadtxt(file, unpack=True)
            self.set_fnu()
        
        if arrays is not None:
            self.wave, self.flux = arrays
            self.set_fnu()
            
    def set_fnu(self):
        self.flux_fnu = self.flux * self.wave**2 / 3.e18
        
    def integrate_filter(self, filter, scale=1., z=0):
        """
        Integrate the template through a `FilterDefinition` filter object.
        
        The `grizli` interpolation module should be used if possible: 
        https://github.com/gbrammer/grizli/
        """
        try:
            import grizli.utils_c
            interp = grizli.utils_c.interp.interp_conserve_c
        except ImportError:
            interp = np.interp
        
        templ_filter = interp(filter.wave, self.wave*(1+z),
                              self.flux_fnu*scale)
                
        # f_nu/lam dlam == f_nu d (ln nu)    
        integrator = np.trapz
        temp_int = integrator(filter.throughput*templ_filter/filter.wave, filter.wave) / filter.norm
        
        return temp_int
        
# class TemplateInterpolator():
#     """
#     Class to use scipy spline interpolator to interpolate pre-computed eazy template 
#     photometry at arbitrary redshift(s).
#     """
#     def __init__(self, bands=None, MAIN_OUTPUT_FILE='photz', OUTPUT_DIRECTORY='./OUTPUT', CACHE_FILE='Same', zout=None, f_lambda=True):
#         from scipy import interpolate
#         #import threedhst.eazyPy as eazy
#         
#         #### Read the files from the specified output
#         tempfilt, coeffs, temp_seds, pz = eazy.readEazyBinary(MAIN_OUTPUT_FILE=MAIN_OUTPUT_FILE, OUTPUT_DIRECTORY=OUTPUT_DIRECTORY, CACHE_FILE = CACHE_FILE)
#         
#         if bands is None:
#             self.bands = np.arange(tempfilt['NFILT'])
#         else:
#             self.bands = np.array(bands)
#         
#         self.band_names = ['' for b in self.bands]
#         
#         if zout is not None:
#             param = eazy.EazyParam(PARAM_FILE=zout.filename.replace('.zout','.param'))
#             self.band_names = [f.name for f in param.filters]
#             self.bands = np.array([f.fnumber-1 for f in param.filters])
#                         
#         self.NFILT = len(self.bands)
#         self.NTEMP = tempfilt['NTEMP']
#         self.lc = tempfilt['lc'][self.bands]
#         self.sed = temp_seds
#         self.templam = self.sed['templam']
#         self.temp_seds = self.sed['temp_seds']
#         
#         # if True:
#         #     import threedhst
#         #     import unicorn
#         #     threedhst.showMessage('Conroy model', warn=True)
#         #     cvd12 = np.loadtxt(unicorn.GRISM_HOME+'/templates/cvd12_t11_solar_Chabrier.dat')
#         #     self.temp_seds[:,0] = np.interp(self.templam, cvd12[:,0], cvd12[:,1])
#         
#         self.in_zgrid = tempfilt['zgrid']
#         self.tempfilt = tempfilt['tempfilt'][self.bands, :, :]
#         if f_lambda:
#             for i in range(self.NFILT):
#                 self.tempfilt[i,:,:] /= (self.lc[i]/5500.)**2
#                 
#         ###### IGM absorption
#         self.igm_wave = []
#         self.igm_wave.append(self.templam < 912)
#         self.igm_wave.append((self.templam >= 912) & (self.templam < 1026))
#         self.igm_wave.append((self.templam >= 1026) & (self.templam < 1216))
#         
#         self._spline_da = interpolate.InterpolatedUnivariateSpline(self.in_zgrid, temp_seds['da'])
#         self._spline_db = interpolate.InterpolatedUnivariateSpline(self.in_zgrid, temp_seds['db'])
#         
#         #### Make a 2D list of the spline interpolators
#         self._interpolators = [range(self.NTEMP) for i in range(self.NFILT)]                
#         for i in range(self.NFILT):
#             for j in range(self.NTEMP):
#                 self._interpolators[i][j] = interpolate.InterpolatedUnivariateSpline(self.in_zgrid, self.tempfilt[i, j, :])
#         #
#         self.output = None
#         self.zout = None
#     
#     def interpolate_photometry(self, zout):
#         """
#         Interpolate the EAZY template photometry at `zout`, which can be a number or an 
#         array.
#         
#         The result is returned from the function and also stored in `self.output`.
#         """               
#         output = [range(self.NTEMP) for i in range(self.NFILT)]                
#         for i in range(self.NFILT):
#             for j in range(self.NTEMP):
#                 output[i][j] = self._interpolators[i][j](zout)
#         
#         self.zgrid = np.array(zout)
#         self.output = np.array(output)
#         return self.output
#         
#     def check_extrapolate(self):
#         """
#         Check if any interpolated values are extrapolated from the original redshift grid
#         
#         Result is both returned and stored in `self.extrapolated`
#         """
#         if self.zout is None:
#             return False
#         
#         self.extrapolated = np.zeros(self.output.shape, dtype=np.bool) ## False
# 
#         bad = (self.zgrid < self.in_zgrid.min()) | (self.zgrid > self.in_zgrid.max())
#         self.extrapolated[:, :, bad] = True
#         
#         return self.extrapolated
#         
#     def get_IGM(self, z, matrix=False, silent=False):
#         """
#         Retrieve the full SEDs with IGM absorption
#         """
#         ###### IGM absorption
#         # lim1 = self.templam < 912
#         # lim2 = (self.templam >= 912) & (self.templam < 1026)
#         # lim3 = (self.templam >= 1026) & (self.templam < 1216)
#         
#         igm_factor = np.ones(self.templam.shape[0])
#         igm_factor[self.igm_wave[0]] = 0.
#         igm_factor[self.igm_wave[1]] = 1. - self._spline_db(z)
#         igm_factor[self.igm_wave[1]] = 1. - self._spline_da(z)
#         
#         if matrix:
#             self.igm_factor = np.dot(igm_factor.reshape(-1,1), np.ones((1, self.NTEMP)))
#         else:
#             self.igm_factor = igm_factor
#             
#         self.igm_z = z
#         self.igm_lambda = self.templam*(1+z)
#         
#         if not silent:
#             return self.igm_lambda, self.igm_factor

