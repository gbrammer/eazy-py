import os
import numpy as np

import astropy.units as u

from . import utils

#import unicorn

__all__ = ["TemplateError", "Template"]

class TemplateError():
    """
    Make an easy (spline) interpolator for the template error function
    """
    def __init__(self, file='templates/TEMPLATE_ERROR.eazy_v1.0', arrays=None, lc=[5500.], scale=1.):
        self.file = file
        if arrays is None:
            self.te_x, self.te_y = np.loadtxt(file, unpack=True)
        else:
            self.te_x, self.te_y = arrays
                   
        self.scale = scale
        self.lc = lc
        self._set_limits()
        self._init_spline()
    
    def _set_limits(self):
        """
        Limits to control extrapolation
        """
        nonzero = self.te_y > 0
        self.min_wavelength = self.te_x[nonzero].min()
        self.max_wavelength = self.te_x[nonzero].max()
        
    def _init_spline(self):
        from scipy import interpolate
        self._spline = interpolate.CubicSpline(self.te_x, self.te_y)
        
    def interpolate(self, filter_wavelength=5500., z=1.):
        """
        observed_wavelength is observed wavelength of photometric filters.  But 
        these sample the *rest* wavelength of the template error function at lam/(1+z)
        """
        return self._spline(filter_wavelength/(1+z))*self.scale
    
    def __call__(self, z):
        lcz = np.array(self.lc)/(1+z)
        tef_z = self._spline(self.lc/(1+z))*self.scale 
        clip = (lcz < self.min_wavelength) | (lcz > self.max_wavelength)
        tef_z[clip] = 0.
        
        return tef_z

class Redden():
    """
    Wrapper function for `~dust_attenuation` and `~dust_extinction` 
    reddening laws
    """
    def __init__(self, model=None, Av=0., **kwargs):
        """
        model: extinction/attenuation object or str
            
            Allowable string arguments:
                
                'smc' = `~dust_extinction.averages.G03_SMCBar`
                'lmc' = `~dust_extinction.averages.G03_LMCAvg`
                'mw','f99' = `~dust_extinction.parameter_averages.F99`
                'calzetti00', 'c00' = `~dust_attenuation.averages.C00`
                'wg00' = '~dust_attenuation.radiative_transfer.WG00`
        
        Av: selective extinction/attenuation
            (passed as tau_V for `WG00`)
            
        """
        allowed = ['smc', 'lmc', 'mw', 'f99', 'c00', 'calzetti00', 'wg00']
        
        if isinstance(model, str):
            if model in ['smc']:
                from dust_extinction.averages import G03_SMCBar
                self.model = G03_SMCBar()
            elif model in ['lmc']:
                from dust_extinction.averages import G03_LMCAvg
                self.model = G03_LMCAvg()
            elif model in ['mw','f99']:
                from dust_extinction.parameter_averages import F99 
                self.model = F99()
            elif model in ['calzetti00', 'c00']:
                from dust_attenuation.averages import C00
                self.model = C00(Av=Av)
            elif model in ['wg00']:
                from dust_attenuation.radiative_transfer import WG00
                if 'tau_V' in kwargs:
                    self.model = WG00(**kwargs)
                else:
                    self.model = WG00(tau_V=Av, **kwargs)   
            else:
                msg = "Requested model ('{model}') not in {allowed}."          
                raise IOError(msg.format(model=model, allowed=allowed))
        else:
            self.model = model
        
        for k in ['Av', 'tau_V']:
            if hasattr(model, k):
                Av = getattr(model, k)
                break
                
        self.Av = Av
    
    @property 
    def ebv(self):
        if hasattr(self.model, 'Rv'):
            return self.Av/self.model.Rv
        else:
            print('Warning: Rv not defined for model: ' + self.__repr__())
            return 0.
            
    def __repr__(self):
        return '{0}, Av/tau_V={1}'.format(self.model.__repr__(), self.Av)
        
    def __call__(self, wave, left=0, right=1., **kwargs):
        """
        Return reddening factor.  If input has no units, assume 
        `~astropy.units.Angstrom`.
        """
                    
        if not hasattr(wave, 'unit'):
            xu = wave*u.Angstrom
        else:
            if wave.unit is None:
                xu.unit = u.Angstrom
            else:
                xu = wave
        
        if 'Av' in kwargs:
            self.Av = kwargs['Av']
        
        if 'tau_V' in kwargs:
            self.Av = kwargs['tau_V']
        
        for k in kwargs:
            if hasattr(self.model, k):
                setattr(self.model, k, kwargs[k])
        
        ext = np.atleast_1d(np.ones_like(xu.value))
        
        if hasattr(self.model, 'x_range'):
            if hasattr(self.model, 'extinguish'):
                # dust_extinction has x_range in 1/micron
                xblue = (1./xu.to(u.micron)).value > self.model.x_range[1]
                xred = (1./xu.to(u.micron)).value < self.model.x_range[0]
            else:
                # dust_attenuation has x_range in micron
                xblue = (xu.to(u.micron)).value < self.model.x_range[0]
                xred = (xu.to(u.micron)).value > self.model.x_range[1]
                
            ext[xblue] = left
            ext[xred] = right
            xr = (~xblue) & (~xred)
        else:
            xr = np.isfinite(wave)    
        
        if (self.model is None) | (self.Av <= 0):
            # Don't do anything
            pass             
        elif hasattr(self.model, 'extinguish'):
            # extinction
            ext[xr] = self.model.extinguish(xu[xr], Av=self.Av)
        elif hasattr(self.model, 'attenuate'):
            # attenuation
            if hasattr(self.model, 'tau_V'):
                # WG00
                self.model.tau_V = self.Av
            else:
                self.model.Av = self.Av
            
            ext[xr] = self.model.attenuate(xu[xr])
        else:
            msg = ('Dust model must have either `attenuate` or `extinguish`'
                  ' method.')
            raise IOError(msg)
        
        if hasattr(wave, '__len__'):
            return ext
        elif ext.size == 1:
            return ext[0]
        else:
            return ext
            
class Template():
    def __init__(self, sp=None, file=None, name=None, arrays=None, meta={}, to_angstrom=1., velocity_smooth=0, norm_filter=None, resample_wave=None, fits_column='flux', redfunc=Redden()):
        """
        Template object.
        
        Attributes:
        
        wave = wavelength in `~astropy.units.Angstrom`
        flux = flux density, f-lambda
        name = str
        meta = dict
        redfunc = optional `Redden` object
        
        Properties: 
        
        flux_fnu = flux density, f-nu
        
        """
        from astropy.table import Table
        import astropy.units as u
        
        self.wave = None
        self.flux = None
        
        self.name = 'None'
        self.meta = meta
                
        self.velocity_smooth = velocity_smooth
        
        if name is None:
            if file is not None:
                self.name = os.path.basename(file)
        else:
            self.name = name
        
        if sp is not None:
            # Prospector        
            self.wave = np.cast[np.float](sp.wave)
            self.flux = np.cast[np.float](sp.flux)
            # already fnu
            self.flux *= utils.CLIGHT*1.e10 / self.wave**2
            
        elif file is not None:
            # Read from a file
            if file.split('.')[-1] in ['fits','csv','ecsv']:
                tab = Table.read(file)
                self.wave = tab['wave'].data.astype(np.float)
                self.flux = tab[fits_column].data.astype(np.float)
                for k in tab.meta:
                    self.meta[k] = tab.meta[k]
                    
            else:
                _arr = np.loadtxt(file, unpack=True)
                self.wave, self.flux = _arr[0], _arr[1]
                        
        elif arrays is not None:
            self.wave, self.flux = arrays[0]*1, arrays[1]*1
            self.set_fnu()
        else:
            raise TypeError('Must specify either `sp`, `file` or `arrays`')
        
        # Handle optional units
        if hasattr(self.wave, 'unit'):
            if self.wave.unit is not None:
                self.wave = self.wave.to(u.Angstrom).value
            else:
                self.wave = self.wave.data
        else:
            self.wave *= to_angstrom

        flam_unit = u.erg/u.second/u.cm**2/u.Angstrom
        
        if hasattr(self.flux, 'unit'):
            if self.flux.unit is not None:
                equiv = u.equivalencies.spectral_density(self.wave*u.Angstrom)
                flam = self.flux.to(flam_unit, equivalencies=equiv) 
                self.flux = flam.value
            else:
                self.flux = self.flux.data
                
        # Smoothing   
        if velocity_smooth > 0:
            self.smooth_velocity(velocity_smooth, in_place=True)
        
        # Resampling
        self.resample(resample_wave, in_place=True)
                
        #self.set_fnu()
         
        # Reddening function
        self.redfunc = redfunc
        _red = self.redden # test to break at init if fails
            
    def __repr__(self):
        if self.name is None:
            return self.__class__
        else:
            return '{0}: {1}'.format(self.__class__, self.name)
    
    @property
    def absorbed_energy(self):
        diff = self.flux*(1-self.redden)*(self.redden > 0)
        return np.trapz(diff, self.wave)
        
    @property
    def redden(self):
        """
        Return multiplicative scaling from `self.redfunc`, which is expected
        to return attenuation in magnitudes.
        """
        if self.redfunc is not None:
            red = self.redfunc(self.wave*u.Angstrom)
        else:
            red = 1.
        
        return red
        
    @property 
    def flux_fnu(self):
        """
        self.flux is flam.  Scale to fnu
        """
        return self.flux * self.wave**2 / (utils.CLIGHT*1.e10) * self.redden
                    
    def set_fnu(self):
        """
        Deprecated.  `flux_fnu` is now a `@property`.
        """
        pass
        #self.flux_fnu = self.flux * self.wave**2 / 3.e18
    
    def smooth_velocity(self, velocity_smooth, in_place=True, raise_error=False):
        """
        Smooth template in velocity using `prospect`
        """
        try:
            from prospect.utils.smoothing import smooth_vel
        except:
            if raise_error:
                raise ImportError("Couldn't import `prospect.utils.smoothing")
            else:
                return None
        
        if velocity_smooth <= 0:
            if in_place:
                return True
            else:
                return self
                
        sm_flux = smooth_vel(self.wave, self.flux, self.wave, 
                             velocity_smooth)
        sm_flux[~np.isfinite(sm_flux)] = 0.
        
        if in_place:
            self.flux_orig = self.flux*1
            self.velocity_smooth = velocity_smooth
            self.flux = sm_flux
            return True
        else:
            return Template(arrays=(self.wave, sm_flux), name=self.name, 
                            meta=self.meta)
            
    def resample(self, new_wave, z=0, in_place=True, return_array=False, interp_func=utils.interp_conserve):
        """
        Resample the template to a new wavelength grid
        """
        import astropy.units as u
        
        breakme = False
        if isinstance(new_wave, str):
            if new_wave == 'None':
                breakme = True
            elif not os.path.exists(new_wave):
                msg = 'WARNING: new_wave={0} could not be found'
                print(msg.format(new_wave))
                breakme = True
            else:
                new_wave = np.loadtxt(new_wave)
        
        elif new_wave is None:
            breakme = True
            
        if breakme:
            if in_place:
                return False
            else:
                return self
                
        if hasattr(new_wave, 'unit'):
            new_wave = new_wave.to(u.Angstrom).value
            
        new_flux = interp_func(new_wave, self.wave*(1+z), self.flux)
        if in_place:
            self.wave = new_wave*1
            self.flux = new_flux
            return True
        else:
            if return_array:
                return new_flux
            else:
                return Template(arrays=(new_wave, new_flux), name=self.name, 
                            meta=self.meta)
                
    def integrate_filter(self, filt, flam=False, scale=1., z=0, include_igm=False):
        """
        Integrate the template through a `FilterDefinition` filter object.
        
        The `grizli` interpolation module should be used if possible: 
        https://github.com/gbrammer/grizli/
        """
        try:
            import grizli.utils_c
            interp = grizli.utils_c.interp.interp_conserve_c
        except ImportError:
            interp = utils.interp_conserve
        
        if hasattr(filt, '__len__'):
            filts = filt
            single = False
        else:
            filts = [filt]
            single = True
        
        if include_igm > 0:
            igmz = self.igm_absorption(z, pow=include_igm)
        else:
            igmz = 1.
        
        # Fnu flux density, with IGM and scaling
        fnu = self.flux_fnu*scale*igmz
        
        fluxes = []
        for filt_i in filts:    
            templ_filt = interp(filt_i.wave, self.wave*(1+z),
                                fnu, left=0, right=0)
                
            # f_nu/lam dlam == f_nu d (ln nu)    
            integrator = np.trapz
            temp_int = integrator(filt_i.throughput*templ_filt/filt_i.wave, 
                                  filt_i.wave) 
            temp_int /= filt_i.norm
        
            if flam:
                temp_int *= utils.CLIGHT*1.e10/(filt_i.pivot/(1+z))**2
            
            fluxes.append(temp_int)
        
        if single:
            return fluxes[0]
        else:
            return np.array(fluxes)
                    
    def igm_absorption(self, z, pow=1):
        """
        Compute IGM absorption.  
        
        `power` scales the absorption strength as `~eazy.igm.Inoue14()**pow`.
        """
        try:
            from . import igm as igm_module
        except:
            from eazy import igm as igm_module
        
        igm = igm_module.Inoue14()
        igmz = self.wave*0.+1
        lyman = self.wave < 1300
        igmz[lyman] = igm.full_IGM(z, (self.wave*(1+z))[lyman])**pow
        return igmz
        
    def integrate_filter_list(self, filters, include_igm=True, norm_index=None, **kwargs):
        """
        Integrate template through all filters
        
        filters: list of `~eazy.filters.Filter` objects
        
        [rewritten as simple wrapper]
        """                    
        fluxes = self.integrate_filter(filters, include_igm=include_igm, 
                                       **kwargs)        
        if isinstance(norm_index, int):
            fluxes /= fluxes[norm_index]
            
        return fluxes
        
    def to_table(self, formats={'wave':'.5e', 'flux':'.5e'}, with_units=False):
        from astropy.table import Table
        import astropy.units as u
        
        tab = Table()
        tab['wave'] = self.wave
        tab['flux'] = self.flux
        
        if with_units:
            tab['wave'].unit = u.Angstrom
            tab['flux'].unit = u.erg/u.second/u.cm**2/u.Angstrom

        for c in tab.colnames:
            if c in formats:
                tab[c].format = formats[c]
                
        tab.meta = self.meta
        return tab
        
class ModifiedBlackBody():
    """
    Modified black body: nu**beta * BB(nu) 
    + FIR-radio correlation
    
    
    """
    def __init__(self, Td=47, beta=1.6, q=2.34, alpha=-0.75):
        self.Td = Td
        self.q = q
        self.beta = beta
        self.alpha = alpha
    
    @property 
    def bb(self):
        from astropy.modeling.models import BlackBody
        return BlackBody(temperature=self.Td*u.K)
        
    def __call__(self, wave, q=None):
        """
        Return modified BlackBody (fnu) as a function of wavelength
        """
        from astropy.constants import L_sun, h, k_B, c
        if not hasattr(wave, 'unit'):
            mu = wave*u.micron
        else:
            mu = wave
                 
        nu = (c/mu).to(u.Hz).value
        mbb = (self.bb(nu)*nu**self.beta).value
        lim = (mu > 40*u.micron) & (mu < 120*u.micron) 
        lir = np.trapz(mbb[lim][::-1], nu[lim][::-1])
        
        if q is None:
            q = self.q
            
        radio = 10**(np.log10(lir/3.75e12)-q)
        radio *= (nu/1.4e9)**self.alpha
        
        return (mbb+radio)
    
    def __repr__(self):
        label = r'$T_\mathrm{{{{d}}}}$={0:.0f}, $\beta$={1:.1f}'
        return label.format(self.Td, self.beta)
        
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

