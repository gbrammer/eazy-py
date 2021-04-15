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
        allowed = ['smc', 'lmc', 'mw', 'f99', 'c00', 'calzetti00', 'wg00',
                   'kc13','reddy15','zafar15']
                
        if isinstance(model, str):
            self.model_name = model
        
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
            elif model.lower() in ['kc13']:
                from eazy.sps import KC13
                self.model = KC13(Av=Av, **kwargs)
            elif model.lower() in ['reddy15']:
                from eazy.sps import Reddy15
                self.model = Reddy15(Av=Av, **kwargs)
            elif model.lower() in ['zafar15']:
                from eazy.sps import Zafar15
                self.model = Zafar15(Av=Av)
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
            self.model_name = 'Unknown'
            
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
        return '<Redden {0}, Av/tau_V={1}>'.format(self.model.__repr__(), self.Av)
        
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
    def __init__(self, sp=None, file=None, name=None, arrays=None, meta={}, to_angstrom=1., velocity_smooth=0, norm_filter=None, resample_wave=None, fits_column='flux', redfunc=Redden(), template_redshifts=[0], verbose=True):
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
        
        Can optionally specify a 2-dimensional flux array with the first
        dimension indicating the template for the nearest redshift in the 
        correspoinding ``template_redshifts`` list.  When integrating the 
        filter fluxes with ``integrate_filter``, the template index with the 
        redshift nearest to the specified redshift will be used.
        
        """
        import copy
        from astropy.table import Table
        import astropy.units as u
        
        self.wave = None
        self.flux = None
        
        self.name = 'None'
        self.meta = copy.deepcopy(meta)
                
        self.velocity_smooth = velocity_smooth
        
        if name is None:
            if file is not None:
                self.name = os.path.basename(file)
        else:
            self.name = name
        
        self.orig_table = None
        
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
                self.orig_table = tab
                
                # Transpose because FITS tables stored like NWAVE, NZ
                if self.flux.ndim == 2:
                    self.flux = self.flux.T
                    
                for k in tab.meta:
                    self.meta[k] = tab.meta[k]
                    
            else:
                _arr = np.loadtxt(file, unpack=True)
                self.wave, self.flux = _arr[0], _arr[1]
                        
        elif arrays is not None:
            self.wave, self.flux = arrays[0]*1., arrays[1]*1.
            #self.set_fnu()
        else:
            raise TypeError('Must specify either `sp`, `file` or `arrays`')
        
        if self.flux.ndim == 1:
            # For redshift dependence
            self.flux = np.atleast_2d(self.flux)
            self.template_redshifts = np.zeros(1)
            self.NZ, self.NWAVE = self.flux.shape
        else:
            self.NZ, self.NWAVE = self.flux.shape
            if 'NZ' in self.meta:
                template_redshifts = [self.meta[f'Z{j}'] 
                                      for j in range(self.meta['NZ'])]
            
            if len(template_redshifts) != self.NZ:
                msg = (f'template_redshifts ({len(template_redshifts)})'
                       f' doesn\'t match flux dimension ({self.NZ})!')
                raise ValueError(msg)
            
            self.template_redshifts = np.array(template_redshifts)
        
            # if verbose:
            #     print(f'Redshift dependent! (NZ={self.NZ})')
                
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
    
    #@property
    def absorbed_energy(self, i=0):
        diff = self.flux[i,:]*(1-self.redden)*(self.redden > 0)
        absorbed = np.trapz(diff, self.wave)
        return absorbed
        # if self.NZ == 1:
        #     return absorbed[0]
        # else:
        #     return absorbed
               
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
        
    #@property 
    def flux_fnu(self, i=0):
        """
        self.flux is flam.  Scale to fnu
        """
        return self.flux[i,:] * self.wave**2 / (utils.CLIGHT*1.e10) * self.redden
                    
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
                
        sm_flux = np.array([smooth_vel(self.wave, self.flux[i,:], self.wave, 
                             velocity_smooth) for i in range(self.NZ)])
                             
        sm_flux[~np.isfinite(sm_flux)] = 0.
        
        if in_place:
            self.flux_orig = self.flux*1
            self.velocity_smooth = velocity_smooth
            self.flux = sm_flux
            return True
        else:
            return Template(arrays=(self.wave, sm_flux), 
                            name=self.name, meta=self.meta, 
                            template_redshifts=self.template_redshifts)
            
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
            
        new_flux = [interp_func(new_wave, self.wave*(1+z), self.flux[i,:])
                    for i in range(self.NZ)]
        new_flux = np.array(new_flux)
        
        if in_place:
            self.wave = new_wave*1
            self.flux = new_flux
            return True
        else:
            if return_array:
                return new_flux
            else:
                return Template(arrays=(new_wave, new_flux), 
                                name=self.name, meta=self.meta, 
                                template_redshifts=self.template_redshifts)
    
    def zindex(self, z=0., redshift_type='nearest'):
        """
        Get the redshift index of a multi-dimensional template array
        """
        #dz = z - self.template_redshifts
        
        zint = np.interp(z, self.template_redshifts, np.arange(self.NZ),
                         left=0, right=self.NZ-1)
                         
        if redshift_type == 'nearest':
            iz = np.round(zint).astype(int)
        else:
            iz = zint.astype(int)
                    
        return iz
        
    def integrate_filter(self, filt, flam=False, scale=1., z=0, include_igm=False, redshift_type='nearest', iz=None):
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
        if iz is None:
            iz = self.zindex(z=z, redshift_type=redshift_type)
        
        fnu = self.flux_fnu(iz)*scale*igmz
                        
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
                    
    def igm_absorption(self, z, scale_tau=1., pow=1):
        """
        Compute IGM absorption.  
        
        `power` scales the absorption strength as `~eazy.igm.Inoue14()**pow`.
        """
        try:
            from . import igm as igm_module
        except:
            from eazy import igm as igm_module
        
        igm = igm_module.Inoue14(scale_tau=scale_tau)
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
        
    def to_table(self, formats={'wave':'.5e', 'flux':'.5e'}, with_units=False, flatten=True):
        from astropy.table import Table
        import astropy.units as u
        import copy
        
        tab = Table()
        tab['wave'] = self.wave
        tab['flux'] = self.flux.T
        
        if with_units:
            tab['wave'].unit = u.Angstrom
            tab['flux'].unit = u.erg/u.second/u.cm**2/u.Angstrom

        for c in tab.colnames:
            if c in formats:
                tab[c].format = formats[c]
                
        tab.meta = copy.deepcopy(self.meta)
        if self.NZ > 1:
            tab.meta['NZ'] = self.NZ
            for j in range(self.NZ):
                tab.meta[f'Z{j}'] = self.template_redshifts[j]
        else:
            if flatten:
                tab['flux'] = self.flux[0,:]
                               
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


PHOENIX_LOGG_FULL = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
PHOENIX_LOGG = [4.0, 4.5, 5.0, 5.5]

PHOENIX_TEFF_FULL = [400.0, 420.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 1550.0, 1600.0, 1650.0, 1700.0, 1750.0, 1800.0, 1850.0, 1900.0, 1950.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0, 4300.0, 4400.0, 4500.0, 4600.0, 4700.0, 4800.0, 4900.0, 5000.0]

PHOENIX_TEFF = [400.,  420., 450., 500.,  550., 600.,  650., 700.,  750.,
       800.,  850., 900., 950., 1000., 1050., 1100., 1150., 1200.,
       1300., 1400., 1500., 1600., 1700., 1800., 1900., 2000., 2100.,
       2200., 2300., 2400., 2500., 2600., 2700., 2800., 2900., 3000.,
       3100., 3200., 3300., 3400., 3500., 3600., 3700., 3800., 3900., 4000.,
       4200., 4400., 4600., 4800., 5000., 5500., 5500, 6000., 6500., 7000.]

PHOENIX_ZMET_FULL = [-2.5, -2.0, -1.5, -1.0, -0.5, -0., 0.5]
PHOENIX_ZMET = [-1.0, -0.5, -0.]

def load_phoenix_stars(logg_list=PHOENIX_LOGG, teff_list=PHOENIX_TEFF, zmet_list=PHOENIX_ZMET, add_carbon_star=True, file='bt-settl_t400-7000_g4.5.fits'):
    """
    Load Phoenix stellar templates
    """
    try:
        from urllib.request import urlretrieve
    except:
        from urllib import urlretrieve

    from astropy.table import Table
    import astropy.io.fits as pyfits
    
    paths = ['/tmp', './templates/', './']
    hdu = None
    for path in paths:
        templ_path = os.path.join(path, file)
        if os.path.exists(templ_path):
            print(f'phoenix_templates: {templ_path}')
            hdu = pyfits.open(templ_path)
            break
    
    if hdu is None:
        url = 'https://s3.amazonaws.com/grizli/CONF'
        print('Fetch {0}/{1}'.format(url, file))

        #os.system('wget -O /tmp/{1} {0}/{1}'.format(url, file))
        res = urlretrieve('{0}/{1}'.format(url, file),
                          filename=templ_path)

        hdu = pyfits.open(templ_path)

    tab = Table.read(hdu[1])

    tstars = []
    N = tab['flux'].shape[1]
    for i in range(N):
        teff = tab.meta['TEFF{0:03d}'.format(i)]
        logg = tab.meta['LOGG{0:03d}'.format(i)]
        if 'ZMET{0:03d}'.format(i) in tab.meta:
            met = tab.meta['ZMET{0:03d}'.format(i)]
        else:
            met = 0.

        if (logg not in logg_list) | (teff not in teff_list) | (met not in zmet_list):
            #print('Skip {0} {1}'.format(logg, teff))
            continue

        label = 'bt-settl_t{0:05.0f}_g{1:3.1f}_m{2:.1f}'.format(teff, logg, met)
        arrays = (tab['wave'], tab['flux'][:, i])
        tstars.append(Template(arrays=arrays, name=label, redfunc=None))

    cfile = 'templates/stars/carbon_star.txt'
    if add_carbon_star & os.path.exists(cfile):
        sp = Table.read(cfile, format='ascii.commented_header')
        if add_carbon_star > 1:
            import scipy.ndimage as nd
            cflux = nd.gaussian_filter(sp['flux'], add_carbon_star)
        else:
            cflux = sp['flux']

        tstars.append(Template(arrays=(sp['wave'], cflux), 
                               name='carbon-lancon2002'))

    return tstars
            
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

def param_table(templates):
    """
    Try to generate parameters for a list of templates from their 
    metadata
    
    (TBD)
    """
    pass


def bspline_templates(wave, degree=3, df=6, get_matrix=True, log=False, clip=1.e-4, minmax=None):
    """
    B-spline basis functions, modeled after `~patsy.splines`
    """
    from collections import OrderedDict
    from scipy.interpolate import splev

    order = degree+1
    n_inner_knots = df - order
    inner_knots = np.linspace(0, 1, n_inner_knots + 2)[1:-1]

    norm_knots = np.concatenate(([0, 1] * order,
                                inner_knots))
    norm_knots.sort()

    if log:
        xspl = np.log(wave)
    else:
        xspl = wave*1

    if minmax is None:
        mi = xspl.min()
        ma = xspl.max()
    else:
        mi, ma = minmax

    width = ma-mi
    all_knots = norm_knots*width+mi

    n_bases = len(all_knots) - (degree + 1)
    basis = np.empty((xspl.shape[0], n_bases), dtype=float)

    coefs = np.identity(n_bases)
    basis = splev(xspl, (all_knots, coefs, degree))

    for i in range(n_bases):
        out_of_range = (xspl < mi) | (xspl > ma)
        basis[i][out_of_range] = 0

    wave_peak = np.round(wave[np.argmax(basis, axis=1)])

    maxval = np.max(basis, axis=1)
    for i in range(n_bases):
        basis[i][basis[i] < clip*maxval[i]] = 0

    if get_matrix:
        return np.vstack(basis).T

    temp = OrderedDict()
    for i in range(n_bases):
        key = 'bspl {0} {1:.0f}'.format(i, wave_peak[i])
        temp[key] = Template(arrays=(wave*1., basis[i]), name=key, 
                             meta={'wave_peak':wave_peak[i]})
        #temp[key].name = key
        #temp[key].wave_peak = wave_peak[i]

    temp.knots = all_knots
    temp.degree = degree
    temp.xspl = xspl

    return temp


def gaussian_templates(wave, centers=[], widths=[], norm=False):
    """
    Make Gaussian "templates" for the template correction
    """
    _x = np.array([1/np.sqrt(2*np.pi*w**2)**norm*np.exp(-(wave-c)**2/2/w**2) 
                   for c, w in zip(centers, widths)])
    return _x.T


    
    