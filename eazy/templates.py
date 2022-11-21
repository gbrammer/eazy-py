import os
import warnings

from collections import OrderedDict
import numpy as np

import astropy.units as u
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning

from . import utils

__all__ = ["TemplateError", "Template", "Redden", "ModifiedBlackBody", 
           "read_templates_file", "load_phoenix_stars", 
           "bspline_templates", "gaussian_templates"]

class TemplateError(object):
    def __init__(self, file='templates/TEMPLATE_ERROR.eazy_v1.0', arrays=None, filter_wavelengths=[5500.], scale=1.):
        """
        Template error function with spline interpolation at arbitrary redshift.
        
        Parameters
        ----------
        file : str
            File containing the template error function definition 
            (columns of wavelength in Angstroms and the TEF).
        
        arrays : optional, (wave, TEF)
            Set from arrays rather than reading from ``file``.
        
        filter_wavelengths : list
            List of filter pivot wavelengths (observed-frame Angstroms).
        
        scale : float
            Scale factor multiplied to TEF array, e.g., the ``TEMP_ERR_A2`` 
            parameter.
        
        Attributes
        ----------
        te_x, te_y : arrays
            The input wavelength and TEF arrays.
        
        min_wavelength, min_wavelength : float
            Min/max of the wavelengths in ``te_x``.
        
        clip_lo, clip_hi : float
            Extrapolation limits to use if redshifted filters fall outside
            defined ``te_x`` array

        """

        self.file = file
        if arrays is None:
            self.te_x, self.te_y = np.loadtxt(file, unpack=True)
        else:
            self.te_x, self.te_y = arrays
                   
        self.scale = scale
        self.filter_wavelengths = filter_wavelengths
        self._set_limits()
        self._init_spline()


    def _set_limits(self):
        """
        Limits to control extrapolation
        """
        nonzero = self.te_y > 0
        self.min_wavelength = self.te_x[nonzero].min()
        self.max_wavelength = self.te_x[nonzero].max()
        self.clip_lo = self.te_y[nonzero][0]
        self.clip_hi = self.te_y[nonzero][-1]

    def _init_spline(self):
        """
        Initialize the CubicSpline interpolator
        """
        from scipy import interpolate
        self._spline = interpolate.CubicSpline(self.te_x, self.te_y)


    def interpolate(self, filter_wavelength=5500., z=1.):
        """
        ``filter_wavelength`` is observed wavelength of photometric filters.  
        But these sample the *rest* wavelength of the template error function
        at lam/(1+z)
        """
        return self._spline(filter_wavelength/(1+z))*self.scale


    def __call__(self, z, limits=None):
        """
        Interpolate TEF arrays at a specific redshift
        
        Parameters
        ----------
        z : float
            Redshift
        
        limits : None, (float, float)
            Extrapolation limits.  If not specified, get from 
            ``clip_lo`` and ``clip_hi`` attributes.
            
        """
        lcz = np.atleast_1d(self.filter_wavelengths)/(1+z)
        tef_z = self._spline(np.atleast_1d(self.filter_wavelengths)/(1+z))
        
        if limits is None:
            limits = [self.clip_lo, self.clip_hi]
            
        clip_lo = (lcz < self.min_wavelength) 
        tef_z[clip_lo] = limits[0]

        clip_hi = (lcz > self.max_wavelength)
        tef_z[clip_hi] = limits[1]
        
        return tef_z*self.scale 


class Redden(object):
    def __init__(self, model=None, Av=0., **kwargs):
        """
        Wrapper function for `dust_attenuation` and `dust_extinction` 
        reddening laws

        .. plot::
            :include-source:
            
            import numpy as np
            import matplotlib.pyplot as plt
            
            from eazy.templates import Redden
            
            fig, ax = plt.subplots(1,1,figsize=(6,4))
            
            wave = np.arange(1200, 2.e4)
            
            for model in ['calzetti00', 'mw', 'smc', 'reddy15']:
                redfunc = Redden(model=model, Av=1.0)
                ax.plot(wave, redfunc(wave), label=model)
            
            ax.plot(wave, wave*0+10**(-0.4), color='k', 
                      label=r'$A_\lambda = 1$', linestyle=':')
            
            ax.legend()
            ax.loglog()
            
            ax.set_xticks([2000, 5000, 1.e4])
            ax.set_xticklabels([0.2, 0.5, 1.0])
            
            ax.grid()
            ax.set_xlabel('wavelength, microns')
            ax.set_ylabel('Attenuation / extinction (Av=1 mag)')
            
            fig.tight_layout(pad=0.5)
            
        Parameters
        ----------
        model : `extinction`/`attenuation` object or str
            
            Allowable string arguments:
                
                - 'smc': `dust_extinction.averages.G03_SMCBar`
                - 'lmc': `dust_extinction.averages.G03_LMCAvg`
                - 'mw','f99': `dust_extinction.parameter_averages.F99`
                - 'calzetti00', 'c00': `dust_attenuation.averages.C00`
                - 'wg00': `dust_attenuation.radiative_transfer.WG00`
                - 'kc13': Calzetti with modified slope and dust bump from 
                  Kriek & Conroy (2013)
                - 'reddy15': Reddy et al. (2015)
        
        Av : float
            Selective extinction/attenuation (passed as `tau_V` for ``WG00``)
            
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
        """
        E(B-V) for models that have ``Rv``
        """
        if hasattr(self.model, 'Rv'):
            return self.Av/self.model.Rv
        else:
            print('Warning: Rv not defined for model: ' + self.__repr__())
            return 0.


    def __repr__(self):
        msg = '<Redden {0}, Av/tau_V={1}>'
        return msg.format(self.model.__repr__(), self.Av)


    def __call__(self, wave, left=0, right=1., **kwargs):
        """
        Return reddening factor.  
        
        Parameters
        ----------
        wave : array (NW)
            Wavelength array.  If has no units, assume 
            `~astropy.units.Angstrom`.
        
        left, right : float
            Extrapolation at short/long wavelengths
        
        Returns
        -------
        ext : array (NW)
            Extinction / attenuation as a function of wavelength
            
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
            msg = ('Dust model must have either `attenuate` or `extinguish`' +
                  ' method.')
            raise IOError(msg)
        
        if hasattr(wave, '__len__'):
            return ext
        elif ext.size == 1:
            return ext[0]
        else:
            return ext


def read_templates_file(templates_file=None, as_dict=False, **kwargs):
    """
    Read templates listed in ``templates_file``.
            
    Parameters
    ----------
    templates_file : str
        Filename of the ascii file containing the templates list.  Has format
        like
        
        .. code::

            1 templates/fsps_full/tweak_fsps_QSF_12_v3_001.dat 1.0
            2 templates/fsps_full/tweak_fsps_QSF_12_v3_002.dat 1.0
            ...
            N {path} {scale}

        where ``scale`` is the factor needed to scale the template wavelength 
        array to units of Angstroms.
    
    as_dict : bool
        Return dictionary rather than a list (e.g., for `grizli`).
        
    kwargs : dict
        Extra keyword arguments are passed to `~eazy.templates.Template` 
        with ``file`` and ``to_angstrom`` keywords set automatically.
        
    Returns
    -------
    templates : list
        List of `eazy.templates.Template` objects (`dict` if ``as_dict``)
        
    """
    lines = open(templates_file).readlines()
    templates = []
    
    for line in lines:
        if line.strip().startswith('#'):
            continue
        
        lspl = line.split()
        template_file = lspl[1]
        if len(lspl) > 2:
            to_angstrom = float(lspl[2])
        else:
            to_angstrom = 1.
            
        templ = Template(file=template_file, to_angstrom=to_angstrom, 
                         **kwargs)
        
        templates.append(templ)
    
    if as_dict:
        tdict = OrderedDict()
        for t in templates:
            tdict[t.name] = t
        return tdict
    else:
        return templates


class Template():
    def __init__(self, file=None, name=None, arrays=None, sp=None, meta={}, to_angstrom=1., velocity_smooth=0, norm_filter=None, resample_wave=None, fits_column='flux', redfunc=Redden(), redshifts=[0], verbose=True, flux_unit=(u.L_sun/u.Angstrom), **kwargs):
        """
        Template object.
        
        Can optionally specify a 2D flux array with the first
        dimension indicating the template for the nearest redshift in the 
        corresponding ``redshifts`` list.  See When integrating the 
        filter fluxes with ``integrate_filter``, the template index with the 
        redshift nearest to the specified redshift will be used.
        
        Parameters
        ----------
        file : str
            Filename of ascii or FITS template
        
        arrays : (array, array)
            Tuple of ``wave``, ``flux`` arrays.  Here ``flux`` assumed to 
            have units f-lambda.
        
        sp : object
            Object with ``wave``, ``flux`` attributes, e.g., from
            ``prospector``.  Here ``flux`` is assumed to have units of f-nu.

        to_angstrom : float
            Scale factor such that ``wave * to_angstrom`` has units of 
            `astropy.units.Angstrom`
        
        velocity_smooth : float
            Velocity smooothing in km/s, applied if > 0
        
        resample_wave : array
            Grid to resample the template wavelengths read from the input 
        
        fits_column : str
            Column name of the flux column if arrays read from a ``file``
        
        redfunc : `eazy.templates.Redden`
            Object to apply additional reddening.  
        
        redshifts : array-like
            Redshift grid for redshift-dependent templates

        flux_unit : `astropy.units.core.Unit`
            Units of ``flux`` array.
        
        Attributes
        ----------
        wave : array
            wavelength in `astropy.units.Angstrom`, dimensions ``[NWAVE]``.
        
        flux : array
            Flux density f-lambda, can have redshift dependence, dimensions
            ``[NZ, NWAVE]``.
        
        name : str
            Label name
            
        meta : dict
            Metadata
            
        redfunc : `eazy.templates.Redden`, optional
            Object for applying dust reddening.
        
            
        """
        import copy
        from astropy.table import Table
        import astropy.units as u
        
        self.wave = None
        self.flux = None
        self.flux_unit = flux_unit
        
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
            self.wave = np.cast[float](sp.wave)
            self.flux = np.cast[float](sp.flux)
            # already fnu
            self.flux *= utils.CLIGHT*1.e10 / self.wave**2
            
        elif file is not None:
            # Read from a file
            if file.split('.')[-1] in ['fits','csv','ecsv']:
                tab = Table.read(file)
                self.wave = tab['wave'].data.astype(float)
                if fits_column not in tab.colnames:
                    msg = (f"'{fits_column}' not in {file}; " +
                           f"available columns are {tab.colnames}.")
                    raise ValueError(msg)

                self.flux = tab[fits_column].data.astype(float)
                self.orig_table = tab
                
                if hasattr(tab[fits_column], 'unit'):
                    if tab[fits_column].unit is not None:
                        self.flux_unit = tab[fits_column].unit
                        
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
            if arrays[0].shape[0] != np.atleast_2d(arrays[1]).shape[1]:
                raise ValueError("Array dimensions don't match: "+
                                 f'arrays[0]: {arrays[0].shape}, '+
                                 f'arrays[1]: {arrays[1].shape}, ')
                
            if hasattr(self.flux, 'unit'):
                self.flux_unit = self.flux.unit
                
                if hasattr(self.flux, 'value'):
                    self.flux = self.flux.value
                
            #self.set_fnu()
        else:
            raise TypeError('Must specify either `sp`, `file` or `arrays`')
        
        if self.flux.ndim == 1:
            # For redshift dependence
            self.flux = np.atleast_2d(self.flux)
            self.redshifts = np.zeros(1)
            self.NZ, self.NWAVE = self.flux.shape
        else:
            self.NZ, self.NWAVE = self.flux.shape
            if 'NZ' in self.meta:
                redshifts = [self.meta[f'Z{j}'] 
                                      for j in range(self.meta['NZ'])]
                
            if len(redshifts) != self.NZ:
                msg = (f'redshifts ({len(redshifts)})'
                       f' doesn\'t match flux dimension ({self.NZ})!')
                raise ValueError(msg)
            
            self.redshifts = np.array(redshifts)
        
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


    @property
    def shape(self):
        """
        Shape of flux attribute
        """
        return self.flux.shape
    
    
    def flux_flam(self, iz=0, z=None, redshift_type='nearest'):
        """
        Get redshift-dependent template in units of f-lambda
        
        Parameters
        ----------
        iz : int
            Index of template to retrieve
        
        z : float, None
            If specified, get the redshift index with 
            `~eazy.templates.Template.zindex`.
        
        redshift_type : 'nearest', 'interp'
            See `~eazy.templates.Template.zindex`.
        
        Returns
        -------
        flam : array
            Template flux density in units of f-lambda, including any 
            reddening specified in the ``redden`` attribute.
            
        """
        if z is not None:
            if redshift_type == 'interp':
                iz, frac = self.zindex(z=z, redshift_type=redshift_type)
                if frac == 1:
                    flam = self.flux[iz,:]
                else:
                    flam = frac*self.flux[iz,:]
                    flam += (1-frac)*self.flux[iz+1,:]
            else:
                iz = self.zindex(z=z, redshift_type=redshift_type)
                flam = self.flux[iz,:]
        else:
            flam = self.flux[iz,:]

        return flam * self.redden


    def flux_fnu(self, iz=0, z=None, redshift_type='nearest'):
        """
        Get redshift-dependent template in units of f-nu
        
        Parameters
        ----------
        iz : int
            Index of template to retrieve
        
        z : float, None
            If specified, get the redshift index with 
            `~eazy.templates.Template.zindex`.
        
        redshift_type : str
            See `~eazy.templates.Template.zindex`.
        
        Returns
        -------
        fnu : array
            Template flux density in units of f-nu, including any 
            reddening specified in the ``redden`` attribute.
            
        """
        flam = self.flux_flam(iz=iz, z=z, redshift_type=redshift_type)
        return (flam * self.wave**2 / (utils.CLIGHT*1.e10))


    def set_fnu(self):
        """
        Deprecated.  ``flux_fnu`` is now a more cmoplicated function.
        """
        print('Deprecated.  ``flux_fnu`` is now a function.')
        pass


    def smooth_velocity(self, velocity_smooth, in_place=True, raise_error=False):
        """
        Smooth template in velocity using ``astro-prospector``
        
        Parameters
        ----------
        velocity_smooth: float
            Velocity smoothing *sigma*, in km/s.
        
        in_place : bool
            Set internal ``flux`` array to the smoothed array.  If False, then
            return a new `~eazy.templates.Template` object.
        
        raise_error : bool
            If ``from prospect.utils.smoothing import smooth_vel`` fails, 
            raise an exception or die quietly.
            
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
                                       velocity_smooth) 
                             for i in range(self.NZ)])
                             
        sm_flux[~np.isfinite(sm_flux)] = 0.
        
        if in_place:
            self.flux_orig = self.flux*1
            self.velocity_smooth = velocity_smooth
            self.flux = sm_flux
            return True
        else:
            return Template(arrays=(self.wave, sm_flux), 
                            name=self.name, meta=self.meta, 
                            redshifts=self.redshifts)


    def to_observed_frame(self, z=0, scalar=1., extra_sigma=0, lsf_func=None, to_air=False, wavelengths=None, smoothspec_kwargs={'fftsmooth':False}, include_igm=True, clip_wavelengths=[4500,9400]):
        """
        Smooth and resample to observed-frame wavelengths, including an
        optional Line Spread Function (LSF)
                
        Note that the smoothing is performed with 
        `prospect.utils.smoothing.smoothspec <https://prospect.readthedocs.io/en/latest/api/utils_api.html>`_, 
        which doesn't integrate precisely over "pixels" for spectral
        resolutions that are similar to or less than the target smoothing
        factor.
        
        Parameters
        ----------
        z : float
            Target redshift.  Note that only the wavelength array is shifted
            by ``(1+z)``.  The flux densities optionally include IGM
            absorption (and dust from the ``redfunc`` attribute) but don't
            include the ``fl_obs = fl_rest / (1+z)`` scaling.
        
        scalar : float, array
            Scalar value or array with same dimensions as ``wave`` and 
            ``flux`` attributes
            
        extra_sigma : float
            Extra velocity dispersion (sigma, km/s) to add in quadrature with 
            the MUSE LSF
        
        lsf_func : 'Bacon', function
            Line Spread Function (LSF).  If ``'Bacon'``, then use the "UDF-10"
            MUSE LSF from `Bacon et al. 2017 
            <https://ui.adsabs.harvard.edu/abs/2017A%26A...608A...1B>`_ (Eq. 
            8). Can also be a ``function`` that takes an argument of
            wavelength in Angstroms and returns the LSF sigma, in Angstroms.
            If neither of these, then only `extra_sigma` will be applied.
            
        to_air : bool
            Apply vacuum-to-air conversion with `mpdaf.obj.vactoair <https://mpdaf.readthedocs.io/en/latest/api/mpdaf.obj.vactoair.html>`_
        
        wavelengths : array, None
            Optional wavelength grid (observed frame) of the target output 
            (e.g., MUSE) spectrum
        
        smoothspec_kwargs : dict
            Extra keyword arguments to pass to the Prospector smoothing 
            function `prospect.utils.smoothing.smoothspec <https://prospect.readthedocs.io/en/latest/api/utils_api.html>`_.  
            When testing with very high resolution templates around a specific
            wavelength, ``smoothspec_kwargs = {'fftsmooth':True}`` did not
            always work as expected, so be careful with this option (which is
            much faster).
        
        include_igm : bool
            Include IGM absorption at indicated redshift
        
        clip_wavelengths : [float, float]
            Trim the full observed-frame wavelength array before convolving.  
            The defaults bracket the nominal MUSE range.
            
        Returns
        -------
        tobs : `~eazy.template.Template`
            Smoothed and resampled `~eazy.template.Template` object
            
        """
        from astropy.stats import gaussian_sigma_to_fwhm
        from prospect.utils.smoothing import smoothspec
        
        wobs = self.wave*(1+z)
        
        if include_igm:
            igmz = self.igm_absorption(z, pow=include_igm)
        else:
            igmz = 1.
        
        if to_air:
            try:
                from mpdaf.obj import vactoair
                wobs = vactoair(wobs)
            except ImportError:
                msg = ("`to_air` requested but `from mpdaf.obj import " + 
                       "vactoair` failed")
                warnings.warn(msg, AstropyUserWarning)
        
        if clip_wavelengths is not None:
            clip = wobs >= clip_wavelengths[0]
            clip &= wobs <= clip_wavelengths[1]
            
            if clip.sum() == 0:
                raise ValueError('No template wavelengths found in the '+
                                 f'clipping range {clip_wavelengths} '+ 
                                 'Angstroms')
        else:
            clip = wobs > 0
            
        if lsf_func in ['Bacon']:
            # UDF-10 LSF from Bacon et al. 2017
            bacon_lsf_fwhm = lambda w: 5.866e-8 * w**2 - 9.187e-4*w + 6.04
            sig_ang = bacon_lsf_fwhm(wobs[clip]) / gaussian_sigma_to_fwhm
            
            lsf_sigma = sig_ang/wobs[clip]*3.e5
            lsf_func_name = 'MUSE-LSF'
            
        elif hasattr(lsf_func, '__call__'):
            lsf_sigma = lsf_func(wobs[clip])/wobs[clip]*3.e5
            lsf_func_name = 'user'
        
        else:
            lsf_sigma = 0.
            lsf_func_name = None
        
        # Quadrature sum of LSF and extra velocities    
        vel_sigma = np.sqrt(lsf_sigma**2 + extra_sigma**2)
        
        # In Angstroms
        smooth_lambda = vel_sigma / 3.e5 * wobs[clip]
        
        # Do the smoothing
        flux_smooth = smoothspec(wobs[clip], 
                                 (self.flux_flam(z=z)*igmz*scalar)[clip], 
                                 resolution=smooth_lambda, 
                                 smoothtype='lsf', **smoothspec_kwargs)
        
        newname = self.name + f' z={z:.3f}' 
        if lsf_func_name is not None:
            newname += ' + ' + lsf_func_name
        
        if extra_sigma > 0:
            newname += ' + {extra_sigma:.1f} km/s'
            
        tobs = Template(arrays=(wobs[clip], flux_smooth), 
                        name=newname, resample_wave=wavelengths, 
                        redshifts=[z])
        
        return tobs


    def resample(self, new_wave, z=0, in_place=True, return_array=False, interp_func=None):
        """
        Resample the template to a new wavelength grid
        
        Parameters
        ----------
        new_wave : array
            New wavelength array, can have units.
        
        z : float
            Redshift internal wavelength before resampling.  
            (z=0 yields no shift).
        
        in_place : bool
            Set internal ``wave`` and ``flux`` arrays to the resampled
            values
        
        return_array : bool
            Return the resampled ``flux`` array if true, else return a new
            `~eazy.templates.Template` object.
        
        interp_func : None
            Interpolation function.  If nothing specified, tries to use
            `grizli.utils_c.interp.interp_conserve_c` and falls back to 
            `eazy.utils.interp_conserve`.  
        
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
        
        if interp_func is None:
            try:
                from grizli.utils_c import interp
                interp_func = interp.interp_conserve_c
            except:
                interp_func = utils.interp_conserve
                
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
                                redshifts=self.redshifts)


    def zindex(self, z=0., redshift_type='nearest'):
        """
        Get the redshift index of a multi-dimensional template array
        
        Parameters
        ----------
        z : float
            Redshift to retrieve
        
        redshift_type : 'nearest', 'interp', 'floor'
            Interpolation type:
            
            - 'nearest': nearest step in the template redshift grid
            - 'interp': Returns index below ``z`` and interpolation fraction
            - 'floor': last index where redshift grid < ``z``.
        
        Returns
        -------
        iz : int
            Array index, i.e, ``self.flux[iz,:]``
        
        frac : float, optional
            Fraction for interpolation, if ``redshift_type == 'interp'``.
        
        """
        
        zint = np.interp(z, self.redshifts, np.arange(self.NZ),
                         left=0, right=self.NZ-1)
                         
        if redshift_type == 'nearest':
            iz = np.round(zint).astype(int)
            
        elif redshift_type == 'interp':
            iz = int(np.floor(zint))
            if z < self.redshifts[0]:
                frac = 1.
                
            elif iz < self.NZ-1:
                frac = 1. - ( (z - self.redshifts[iz]) / 
                              np.diff(self.redshifts)[iz] )
            else:
                frac = 1.
            
            return iz, frac
        elif redshift_type == 'floor':
            iz = int(np.floor(zint))
        else:
            raise ValueError(f"redshift_type ({redshift_type}) must be " + 
                             "'nearest', 'interp', or 'floor'.")
        return iz


    def zscale(self, z, scalar=1, include_igm=True, **kwargs):
        """Redshift the template and multiply by a scalar.

        Parameters
        ----------
        z : float
            Redshift to use.

        scalar : float or array
            Multiplicative factor.  Additional factor of 1./(1+z) is implicit.
        
        include_igm : bool
            Include Inoue (2014) IGM absorption (also can be passed as 
            ``apply_igm`` in ``kwargs``.)
        
        Returns
        -------
        ztemp : `~eazy.templates.Template`
            Redshifted and scaled spectrum.

        """
        if 'apply_igm' in kwargs:
            include_igm = kwargs['apply_igm']
            
        if include_igm:
            igmz = self.igm_absorption(z, pow=include_igm)
        else:
            igmz = 1.
        
        return Template(arrays=(self.wave*(1+z),
                                self.flux_flam(z=z)*scalar/(1+z)*igmz), 
                        name=f'{self.name} z={z}')


    def integrate_filter(self, filt, flam=False, scale=1., z=0, include_igm=False, redshift_type='nearest', iz=None):
        """
        Integrate the template through a `FilterDefinition` filter object.
        
        .. note:: The `grizli` interpolation function 
                  `grizli.utils_c.interp.interp_conserve_c` will be used if 
                  available.
        
        Parameters
        ----------
        filt : `~eazy.filters.FilterDefinition` object or a list of them
            Filter(s) to interpolate
        
        flam : bool
            Return integrated fluxes in f-lambda, rather than f-nu
        
        scale : float, array
            Scale factor applied to template before integrating.  If an 
            array is specified, it must have the same size as the template
            ``wave`` array.
        
        z : float
            Redshift the template before integrating through the filter
        
        include_igm : bool
            Include IGM absorption
        
        redshift_type : str
            See `~eazy.templates.Template.zindex`.
        
        iz : int
            Evaluate for a specific index of the ``flux`` array rather than
            calculating with ``zindex``
            
        Returns
        -------
        fnu : float or array
            Template integrated through one or more filters from ``filt``.  By 
            defaults has units of fnu
            
            .. note:: The interpolated fluxes *do not* include factors of 
                      (1+z) from the redshifted templates.
            
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
            fnu = self.flux_fnu(z=z, redshift_type=redshift_type)
        else:
            fnu = self.flux_fnu(iz=iz)
                        
        fnu *= scale*igmz
        
        fluxes = []
        for filt_i in filts:    
            templ_filt = interp(filt_i.wave.astype(float),
                                self.wave.astype(float)*(1+z),
                                fnu.astype(float), left=0, right=0)
                
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
        Compute IGM absorption with `eazy.igm.Inoue14`.
        
        Parameters
        ----------
        z : float
            Redshift to use for IGM absorption factors
        
        scale_tau : float
            Scale factor multiplied to of IGM ``tau`` values
        
        pow : float 
            Scale the absorption strength as ``eazy.igm.Inoue14()**pow``.
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
        """
        Return template as an `astropy.table.Table`.
        
        Parameters
        ----------
        formats : dict
            Set ``format`` attributes of table columns
        
        with_units : bool
            Set ``unit`` attributes of table columns
        
        flatten : bool
            If no redshift dependence (``NZ==0``), columns are 1D arrays.
        
        Returns
        -------
        tab : `astropy.table.Table`
            Output table
            
        """
        from astropy.table import Table
        import astropy.units as u
        import copy
        
        tab = Table()
        tab['wave'] = self.wave
        tab['flux'] = self.flux.T
        
        if with_units:
            tab['wave'].unit = u.Angstrom
            tab['flux'].unit = self.flux_unit

        for c in tab.colnames:
            if c in formats:
                tab[c].format = formats[c]
                
        tab.meta = copy.deepcopy(self.meta)
        if self.NZ > 1:
            tab.meta['NZ'] = self.NZ
            for j in range(self.NZ):
                tab.meta[f'Z{j}'] = self.redshifts[j]
        else:
            if flatten:
                tab['flux'] = self.flux[0,:]
                               
        return tab


class ModifiedBlackBody(object):
    def __init__(self, Td=47, beta=1.6, q=2.34, alpha=-0.75):
        """
        Modified black body: nu**beta * BB(nu, Td) + FIR-radio correlation
        
        Parameters
        ----------
        Td : float
            Dust temperature
        
        beta : float
            Slope parameter
        
        q : float
            FIR-radio normalization
        
        alpha : float
            Radio spectral slope: ``fnu = (nu/1.4e9)**alpha``
            
        """
        self.Td = Td
        self.q = q
        self.beta = beta
        self.alpha = alpha
    
    @property 
    def bb(self):
        """
        `astropy.modeling.models.BlackBody` function with ``self.Td`` dust temperature
        """
        from astropy.modeling.models import BlackBody
        return BlackBody(temperature=self.Td*u.K)
        
    def __call__(self, wave, q=None):
        """
        Return modified BlackBody (fnu) as a function of wavelength
        
        Parameters
        ----------
        wave : array
            Wavelength array.  If no ``unit`` attribute, assume 
            `astropy.units.micron`.
        
        q : float
            Parameter of FIR-radio correlation.  If not specified, use 
            internal ``self.q``.
            
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

def load_phoenix_stars(logg_list=PHOENIX_LOGG, teff_list=PHOENIX_TEFF, zmet_list=PHOENIX_ZMET, add_carbon_star=True, file='bt-settl_t400-7000_g4.5.fits', sonora_dwarfs=True):
    """
    Load Phoenix stellar templates
    
    `file` is available at
    https://s3.amazonaws.com/grizli/CONF/bt-settl_t400-7000_g4.5.fits
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
        #url = 'https://s3.amazonaws.com/grizli/CONF'
        url = 'https://erda.ku.dk/vgrid/Gabriel%20Brammer/CONF/'
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
    
    if sonora_dwarfs:
        tstars += load_sonora_stars()
        
    return tstars


def load_sonora_stars():
    """
    Load Sonora brown dwarf models
    """
    import glob
    
    paths = ['/tmp', './templates/', './']
    found = False
    
    for path in paths:
        templ_path = os.path.join(path, 'sonora')
        if os.path.exists(templ_path):
            print(f'sonora_stars: {templ_path}')
            found = True
            break
    
    if not found:
        return []
    
    files = glob.glob(os.path.join(templ_path, 'sonora*fits'))
    files.sort()
    
    stars = []
    for file in files:
        name = ' ' + os.path.basename(file).split('.fits')[0]
        stars.append(Template(file=file, name=name))

    return stars


def param_table(templates):
    """
    Try to generate parameters for a list of templates from their 
    metadata
    
    (TBD)
    """
    pass


def bspline_templates(wave, degree=3, df=6, get_matrix=True, log=False, clip=1.e-4, minmax=None):
    """
    B-spline basis functions, modeled after `patsy.splines <https://patsy.readthedocs.io/en/latest/spline-regression.html>`_
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


    
    