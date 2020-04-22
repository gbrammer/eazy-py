"""
Tools for making FSPS templates
"""
import os
from collections import OrderedDict

import numpy as np
import astropy.units as u
try:
    from fsps import StellarPopulation
except:
    # Broken, but imports
    StellarPopulation = object

from . import utils
from . import templates

DEFAULT_LABEL = 'fsps_tau{tau:3.1f}_logz{logzsol:4.2f}_lage{log_age:4.2f}_av{Av:4.2f}'

WG00_DEFAULTS = dict(geometry='shell', dust_type='mw', 
                   dust_distribution='homogeneous')

def agn_templates():
    
    import fsps
    import eazy
    
    sp = fsps.StellarPopulation()
    res = eazy.filters.FilterFile(path=None)  
    jfilt = res[161]
    
    sp.params['dust_type'] = 2
    sp.params['dust2'] = 0
    
    sp.params['agn_tau'] = 10
    sp.params['fagn'] = 0
    wave, f0 = sp.get_spectrum(tage=0.5, peraa=True)

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    
    sp.params['fagn'] = 1.    
    for tau in [5,10,20,50,80,120]:
        sp.params['agn_tau'] = tau
        wave, f1 = sp.get_spectrum(tage=0.5, peraa=True)
        
        label = 'agn_tau{0:03d}'.format(sp.params['agn_tau'])
        ax.plot(wave/1.e4, f1-f0, label=label, alpha=0.5)
        
        #fnorm = np.trapz(f1-f0, wave)
        templ = templates.Template(arrays=(wave, f1-f0), name=label, meta={'agn_tau':sp.params['agn_tau']})
        fnorm = templ.integrate_filter(jfilt, flam=True)/jfilt.pivot
        templ.flux *= fnorm
        
        file = 'templates/agn_tau/{0}.fits'.format(templ.name)
        print(tau, file)
        
        tab = templ.to_table() 
        tab.write(file, overwrite=True)
    
    ax.legend()
    ax.loglog()
    ax.grid()
    ax.set_ylim(1.e-7, 1.e-3)
    ax.set_xlim(90/1.e4, 200)
    
    ax.set_xlabel(r'$\lambda$, $\mu$m')
    ax.set_ylabel(r'$f_\lambda$')
    
    fig.tight_layout(pad=0.1)
    fig.savefig('templates/agn_tau/fsps_agn.png')
    
def extreme_starburst():
    
    #self = sps.ExtendedFsps(zmet=10, zcontinuous=False, add_neb_emission=True, sfh=4, tau=0.2)
    self = sps.ExtendedFsps(logzsol=0, zcontinuous=True, add_neb_emission=True, sfh=4, tau=0.2)

    self._set_extend_attrs(line_sigma=100, lya_sigma=300)
    #self._set_extend_attrs(line_sigma=50, lya_sigma=200)
    #self._set_extend_attrs(line_sigma=400, lya_sigma=400)
    
    self.set_fir_template()
    self.set_dust()
    
    self.params['tau'] = 0.2
    
    path = 'templates/fsps_full_2019/'
    par = grizli.utils.read_catalog(path+'xfsps_QSF_12_v3.param.fits')
    files = glob.glob(path+'xfsps_QSF_12_v3_0??.dat')
    files.sort()
    par['file'] = files
    
    Av = 5
    templ = self.get_full_spectrum(tage=0.3, Av=Av, set_all_templates=True)
    
    res = eazy.filters.FilterFile(os.getenv('EAZYCODE') + 'filters/FILTER.RES.latest') 
    
    vfilt = res.filters[154]
    jfilt = res.filters[160]
    Lv = templ.integrate_filter(vfilt, flam=False)*3.e18/vfilt.pivot
    jflux = templ.integrate_filter(jfilt, flam=True)
    
    # Replace 6 template in param.fits
    tab = templ.to_table()
    tab['flux'] /= jflux*jfilt.pivot
    plt.plot(tab['wave'], tab['flux'], alpha=0.5)

    if False:
        plt.plot(self.templ_unred.wave, self.templ_unred.flux/(jflux*jfilt.pivot), alpha=0.5)
        plt.plot(self.templ_orig.wave, self.templ_orig.flux/(jflux*jfilt.pivot), alpha=0.5)
    
    # Replace the dusty-old template with this starburst
    file = 'templates/fsps_full_2019/{0}.fits'.format(templ.name)
    tab.write(file, overwrite=True)
    
    par['file'][5] = os.path.basename(file)
    par['mass'][5] = tab.meta['stellar_mass']
    par['energy_abs'][5] = tab.meta['energy_absorbed']
    par['Lv'][5] = Lv
    par['Av'][5] = tab.meta['Av']
    par['sfr'][5] = tab.meta['sfr']
    #par['sfr100'][5] = tab.meta['sfr100']
    par['LIR'][5] = tab.meta['energy_absorbed']
        
    par.write(path+'xfsps_QSF_12_v3.SB.param.fits', overwrite=True)
    
    # Blue template without Lyman alpha
    
    ex = sps.ExtendedFsps(logzsol=0, zcontinuous=True, add_neb_emission=True, sfh=4, tau=0.2)
    ex.set_fir_template()
    
    ex.params['logzsol'] = -1
    ex.params['gas_logz'] = -1
    ex.params['gas_logu'] = -2
    ex.params['tau'] = 0.3
    tage, Av = 0.2, 0.2

    templ = ex.get_full_spectrum(tage=tage, Av=Av, scale_lyman_series=0., set_all_templates=False)
    
    vfilt = res.filters[154]
    jfilt = res.filters[160]
    Lv = templ.integrate_filter(vfilt, flam=False)*3.e18/vfilt.pivot
    jflux = templ.integrate_filter(jfilt, flam=True)
    
    # Replace 6 template in param.fits
    tab = templ.to_table()
    tab['flux'] /= jflux*jfilt.pivot
    plt.plot(tab['wave'], tab['flux'])
    plt.loglog()
    
    # Replace the blue template with this one
    file = 'templates/fsps_full_2019/{0}.fits'.format(templ.name)
    tab.write(file, overwrite=True)
    
    path = 'templates/fsps_full_2019/'
    par = grizli.utils.read_catalog(path+'xfsps_QSF_12_v3.SB.param.fits')
    
    ix = 6
    par['file'][ix] = os.path.basename(file)
    par['mass'][ix] = tab.meta['stellar_mass']
    par['energy_abs'][ix] = tab.meta['energy_absorbed']
    par['Lv'][ix] = Lv
    par['Av'][ix] = tab.meta['Av']
    par['sfr'][ix] = tab.meta['sfr']
    par['LIR'][ix] = tab.meta['energy_absorbed']
        
    par.write(path+'xfsps_QSF_12_v3.SB.param.fits', overwrite=True)
    
    # Remove Ly-alpha 
    for file in par['file']:
        if file.endswith('dat'):
            file = file.replace('v3_', 'v3_gm09_')
            print(file)
            templ = grizli.utils.read_catalog(file)
            lya = np.abs(templ['wave'] - 1216) < 70
            interp = np.interp(templ['wave'][lya], templ['wave'][~lya], templ['flux'][~lya])
            templ['flux'][lya] = interp
            templ.write(file.replace('v3_', 'v3_nolya_'), format='ascii.commented_header', overwrite=True)
            
def fit_dust_wg00():
    """
    Fit WG00 as flexible model
    """
    from dust_attenuation import averages, shapes, radiative_transfer
    reload(averages); reload(shapes)
    reload(averages); reload(shapes)
    
    shapes.x_range_N09 = [0.01, 1000] 
    averages.x_range_C00 = [0.01, 1000]
    averages.x_range_L02 = [0.01, 0.18]

    from astropy.modeling.fitting import LevMarLSQFitter
    tau_V = 1.0
    tau_V_grid = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0,
                           3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0,
                           9.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
                           40.0, 45.0, 50.0])
    
    tau_V_grid = tau_V_grid[tau_V_grid < 11]
    
    i=-1
    
    params = []
    for i in range(len(tau_V_grid)):
        tau_V = tau_V_grid[i]                      
        wg00 = radiative_transfer.WG00(tau_V=tau_V, **WG00_DEFAULTS)
    
        model = shapes.N09()
        model.fixed['x0'] = True
    
        fitter = LevMarLSQFitter()
    
        wave = np.logspace(np.log10(0.18), np.log10(3), 100)
        y = wg00(wave*u.micron)
        _fit = fitter(model, wave, y)
    
        plt.plot(wave, y, label='tau_V = {0:.2f}'.format(tau_V), color='k', alpha=0.4)
        
        #plt.plot(wave, _fit(wave))

        shapes.x_range_N09 = [0.01, 1000] 
        averages.x_range_C00 = [0.01, 1000]
    
        wfull = np.logspace(-1.5, np.log10(20.1), 10000)
        plt.plot(wfull, _fit(wfull), color='r', alpha=0.4)
        params.append(_fit.parameters)
    
    params = np.array(params)
    N = params.shape[1]
    vx = np.linspace(0, 10, 100)
    order = [3, 5, 5, 5, 5]
    
    coeffs = {}
    
    for i in range(N):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(tau_V_grid, params[:,i], marker='o', label=_fit.param_names[i])
        
        c = np.polyfit(tau_V_grid, params[:,i], order[i])
        ax.plot(vx, np.polyval(c, vx))
        ax.legend()
        ax.grid()
        coeffs[_fit.param_names[i]] = c

try:
    from dust_attenuation.baseclasses import BaseAttAvModel
except:
    BaseAtttauVModel = object
    
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
    include_bump = False
    
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
        
        if not hasattr(self, 'N09'):
            self._init_N09()
            
        tau_V = self.get_tau(Av)
        
        #Av = np.polyval(self.coeffs['Av'], tau_V)
        x0 = np.polyval(self.coeffs['x0'], tau_V)
        gamma = np.polyval(self.coeffs['gamma'], tau_V)
        if self.include_bump:
            ampl = np.polyval(self.coeffs['ampl'], tau_V)
        else:
            ampl = 0.
            
        slope = np.polyval(self.coeffs['slope'], tau_V)
        
        return self.N09.evaluate(x, Av, x0, gamma, ampl, slope)
        
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
    
class ExtendedFsps(StellarPopulation):
    """
    Extended functionality for the `~fsps.StellarPopulation` object
    """
    
    def _set_extend_attrs(self, line_sigma=50, lya_sigma=200):
        """
        Set attributes on `~fsps.StellarPopulation` object used by `narrow_lines`.

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
        
        # Precomputed arrays for WG00 reddening defined between 0.1..3 um
        self.wg00lim = (self.wavelengths > 1000) & (self.wavelengths < 3.e4)
        self.wg00red = (self.wavelengths > 1000)*1.
        
        self.exec_params = None
        self.narrow = None
        
    def narrow_emission_lines(self, tage=0.1, emwave=DEFAULT_LINES, line_sigma=100, oversample=5, clip_sigma=10, verbose=False, get_eqw=True, scale_lyman_series=1., force_recompute=False, **kwargs):
        """
        Replace broad FSPS lines with specified line widths
    
        tage : age in Gyr of FSPS model
        FSPS sigma: line width in A in FSPS models
        emwave : (approx) wavelength of line to replace
        line_sigma : line width in km/s of new line
        oversample : factor by which to sample the Gaussian profiles
        clip_sigma : sigmas from line center to use for the line
        scale_lyman_series : scaling to apply to Lyman-series emission lines
        
        Returns: `dict` with keys
            wave_full, flux_full, line_full = wave and flux with fine lines
            wave, flux_line, flux_clean = original model + removed lines
            ymin, ymax = range of new line useful for plotting
        
        """
        if not hasattr(self, 'emline_dlam'):
            self._set_extend_attrs(line_sigma=line_sigma, **kwargs)
        
        self.params['add_neb_emission'] = True
        
        # Avoid recomputing if all parameters are the same (i.e., change Av)
        call_params = np.hstack([self.param_floats, emwave, 
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
    
        fsps_sigma = [2*self.emline_dlam[ix] for ix in line_ix]
        line_sigma = [self.emline_sigma[ix] for ix in line_ix]
        line_dlam = [sig/3.e5*lwave 
                     for sig, lwave in zip(line_sigma, line_wave)]
    
        clean = line*1
        wlimits = [np.min(emwave), np.max(emwave)]
        wlimits = [2./3*wlimits[0], 4.3*wlimits[1]]
    
        wfine = utils.log_zgrid(wlimits, np.min(line_sigma)/oversample/3.e5)
        qfine = wfine < 0
    
        if verbose:
            msg = 'Matched line: {0} [{1}], lum={2}'
            for i, ix in enumerate(line_ix):
                print(msg.format(line_wave[i], ix, line_lum[i]))
    
        ######### 
        # Remove lines from FSPS
        # line width seems to be 2*dlam at the line wavelength
        for i, ix in enumerate(line_ix):
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
                    
            norm = line_lum[i]/np.sqrt(2*np.pi*line_dlam[i]**2)
            gline = norm*np.exp(-(wfull - line_wave[i])**2/2/line_dlam[i]**2)
            if self.emline_names[line_ix[i]].startswith('Ly'):
                gline *= scale_lyman_series

            gfull += gline
            
            if get_eqw:
                clip = np.abs(wfull - line_wave[i]) < clip_sigma*line_dlam[i]
                eqw = np.trapz(gline[clip]/cfull[clip], wfull[clip])
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
            
    def set_fir_template(self, arrays=None, file='templates/magdis/magdis_09.txt'):
        """
        Set the far-IR template for reprocessed dust emission
        """
        if os.path.exists(file):
            _ = np.loadtxt(file, unpack=True)
            wave, flux = _[0], _[1]
        else:
            wave, flux = arrays
            
        fir_flux = np.interp(self.wavelengths, wave, flux, left=0, right=0)
        self.fir_template = fir_flux/np.trapz(fir_flux, self.wavelengths)
        self.fir_filename = file
        self.fir_arrays = arrays
        
    def set_dust(self, Av=0., dust_obj_type='WG00x', wg00_kwargs=WG00_DEFAULTS):
        """
        Set `dust_obj` attribute
        
        dust_obj_type: 
        
            'WG00' = `~dust_attenuation.radiative_transfer.WG00`
            'C00' = `~dust_attenuation.averages.C00`
            'WG00x' = `ParameterizedWG00`
            
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
            else:
                self.dust_obj= averages.C00(Av=Av)
            
            print('Init dust_obj: {0} {1}'.format(dust_obj_type, self.dust_obj.param_names))
            
        self.Av = Av
        
        if dust_obj_type == 'WG00':
            Avs = np.array([0.151, 0.298, 0.44 , 0.574, 0.825, 1.05 , 1.252, 1.428, 1.584, 1.726, 1.853, 1.961, 2.065, 2.154, 2.318, 2.454, 2.573, 2.686, 3.11 , 3.447, 3.758, 4.049, 4.317, 4.59 , 4.868, 5.148])
            taus = np.array([ 0.25,  0.5 ,  0.75,  1.  ,  1.5 ,  2.  ,  2.5 ,  3.  ,  3.5 , 4.  ,  4.5 ,  5.  ,  5.5 ,  6.  ,  7.  ,  8.  ,  9.  , 10.  , 15.  , 20.  , 25.  , 30.  , 35.  , 40.  , 45.  , 50.  ])
            tau_V = np.interp(Av, Avs, taus, left=0.25, right=50)
            self.dust_obj.tau_V = tau_V
            self.Av = self.dust_obj(5500*u.Angstrom)
        else:
            self.dust_obj.Av = Av
    
    def get_full_spectrum(self, tage=0.1, Av=0., get_template=True, set_all_templates=False, **kwargs):
        """
        Get full spectrum with reprocessed emission lines and dust emission
        
        dust_fraction: Fraction of the SED that sees the specified Av
        
        """
        if hasattr(self, 'dust_obj'):
            self.set_dust(Av=Av, dust_obj_type=self.dust_obj_type)
        else:
            self.set_dust(Av=Av, dust_obj_type='WG00x')
        
        for k in kwargs:
            if k in self.params.all_params:
                self.params[k] = kwargs[k]
        
        self.tage = tage
        
        _ = self.narrow_emission_lines(tage=tage, **kwargs)

        wave = _['wave_full']
        flux = _['flux_full']
        lines = _['line_full']

        #self.sfr100 = self.sfr_avg(dt=np.minimum(tage, 0.1))
        
        # Apply dust
        if self.dust_obj_type == 'WG00':
            
            # To template
            red = (wave > 1000)*1.
            wlim = (wave > 1000) & (wave < 3.e4)
            red[wlim] = 10**(-0.4*self.dust_obj(wave[wlim]*u.Angstrom))
            
            # To linees
            red_lines = (self.emline_wavelengths > 1000)*1.
            wlim = (self.emline_wavelengths > 1000) 
            wlim &= (self.emline_wavelengths < 3.e4)
            Alam  = self.dust_obj(self.emline_wavelengths[wlim]*u.Angstrom)
            red_lines[wlim] = 10**(-0.4*Alam)
            
        else:
            red = 10**(-0.4*self.dust_obj(wave*u.Angstrom))
            Alam = self.dust_obj(self.emline_wavelengths*u.Angstrom)
            red_lines = 10**(-0.4*Alam)
        
        # Apply dust to lines
        lred = [llum*lr for llum, lr in 
                        zip(self.emline_luminosity, red_lines)]
        self.emline_reddened = np.array(lred)
        
        # Total energy
        e0 = np.trapz(flux, wave)
        # Energy of reddened template
        
        e1 = np.trapz(flux*red, wave)
        self.energy_absorbed = (e0 - e1)
                
        # Add dust emission
        if hasattr(self, 'fir_template') & self.params['add_dust_emission']:
            dust_em = np.interp(wave, self.wavelengths, self.fir_template)
            dust_em *= self.energy_absorbed
        else:
            dust_em = 0.
        
        meta0 = self.meta
        self.templ = self.as_template(wave, flux*red+dust_em, meta=meta0)
        
        # Set template attributes
        if set_all_templates:
            
            # Original wavelength grid
            owave = self.wavelengths
            oAlam = self.dust_obj(owave[self.wg00lim]*u.Angstrom)
            self.wg00red[self.wg00lim] = 10**(-0.4*oAlam)
            ofir = self.fir_template*self.energy_absorbed
            fl_orig = _['flux_line']*self.wg00red + ofir
            self.templ_orig = self.as_template(owave, fl_orig, meta=meta0)
            
            # No lines
            meta = meta0.copy()
            meta['add_neb_emission'] = False
            fl_cont = (flux - lines)*red+dust_em
            ocont = _['flux_clean']*self.wg00red + ofir
            self.templ_cont = self.as_template(wave, fl_cont, meta=meta)
            self.templ_cont_orig = self.as_template(owave, ocont, meta=meta)
            
            # No dust
            meta = meta0.copy()
            meta['add_neb_emission'] = True
            meta['Av'] = 0
            self.templ_unred = self.as_template(wave, flux, meta=meta)
            self.templ_unred_orig = self.as_template(owave, _['flux_clean'],
                                                     meta=meta)
            
        if get_template:
            return self.templ
        else:
            return self.templ.wave, self.templ.flux
            
    def as_template(self, wave, flux, label=DEFAULT_LABEL, meta=None):
        """
        Return a `~templates.Template` object with metadata
        """
        if meta is None:
            meta = self.meta
            
        templ = templates.Template(arrays=(wave, flux), meta=meta,
                                   name=label.format(**meta))
        return templ
    
    @property 
    def sfr100(self):
        """
        SFR averaged over maximum(tage, 100 Myr) from `sfr_avg`
        """
        return self.sfr_avg(dt=np.minimum(self.params['tage'], 0.1))
               
    @property 
    def meta(self):
        """
        Full metadata, including line properties
        """
        meta = self.param_dict
        
        if self._zcontinuous:
            meta['metallicity'] = 10**self.params['logzsol']*0.019
        else:
            meta['metallicity'] = self.zlegend[self.params['zmet']]
            
        for k in ['log_age','stellar_mass', 'formed_mass', 'log_lbol', 
                  'sfr', 'sfr100', 'dust_obj_type','Av','energy_absorbed', 
                  'fir_filename', '_zcontinuous']:
            if hasattr(self, k):
                meta[k] = self.__getattribute__(k)
        
        if hasattr(self, 'emline_names'):
            has_red = hasattr(self, 'emline_reddened')
            
            for i in range(len(self.emline_wavelengths)):
                n = self.emline_names[i]
                meta['line {0}'.format(n)] = self.emline_luminosity[i]
                if has_red:
                    meta['rline {0}'.format(n)] = self.emline_reddened[i]
                    
                meta['eqw {0}'.format(n)] = self.emline_eqw[i]
                
        return meta
        
    @property 
    def param_dict(self):
        d = OrderedDict()
        for p in self.params.all_params:
            d[p] = self.params[p]
        
        return d
        
    @property 
    def param_floats(self):
        d = []
        for p in self.params.all_params:
            d.append(self.params[p]*1)
        
        return np.array(d)