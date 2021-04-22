import os
import time
import numpy as np

from collections import OrderedDict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except:
    HAS_TQDM = False
    
try:
    from grizli.utils import GTable as Table
except:
    from astropy.table import Table

import astropy.io.fits as pyfits
import astropy.units as u
import astropy.constants as const

from . import filters
from . import param 
from . import igm as igm_module

from . import templates as templates_module 
from .templates import gaussian_templates, bspline_templates

from . import utils 

IGM_OBJECT = igm_module.Inoue14()

TRUE_VALUES = [True, 'y', 'yes', 'Y', 'Yes']

__all__ = ["PhotoZ", "TemplateGrid"]

DEFAULT_UBVJ_FILTERS = [153,154,155,161] # Maiz-Appellaniz & 2MASS

DEFAULT_RF_FILTERS = [270, 274] # UV tophat
DEFAULT_RF_FILTERS += [120, 121] # GALEX
DEFAULT_RF_FILTERS += [156, 157, 158, 159, 160] #SDSS
DEFAULT_RF_FILTERS += [161, 162, 163] # 2MASS

MIN_VALID_FILTERS = 1

NUVRK_FILTERS = [121, 158, 163]

CDF_SIGMAS = np.linspace(-5, 5, 51)

class PhotoZ(object):
    ZML_WITH_PRIOR = None
    ZML_WITH_BETA_PRIOR = None
    ZPHOT_AT_ZSPEC = None
    ZPHOT_USER = None
    
    def __init__(self, param_file='zphot.param', translate_file='zphot.translate', zeropoint_file=None, load_prior=True, load_products=True, params={}, random_seed=0, n_proc=0, cosmology=None, compute_tef_lnp=True, tempfilt=None, **kwargs):
        """
        Main object for fitting templates / photometric redshifts
        """
        from astropy.cosmology import LambdaCDM
        global IGM_OBJECT
        
        self.param_file = param_file
        self.translate_file = translate_file
        self.zeropoint_file = zeropoint_file
        
        self.random_seed = random_seed
        
        # param_file='zphot.param.m0416.uvista'; translate_file='zphot.translate.m0416.uvista'; zeropoint_file='zphot.zeropoint.m0416.uvista'
        # 
        # param_file='zphot.param.goodss.uvista'; translate_file='zphot.translate.goodss.uvista'; zeropoint_file='zphot.zeropoint.goodss.uvista'
        
        # if False:
            # from eazypy.filters import FilterFile
            # from eazypy.param import EazyParam, TranslateFile
            # from eazypy.igm import Inoue14
            # from eazypy.templates import TemplateError
            # from eazypy.photoz import TemplateGrid
            
        ### Read parameters
        self.param = param.EazyParam(param_file, read_templates=False, 
                                     read_filters=False)
        self.translate = param.TranslateFile(translate_file)
        
        if 'IGM_SCALE_TAU' in self.param.params:
            IGM_OBJECT.scale_tau = self.param['IGM_SCALE_TAU']
        
        if 'ARRAY_NBITS' in self.param.params:
            if self.param['ARRAY_NBITS'] == 64:
                self.ARRAY_DTYPE = np.float64
            else:
                self.ARRAY_DTYPE = np.float32
        else:
            self.ARRAY_DTYPE = np.float32
                    
        if 'MW_EBV' not in self.param.params:
            self.param.params['MW_EBV'] = 0.0354 # MACS0416
            #self.param.params['MW_EBV'] = 0.0072 # GOODS-S
            #self.param['SCALE_2175_BUMP'] = 0.4 # Test
        
        for key in params:
            self.param.params[key] = params[key]
            
        ### Read templates
        kws = dict(templates_file=self.param['TEMPLATES_FILE'], 
                    velocity_smooth=self.param['TEMPLATE_SMOOTH'], 
                    resample_wave=self.param['RESAMPLE_WAVE'])
                          
        self.templates = self.param.read_templates(**kws)
        
        ### Set redshift fit grid
        self.set_zgrid()
        
        ### Set cosmology
        if cosmology is None:
            # Simple 
            self.cosmology = LambdaCDM(H0=self.param['H0'], 
                                       Om0=self.param['OMEGA_M'], 
                                       Ode0=self.param['OMEGA_L'], 
                                       Tcmb0=2.725, Ob0=0.048)
        else:
            self.cosmology = cosmology
            
        ### Read catalog and filters
        self.read_catalog()
        if self.NFILT < 1:
            print('\n!! No filters found, maybe a problem with'
                  ' the translate file?\n')
            return None
            
        self.idx = np.arange(self.NOBJ, dtype=int)
        
        ### Read prior file
        self.full_logprior = np.zeros((self.NOBJ, self.NZ), 
                                  dtype=self.ARRAY_DTYPE)
        if load_prior:
            self.read_prior()
        
        self.lnp = np.zeros_like(self.full_logprior)
        self.chi2_fit = np.zeros_like(self.lnp)
        
        self.lnp_beta = self.lnp*0.
        
        if zeropoint_file is not None:
            self.read_zeropoint(zeropoint_file)
        else:
            self.zp = self.f_numbers*0+1.
            
        self.lnpmax = np.zeros(self.NOBJ, dtype=self.ARRAY_DTYPE)
        self.zml = None
        self.zbest = np.zeros_like(self.lnpmax)
        self.chi2_best = np.zeros_like(self.zbest)-1
                
        self.coeffs_best = np.zeros((self.NOBJ, self.NTEMP), 
                                    dtype=self.ARRAY_DTYPE)
        
        self.fit_coeffs = np.zeros((self.NOBJ, self.NZ, self.NTEMP),
                                   dtype=self.ARRAY_DTYPE)
        
        self.coeffs_draws = np.zeros((self.NOBJ, 100, self.NTEMP), 
                                     dtype=self.ARRAY_DTYPE)
        self.get_err = False
        
        ### Interpolate templates        
        if tempfilt is None:
            msg = 'Template grid: {0} (this may take some time)'
            print(msg.format(self.param['TEMPLATES_FILE']))
        
            t0 = time.time()
            self.tempfilt = TemplateGrid(self.zgrid, self.templates, 
                                        RES=self.param['FILTERS_RES'], 
                                        f_numbers=self.f_numbers, 
                                        add_igm=self.param['IGM_SCALE_TAU'], 
                                    galactic_ebv=self.param.params['MW_EBV'], 
                                    Eb=self.param['SCALE_2175_BUMP'], 
                                    n_proc=n_proc, cosmology=self.cosmology, 
                                    array_dtype=self.ARRAY_DTYPE)
            t1 = time.time()
            print('Process templates: {0:.3f} s'.format(t1-t0))
        else:
            self.tempfilt = tempfilt
            
        ### Template Error
        self.set_template_error(compute_tef_lnp=compute_tef_lnp)
        
        self.ubvj = None
        
        ### Load previous products?
        if load_products:
            self.load_products(**kwargs)


    @property 
    def to_flam(self):
        """
        Conversion factor to 1e-19 erg/s/cm**2/A
        """
        to_flam = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
        to_flam *= utils.CLIGHT*1.e10/1.e-19/self.lc**2/self.ext_corr
        return to_flam


    @property 
    def to_uJy(self):
        """
        Conversion of observed fluxes to microJansky
        """
        return 10**(-0.4*(self.param.params['PRIOR_ABZP']-23.9))


    @property 
    def NOBJ(self):
        """
        Number of objects in catalog
        """
        if not hasattr(self, 'cat'):
            return 0
        else:
            return len(self.cat)


    @property
    def NFILT(self):
        """
        Number of filters
        """
        if hasattr(self, 'filters'):
            return len(self.filters)
        else:
            return 0


    @property 
    def lc(self):
        """
        Filter pivot wavelengths
        """        
        if hasattr(self, 'filters'):
            return np.array([f.pivot for f in self.filters])
        else:
            return None


    @property 
    def NTEMP(self):
        """
        Number of templates
        """
        if hasattr(self, 'templates'):
            return len(self.templates)
        else:
            return 0


    @property
    def NZ(self):
        """
        Number of redshift grid points
        """
        if hasattr(self, 'zgrid'):
            return len(self.zgrid)
        else:
            return 0


    def load_products(self, compute_error_residuals=False, fitter='nnls', **kwargs):
        """
        Load results from ``zout`` and ``data`` FITS files.
        """
        zout_file = '{0}.zout.fits'.format(self.param['MAIN_OUTPUT_FILE'])
        if os.path.exists(zout_file):
            print('Load products: {0}'.format(zout_file))

            data_file = '{0}.data.fits'.format(self.param['MAIN_OUTPUT_FILE'])
            data = pyfits.open(data_file)
            self.chi2_fit = data['CHI2'].data*1
            #self.compute_pz()
            self.compute_lnp()
            if 'REST_UBVJ' in data:
                self.ubvj = data['REST_UBVJ'].data*1
            
            self.zout = Table.read(zout_file)
            
            print(' ... Fit templates at zout[z_phot] ')
            
            if compute_error_residuals:
                for iter in range(2):
                    self.fit_at_zbest(zbest=self.zout['z_phot'].data, 
                                      prior=False, fitter=fitter, **kwargs)
                    self.error_residuals()
            else:
                self.fit_at_zbest(zbest=self.zout['z_phot'].data,
                              fitter=fitter, **kwargs)


    def read_catalog(self, verbose=True):
        """
        Read catalog file
        """
        if verbose:
            print('Read CATALOG_FILE:', self.param['CATALOG_FILE'])
             
        if 'fits' in self.param['CATALOG_FILE'].lower():
            self.cat = Table.read(self.param['CATALOG_FILE'], format='fits')
        else:
            self.cat = Table.read(self.param['CATALOG_FILE'], 
                                  format='ascii.commented_header')
        
        if verbose:
            print(f'   >>> NOBJ = {len(self.cat)}')    
            
        # self.NOBJ = len(self.cat)
        self.prior_mag_cat = np.zeros(self.NOBJ)-1
        
        all_filters = filters.FilterFile(self.param['FILTERS_RES'])
        self.RES = all_filters
        np.save(self.param['FILTERS_RES']+'.npy', [all_filters])

        self.filters = []
        self.flux_columns = []
        self.err_columns = []
        self.f_numbers = []
        
        for k in self.cat.colnames:
            if k.startswith('F'):
                try:
                    f_number = int(k[1:])
                except:
                    continue
            
                ke = k.replace('F','E')
                if ke not in self.cat.colnames:
                    continue
                
                self.filters.append(self.RES[f_number])
                self.flux_columns.append(k)
                self.err_columns.append(ke)
                self.f_numbers.append(f_number)
                msg = '{0} {1} ({2:3d}): {3}'
                print(msg.format(k, ke, f_number, 
                                 self.filters[-1].name.split()[0]))
                
        # Apply translation       
        for k in self.translate.trans:
            fcol = self.translate.trans[k]
            if fcol.startswith('F') & ('FTOT' not in fcol):
                try:
                    f_number = int(fcol[1:])
                except:
                    # Has character at the end
                    f_number = int(fcol[1:-1])
                    
                for ke in self.translate.trans:
                    #if self.translate.trans[ke] == 'E{0}'.format(f_number):
                    if self.translate.trans[ke] == fcol.replace('F','E'):
                        break
                                 
                if (k in self.cat.colnames) & (ke in self.cat.colnames):
                    self.filters.append(self.RES[f_number])
                    self.flux_columns.append(k)
                    self.err_columns.append(ke)
                    self.f_numbers.append(f_number)
                    msg = '{0} {1} ({2:3d}): {3}'
                    print(msg.format(k, ke, f_number, 
                                     self.filters[-1].name.split()[0]))
                        
        self.f_numbers = np.array(self.f_numbers)
        
        #self.lc = np.array([f.pivot for f in self.filters])
                
        #self.NFILT = len(self.filters)
        self.fnu = np.zeros((self.NOBJ, self.NFILT), dtype=self.ARRAY_DTYPE)
        efnu = np.zeros((self.NOBJ, self.NFILT), dtype=self.ARRAY_DTYPE)
        self.spatial_offset = None
        
        self.fmodel = self.fnu*0.
        self.efmodel = self.fnu*0.
        
        # MW extinction correction: dered = fnu/self.ext_corr
        ext_mag = [f.extinction_correction(self.param.params['MW_EBV']) 
                   for f in self.filters]
        self.ext_corr = 10**(0.4*np.array(ext_mag))

        # Does catalog already have extinction correction applied?
        # If so, then set an array to put fluxes back in reddened space
        if self.param.params['CAT_HAS_EXTCORR'] in TRUE_VALUES:
            self.ext_redden = self.ext_corr
        else:
            self.ext_redden = np.ones(self.NFILT)

        self.zp = self.ext_corr*0+1
        
        for i in range(self.NFILT):
            self.fnu[:,i] = self.cat[self.flux_columns[i]]*1
            efnu[:,i] = self.cat[self.err_columns[i]]*1
            if self.err_columns[i] in self.translate.error:
                efnu[:,i] *= self.translate.error[self.err_columns[i]]
                
        self.efnu_orig = efnu*1.
        #self.fnu_orig = self.fnu*1.
        
        self.efnu = np.sqrt(self.efnu_orig**2+(self.param['SYS_ERR']*self.fnu)**2)
        
        self.ok_data = ((self.efnu > 0) 
                        & (self.fnu > self.param['NOT_OBS_THRESHOLD']) 
                        & np.isfinite(self.fnu) 
                        & np.isfinite(self.efnu))
                         
        self.fnu[~self.ok_data] = self.param['NOT_OBS_THRESHOLD'] - 9
        self.efnu[~self.ok_data] = self.param['NOT_OBS_THRESHOLD'] - 9
        
        self.nusefilt = self.ok_data.sum(axis=1)
        self.lc_reddest = np.max(self.ok_data*self.lc, axis=1)
        self.lc_zmax = self.zgrid.max()
        self.clip_wavelength = None


    def read_zeropoint(self, zeropoint_file='zphot.zeropoint'):
        """
        Read zphot.zeropoint file
        """
        lines = open(zeropoint_file).readlines()
        for line in lines:
            if not line.startswith('F'):
                continue
            
            fnum = int(line.strip().split()[0][1:])
            if fnum in self.f_numbers:
                ix = self.f_numbers == fnum
                self.zp[ix] = float(line.split()[1])


    def set_template_error(self, TEF=None, compute_tef_lnp=True):
        """
        Set the Template Error Function 
        """
        if TEF is None:
            TEF = templates_module.TemplateError(self.param['TEMP_ERR_FILE'], 
                                              lc=self.lc, 
                                              scale=self.param['TEMP_ERR_A2'])
        
        self.TEF = TEF
        
        self.TEFgrid = np.zeros((self.NZ, self.NFILT), dtype=self.ARRAY_DTYPE)
        for i in range(self.NZ):
            self.TEFgrid[i,:] = self.TEF(self.zgrid[i])
        
        # lnP term for TEF
        if compute_tef_lnp:
            self.compute_tef_lnp(in_place=True)


    def set_zgrid(self):
        """
        Set zgrid attributes from Z_MIN, Z_MAX, Z_STEP params
        """
        zr = [self.param['Z_MIN'], self.param['Z_MAX']]
        self.zgrid = utils.log_zgrid(zr=zr, 
                        dz=self.param['Z_STEP']).astype(self.ARRAY_DTYPE)
        #self.NZ = len(self.zgrid)
        self.trdz = utils.trapz_dx(self.zgrid)
        
    def prior_beta(self, w1=1350, w2=1800, dw=100, sample=None, width_params={'k':-5, 'z_split':4, 'sigma0':20, 'sigma1':0.5, 'center':-1.5}):
        """
        Prior on UV slope beta to try to fix red low-z galaxies put at z>4.  
        
        Beta is defined here as the logarithmic slope between two filters 
        with width `dw` evaluated at wavelengths `w1` and `w2`, set closer
        to the Lyman break than the usual definition to handle cases at 
        z>10 where the slope might be constrained by only a single filter.
        
        To evaluate the prior, the likelihood of the observed beta(z) is 
        computed from a normal distribution with redshift-dependent width set
        by a logistic function
            
            >>> import numpy as np
            >>> zgrid = np.arange(0.1, 6, 0.010)
            >>> k = -5
            >>> z_split = 4
            >>> sigma0 = 20
            >>> sigma1 = 0.5
            >>> sigma_beta_z = 1./(1+np.exp(-k*(zgrid - z_split)))*sigma0 + sigma1
        
        that has width sigma0 at z < z_split and sigma1 otherwise. `center` 
        specifies the middle of the beta distribution.
        
        The prior function is the observed beta drawn from this distribution
        at each redshift.
        
        """        
        from scipy.stats import norm as normal_distribution
        
        wx = np.arange(-0.7*dw, 0.7*dw)
        wy = wx*0.
        wy[np.abs(wx) <= dw/2.] = 1
        
        f1 = filters.FilterDefinition(wave=wx+w1, throughput=wy)
        f2 = filters.FilterDefinition(wave=wx+w2, throughput=wy)
        
        y1 = [t.integrate_filter(f1, flam=True) for t in self.templates]
        y2 = [t.integrate_filter(f2, flam=True) for t in self.templates]
        ln_beta_x = np.log([w1, w2])
        beta_y = np.array([y1, y2]).T
        
        if sample is not None:
            fit_beta_y = np.dot(self.fit_coeffs[sample,:,:], beta_y)
        else:
            fit_beta_y = np.dot(self.fit_coeffs, beta_y)

        ln_fit_beta_y = np.log(fit_beta_y)
        out_beta_y = np.squeeze(np.diff(ln_fit_beta_y, axis=2))/np.diff(ln_beta_x)[0]
        
        # Width of beta distribution, logistic
        #k = -5
        wi = {'k':-5, 'z_split':4, 'sigma0':100, 'sigma1':1, 'center':-1.5}
        for k in width_params:
            wi[k] = width_params[k]
            
        sigma_z = 1./(1+np.exp(-wi['k']*(self.zgrid - wi['z_split']))) 
        sigma_z = sigma_z*wi['sigma0'] + wi['sigma1']
        
        p_beta = np.zeros_like(out_beta_y)
        for i in range(self.NZ):
            n_i = normal_distribution(loc=wi['center'], scale=sigma_z[i])
            p_beta[:,i] = (1 - n_i.cdf(out_beta_y[:,i]))
            
        return p_beta


    def read_prior(self, verbose=True):
        """
        Read ``PRIOR_FILE`` 
        """
        if not os.path.exists(self.param['PRIOR_FILE']):
            return False
            
        prior_raw = np.loadtxt(self.param['PRIOR_FILE'])
        prior_header = open(self.param['PRIOR_FILE']).readline()
        
        self.prior_mags = np.cast[float](prior_header.split()[2:])
        self.prior_data = np.zeros((self.NZ, len(self.prior_mags)))
                                   
        for i in range(self.prior_data.shape[1]):
            self.prior_data[:,i] = np.interp(self.zgrid, prior_raw[:,0], 
                                             prior_raw[:,i+1], 
                                             left=0, right=0)
        
        self.prior_data /= np.trapz(self.prior_data, self.zgrid, axis=0)
        
        if 'PRIOR_FLOOR' in self.param.params:
            prior_floor = self.param['PRIOR_FLOOR']
            self.prior_data += prior_floor
            self.prior_data /= np.trapz(self.prior_data, self.zgrid, axis=0)

        self.prior_map_z = np.arange(self.NZ)

        if isinstance(self.param['PRIOR_FILTER'], str):
            ix = self.flux_columns.index(self.param['PRIOR_FILTER'])
            ix = np.arange(self.NFILT) == ix
        else:
            ix = self.f_numbers == int(self.param['PRIOR_FILTER'])
            
        if ix.sum() == 0:
            print('PRIOR_FILTER ({0}) not found in the catalog!'.format(self.param['PRIOR_FILTER']))
            
            self.prior_mag_cat = np.zeros(self.NOBJ, dtype=self.ARRAY_DTYPE)-1
            
        else:
            self.prior_mag_cat = self.param['PRIOR_ABZP'] 
            self.prior_mag_cat += -2.5*np.log10(np.squeeze(self.fnu[:,ix]))
            self.prior_mag_cat[~np.isfinite(self.prior_mag_cat)] = -1
            
            for i in range(self.NOBJ):
                if self.prior_mag_cat[i] > 0:
                    #print(i)
                    pz = self._get_prior(self.prior_mag_cat[i])
                    self.full_logprior[i,:] = np.log(pz)

        if verbose:
            print('Read PRIOR_FILE: ', self.param['PRIOR_FILE'])
    
    def _get_prior(self, mag):
        mag_clip = np.clip(mag, self.prior_mags[0], self.prior_mags[-1]-0.02)
        
        mag_ix = np.interp(mag_clip, self.prior_mags, np.arange(len(self.prior_mags)))
        int_mag_ix = int(mag_ix)
        f = mag_ix-int_mag_ix
        prior = np.dot(self.prior_data[:,int_mag_ix:int_mag_ix+2], [f,1-f])
        return prior


    def iterate_zp_templates(self, idx=None, update_templates=True, update_zeropoints=True, iter=0, n_proc=4, save_templates=False, error_residuals=False, prior=True, get_spatial_offset=False, spatial_offset_keys={'apply':True}, **kwargs):
        """
        Iterative detemination of zeropoint corrections
        """
        self.fit_parallel(idx=idx, n_proc=n_proc, prior=prior)        
        if error_residuals:
            self.error_residuals()
        
        if idx is not None:
            selection = np.zeros(self.NOBJ, dtype=bool)
            selection[idx] = True
        else:
            selection = None

        label = '{0}_zp_{1:03d}'.format(self.param['MAIN_OUTPUT_FILE'], iter)
        
        fig = self.residuals(update_zeropoints=update_zeropoints,
                       ref_filter=int(self.param['PRIOR_FILTER']),
                       selection=selection, update_templates=update_templates,
                       full_label=label, **kwargs)

        fig_file = '{0}.png'.format(label)
        fig.savefig(fig_file)
        
        if get_spatial_offset:
            self.spatial_statistics(catalog_mask=selection, output_suffix='_{0:03d}'.format(iter), **spatial_offset_keys)
            
        if save_templates:
            self.save_templates()
    
    def zphot_zspec(self, selection=None, min_zphot=0.02, zmin=0, zmax=4, include_errors=True, **kwargs):
        """
        Make zphot - zspec comparison plot
        """
        clip = (self.zbest > min_zphot) 
        clip &= (self.cat['z_spec'] > zmin) & (self.cat['z_spec'] <= zmax)
        
        if selection is not None:
            clip &= selection
            
        if include_errors:
            zlimits = self.pz_percentiles(percentiles=[16,84], oversample=5,
                                      selection=clip)
        else:
            zlimits = None
            
        fig = utils.zphot_zspec(self.zbest, self.cat['z_spec'], 
                          zlimits=zlimits, 
                          selection=selection, min_zphot=min_zphot, 
                          zmin=zmin, zmax=zmax, **kwargs)
        
        return fig


    def save_templates(self, prefix='corr_', ext=None, format=None, overwrite=True, make_param=True):
        """
        Write scaled versions of the templates
        """
        import shutil
        
        pardir = os.path.dirname(self.param['TEMPLATES_FILE'])
        parfile = os.path.basename(self.param['TEMPLATES_FILE'])
                
        new_files = []
        
        for i, templ in enumerate(self.templates):
            tab = templ.to_table()
            
            templ_file = os.path.join('{0}/{1}{2}'.format(pardir, prefix,
                                                          templ.name))
            
            new_files.append(templ_file)
                            
            if ext is not None:
                templ_file += ext
            
            print('Save modified template {0}'.format(templ_file))
            
            file_ext = templ_file.split('.')[-1]
            if (file_ext in ['dat', 'txt']) & (format is None):
                fmt = 'ascii.commented_header'
            else:
                fmt = format
                
            if format is not None:
                tab.write(templ_file, format=format, overwrite=overwrite)    
            else:
                tab.write(templ_file, overwrite=overwrite)

        if make_param:
            
            new_parfile = os.path.join(pardir, prefix+parfile)
            print(f'{parfile} >> {new_parfile}')
            with open(new_parfile,'w') as fp:
                for i, file in enumerate(new_files):
                    fp.write(f'{i+1} {file} 1.0\n')
            
            parfits = os.path.join(pardir, parfile+'.fits')
            if os.path.exists(parfits):
                new_parfits = os.path.join(pardir, prefix+parfile+'.fits')
                print(f'{parfits} >> {new_parfits}')

                partab = Table.read(parfits)
                partab['file'] = [os.path.basename(f) for f in new_files]                
                partab.write(new_parfits, overwrite=True)


    def fit_single_templates(self, verbose=True):
        """
        Fit individual templates on the redshift grid
        """
        from tqdm import tqdm
        
        ampl = np.zeros((self.NTEMP, self.NOBJ, self.NZ), 
                        dtype=self.ARRAY_DTYPE)
        chi2 = np.zeros((self.NTEMP, self.NOBJ, self.NZ),
                        dtype=self.ARRAY_DTYPE)
        
        chiz = np.zeros((self.NZ, self.NOBJ),
                        dtype=self.ARRAY_DTYPE)
        amplz = np.zeros((self.NZ, self.NOBJ),
                         dtype=self.ARRAY_DTYPE)
                
        for i in range(self.NTEMP):
            print('Process template ', i)
            templ_i = self.tempfilt.tempfilt[:,i,:].T
            
            for j in tqdm(range(self.NZ)):
                #print(j)
                tefz = self.TEF(self.zgrid[j])
                full_err = np.sqrt(self.efnu**2+(self.fnu*tefz)**2)
                templ_iz = templ_i[:,j]
                num = (self.fnu/self.zp/full_err*self.ok_data).dot(templ_iz)
                den = (1./full_err*self.ok_data).dot(templ_iz**2)
                ampl_j = num/den
                amplz[j,:] = ampl_j
                
                mz = ampl_j[:,None]*templ_iz[None,:]
                chi = ((mz-self.fnu/self.zp)*self.ok_data)**2/full_err**2
                chiz[j,:] = chi.sum(axis=1)
        
            chi2[i,:,:] = chiz.T
            ampl[i,:,:] = amplz.T
            
        chimin = chi2.min(axis=2).min(axis=0)
        if verbose:
            print('Compute p(z|T)')
            
        logpz = -(chi2 - chimin[None,:,None])/2
        
        pzt = np.exp(logpz).sum(axis=0)
        pznorm = np.trapz(pzt, self.zgrid, axis=1)
        logpz -= pznorm[None,:,None]
        
        return ampl, chi2, logpz


    def fit_parallel(self, idx=None, n_proc=4, verbose=True, get_best_fit=True, prior=False, beta_prior=False, fitter='nnls'):
        """
        This is the main function for fitting redshifts for a full catalog
        and is parallelized by fitting each redshift grid step separately.
        
        ``idx`` is the array indices of a subset if you don't want to fit
        the full catalog.
        
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        import multiprocessing as mp
        
        if idx is None:
            idx_fit = self.idx
            selection = self.idx > -1
        else:
            if idx.dtype == bool:
                idx_fit = self.idx[idx]
                selection = idx
            else:
                idx_fit = idx
                selection = None
                
        # Setup
        fnu_corr = self.fnu[idx_fit,:]*self.ext_redden*self.zp
        efnu_corr = self.efnu[idx_fit,:]*self.ext_redden*self.zp
        
        missing = self.fnu[idx_fit,:] < self.param['NOT_OBS_THRESHOLD']
        efnu_corr[missing] = self.param['NOT_OBS_THRESHOLD'] - 9.
        
        t0 = time.time()
        pool = mp.Pool(processes=n_proc)
        
        results = [pool.apply_async(_fit_vertical, 
                         (iz, self.zgrid[iz],  self.tempfilt(self.zgrid[iz]),
                          fnu_corr, efnu_corr, self.TEF, self.zp, 
                          self.param.params['VERBOSITY'], fitter)) 
                   for iz in range(self.NZ)]

        pool.close()
        pool.join()
        
        # Gather results
        for res in results:
            iz, chi2, coeffs = res.get(timeout=1)
            self.chi2_fit[idx_fit,iz] = chi2
            self.fit_coeffs[idx_fit,iz,:] = coeffs
        
        # Compute maximum likelihood redshift zml
        if get_best_fit:
            if verbose:
                print('Compute best fits')
            
            self.fit_at_zbest(zbest=None, prior=prior, beta_prior=beta_prior)
        else:
            self.compute_lnp(prior=prior, beta_prior=beta_prior)
            
        t1 = time.time()
        if verbose:
            msg = 'Fit {1:.1f} s (n_proc={0}, NOBJ={2})'
            print(msg.format(n_proc, t1-t0, len(idx_fit)))


    def fit_at_redshift(self, iobj, z=None, fitter='nnls'):
        """
        Fit templates at a single redshift
        """
        fnu_i = np.squeeze(self.fnu[iobj, :])*self.ext_redden*self.zp
        efnu_i = np.squeeze(self.efnu[iobj,:])*self.ext_redden*self.zp
        ok_band = (fnu_i/self.zp > self.param['NOT_OBS_THRESHOLD']) 
        ok_band &= (efnu_i/self.zp > 0)
        efnu_i[~ok_band] = self.param['NOT_OBS_THRESHOLD'] - 9.
        
        tef_i = self.TEF(z)
        
        A = np.squeeze(self.tempfilt(z))
        chi2_i, coeffs_i, fmodel, draws = _fit_obj(fnu_i, efnu_i, A, 
                                                   tef_i, self.zp, 
                                                   0, fitter)
        
        return chi2_i, coeffs_i, fmodel


    def fit_on_zgrid(self, iobj, fitter='nnls'):
        """
        Fit a single object on the redshift grid
        """
        chi2 = np.zeros(self.NZ, dtype=self.ARRAY_DTYPE)
        coeffs = np.zeros((self.NZ, self.NTEMP), dtype=self.ARRAY_DTYPE)
    
        for iz, z in enumerate(self.zgrid):
            _ = self.fit_at_redshift(iobj, z=z, fitter=fitter)
            chi2[iz,:], coeffs[iz] = _[:2]
        
        return iobj, chi2, coeffs


    def evaluate_zml(self, prior=False, beta_prior=False, clip_wavelength=1100):
        """
        Evaluate the maximum likelihood redshift with optional priors
        """
        #izbest = np.argmin(self.chi2_fit, axis=1)
        izbest = self.izbest*1
        has_chi2 = (self.chi2_fit != 0).sum(axis=1) > 0 
        
        self.compute_lnp(prior=prior, beta_prior=beta_prior, 
                     clip_wavelength=clip_wavelength) 

        self.zml, _lnpmax = self.get_maxlnp_redshift(prior=prior,
                                                beta_prior=beta_prior,
                                        clip_wavelength=clip_wavelength)
        
        self.zml[~has_chi2] = -1
        
        self.ZML_WITH_PRIOR = prior
        self.ZML_WITH_BETA_PRIOR = beta_prior


    def fit_at_zbest(self, zbest=None, prior=False, beta_prior=False, get_err=False, clip_wavelength=1100, fitter='nnls', selection=None,  n_proc=0, par_skip=10000, **kwargs):
        """
        Recompute the fit coefficients at the "best" redshift.  
        
        If ``zbest`` not specified, then will fit at the maximum likelihood
        redshift ``self.zml``.
        
        """
        import multiprocessing as mp
                
        #izbest = np.argmin(self.chi2_fit, axis=1)
        izbest = self.izbest*1
        has_chi2 = (self.chi2_fit != 0).sum(axis=1) > 0 
        
        self.get_err = get_err
        
        self.zbest_grid = self.zgrid[izbest]
        #self.chi_best
        
        if zbest is None:
            if (self.zml is None):
                recompute_zml = True
            else:
                # Recompute if prior options changed
                recompute_zml = prior is not self.ZML_WITH_PRIOR
                recompute_zml |= beta_prior is not self.ZML_WITH_BETA_PRIOR
                
            if recompute_zml:
                self.evaluate_zml(prior=prior, beta_prior=beta_prior)

            self.ZPHOT_USER = False # user did *not* specify zbest
            self.zbest = self.zml
                            
        else:
            self.zbest = zbest
            self.ZPHOT_USER = True # user *did* specify zbest
                    
        if ((self.param['FIX_ZSPEC'] in TRUE_VALUES) & 
            ('z_spec' in self.cat.colnames)):
            has_zsp = self.cat['z_spec'] > self.zgrid[0]
            self.zbest[has_zsp] = self.cat['z_spec'][has_zsp]
            self.ZPHOT_AT_ZSPEC = True
        else:
            self.ZPHOT_AT_ZSPEC = False
            
        # Compute Risk function at z=zbest
        self.zbest_risk = self.compute_best_risk()
        
        fnu_corr = self.fnu*self.ext_redden*self.zp
        efnu_corr = self.efnu*self.ext_redden*self.zp
        efnu_corr[~self.ok_data] = self.param['NOT_OBS_THRESHOLD'] - 9.
        
        subset = (self.zbest > self.zgrid[0]) & (self.zbest < self.zgrid[-1])
        if selection is not None:
            subset &= selection
            
        idx = self.idx[subset]
        
        # Set seed
        np.random.seed(self.random_seed)
        
        if n_proc <= 0:
            n_proc = np.maximum(mp.cpu_count() - 2, 1)
        
        # Fit in parallel mode        
        t0 = time.time()
        
        skip = np.maximum(len(idx)//par_skip, 1)
        n_proc = np.minimum(n_proc, skip)
        
        pool = mp.Pool(processes=n_proc)
        results = [pool.apply_async(_fit_at_zbest_group, 
                                      (idx[i::skip], 
                                       fnu_corr[idx[i::skip],:], 
                                       efnu_corr[idx[i::skip],:], 
                                       self.zbest[idx[i::skip]], 
                                       self.zp*1, get_err, 
                                       fitter, self.tempfilt, self.TEF,
                                       self.ARRAY_DTYPE, None)) 
                    for i in range(skip)]

        pool.close()
        pool.join()

        for res in results:
            _ = res.get(timeout=1)
            _ix, _coeffs_best, _fmodel, _efmodel, _chi2_best, _coeffs_draws = _
            self.coeffs_best[_ix,:] = _coeffs_best
            self.fmodel[_ix,:] = _fmodel
            self.efmodel[_ix,:] = _efmodel
            self.chi2_best[_ix] = _chi2_best
            self.coeffs_draws[_ix,:,:] = _coeffs_draws
        
        t1 = time.time()
        print(f'fit_best: {t1-t0:.1f} s (n_proc={n_proc}, '
              f' NOBJ={subset.sum()})')


    def error_residuals(self, level=1, verbose=True):
        """
        Force error bars to touch the best-fit model
        """
        
        if verbose:
            print('`error_residuals`: force uncertainties to match residuals')
            
        #self.efnu = self.efnu_orig*1
        self.efnu = np.sqrt(self.efnu_orig**2+(self.param['SYS_ERR']*self.fnu)**2)

        # residual
        r = np.abs(self.fmodel - self.fnu*self.ext_redden*self.zp)
        
        # Update where residual larger than uncertainty
        upd = (self.efnu > 0) & (self.fnu > self.param['NOT_OBS_THRESHOLD'])
        upd &= (r > level*self.efnu) & (self.fmodel > 0)
        upd &= np.isfinite(self.fnu) & np.isfinite(self.efnu)
        
        self.efnu[upd] = r[upd] #np.sqrt(var_new[upd])


    def check_uncertainties(self, apply_correction=False):
        """
        *testing*
        
        Trying to recalibrate uncertainties based on full catalog fits
        """
        import astropy.stats
        from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
        
        TEF_scale = 1.
        
        #izbest = np.argmin(self.chi2_fit, axis=1)
        izbest = self.izbest*1
        zbest = self.zgrid[izbest]
        
        full_err = self.efnu*0.
        teff_err = self.efnu*0
        
        teff_err = self.TEF(np.maximum(self.zbest[:,None], self.zgrid[1]))         
            
        resid = (self.fmodel - self.fnu*self.ext_redden*self.zp)/self.fmodel
        
        self.efnu_i = self.efnu_orig*1
                
        eresid = np.sqrt((self.efnu_i/self.fmodel)**2+self.param.params['SYS_ERR']**2 + teff_err**2)
        
        okz = (self.zbest > 0.1) & (self.zbest < 3)
        scale_errors = self.lc*0.
        
        for ifilt in range(self.NFILT):
            iok = okz & (self.efnu_orig[:,ifilt] > 0) & (self.fnu[:,ifilt] > self.param['NOT_OBS_THRESHOLD'])
            iok &= np.isfinite(resid[:,ifilt])
            if iok.sum() < 10:
               continue
                
            # Spline interp
            xw = self.lc[ifilt]/(1+self.zbest[iok])
            so = np.argsort(xw)
            #spl = UnivariateSpline(xw[so], resid[iok,ifilt][so], w=1/np.clip(eresid[iok,ifilt][so], 0.002, 0.1), s=iok.sum()*4)
            spl = LSQUnivariateSpline(xw[so], resid[iok,ifilt][so], np.exp(np.arange(np.log(xw.min()+100), np.log(xw.max()-100), 0.2)), w=1/eresid[iok,ifilt][so])#, s=10)
            
            nm = utils.nmad((resid[iok,ifilt]-spl(xw))/eresid[iok,ifilt])
            print('{3}: {0} {1:d} {2:.2f}'.format(self.flux_columns[ifilt], self.f_numbers[ifilt], nm, ifilt))
            scale_errors[ifilt] = nm
            if apply_correction:
                self.efnu_i[:,ifilt] *= nm
            
            #plt.hist(resid[iok,ifilt], bins=100, range=[-3,3], alpha=0.5)
        
        # Overall average
        lcz = np.dot(1/(1+self.zbest[:, np.newaxis]), self.lc[np.newaxis,:])


    def make_template_error_function(self, te_wave=None, log_wave=True, selection=None, optimizer=None, optimizer_args={}, in_place=False, sn_limits=[-2, 100], min_err=0.02, scale_errors=False):
        """
        Generate a template error function based on template fit residuals
        
        *under development*
        """    
        from scipy.optimize import nnls, minimize
        
        if optimizer is None:
            optimizer = nnls
            
        if selection is None:
            selection = (self.zbest > 0.1) & (self.zbest < 5)
                
        sigma = self.efnu_orig[selection,:]
        if hasattr(self, 'err_scale') & scale_errors:
            sigma *= self.err_scale
            
        M = (self.fmodel/self.ext_redden/self.zp)[selection,:]*1
        F = self.fnu[selection,:]*1
        lcz = np.dot(1/(1+self.zbest[:, np.newaxis]), self.lc[np.newaxis,:])
        lcz = lcz[selection]
        
        clip = ((M-F)**2 < 25*(sigma**2+(0.03*M)**2)) & np.isfinite(M) & np.isfinite(F) & (M > 0) & (sigma > 0)
        
        _A = np.zeros((clip.sum(), self.NFILT), dtype=self.ARRAY_DTYPE)
        j = 0
        _r = np.zeros(clip.sum())
        _m = np.zeros(clip.sum())
        _w = _m*0.
        _ix = _A*0
        
        for i in range(self.NFILT):
            print(i, j)
            clip_i = clip[:,i]
            csum = clip_i.sum()
            _A[j:j+csum,i] = sigma[clip_i, i]**2
            _ix[j:j+csum,i] = i+1
            _r[j:j+csum] = (F-M)[clip_i, i]
            _m[j:j+csum] = M[clip_i, i]**2
            _w[j:j+csum] = lcz[clip_i, i]
            j += csum
        
        _y = _r**2
            
        _N = clip.sum(axis=0)
        _ok = _N > 300
        _Ax = np.hstack([_A[:, _ok], 0.03**2*_m[:,None]])
        _NF = _ok.sum()
        
        df = 9
        Aspl = bspline_templates(_w, degree=3, df=df, 
                                       get_matrix=True, log=log_wave)
        
        NTEF = Aspl.shape[1]
        
        TEF0 = 0.03
        #_Ax = np.hstack([(_m*(TEF0*Aspl.T)**2).T, _A[:, _ok]])
        _Ax = np.hstack([_A[:, _ok], (_m*(TEF0*Aspl.T)**2).T])
        
        RHS = _y.sum()
        def _objfun_nmad(scl, _Ax, _r, _NF, norm, indices, verb):
            val = 0.
            LHS = _Ax*(scl/norm)**2
            
            sig = np.sqrt(LHS.sum(axis=1))
            #val = (utils.nmad(_r/sig)-1)**2
            val = 0.
            for ix in indices:
                val += (utils.nmad((_r/sig)[ix])-1)**2
                
            #val = (RHS - (_Ax*scl/norm).sum())**2
            #val += ((scl[-_NF:]-norm)**2).sum()
            #val += ((scl[:_NF]/norm-1)**2).sum()
            #val += ((scl[:]/norm-1)**2).sum()*ix.sum()
            if verb:
                print('{0} {1:.4f}'.format(scl/norm, val))
                
            return val
        
        def _objfun_resid(scl, _Ax, _y, _NF, norm, verb):
            val = 0.
            LHS = _Ax*(scl/norm)**2
            for i in range(_NF):
                ix = _Ax[:,i] > 0
                val += (_y[ix].sum() - LHS[ix,:].sum())**2
                
            #val = (RHS - (_Ax*scl/norm).sum())**2
            #val += ((scl[-_NF:]-norm)**2).sum()
            #val += ((scl[:_NF]/norm-1)**2).sum()
            #val += ((scl[:]/norm-1)**2).sum()*ix.sum()
            if verb:
                print('{0} {1:.1f}'.format(scl/norm, val))
                
            return val
        
        x0 = np.ones(_Ax.shape[1])
        #x0[0] = 2.
        #x0[NTEF-2:NTEF] = 10.
        
        norm = 20.
        
        args = (_Ax, _y, _NF, norm, True)
        _x = minimize(_objfun_resid, x0*norm, args=args, method='COBYLA')

        indices = [_Ax[:,i] > 0 for i in range(_NF)]
        args = (_Ax, _r, _NF, norm, indices, True)
        _x = minimize(_objfun_nmad, x0*norm, args=args, method='COBYLA')
        
        coeffs = (_x.x/norm)
        tef_y = np.dot(Aspl, coeffs[-NTEF:])*TEF0
        
        LHS = _Ax*(coeffs**2)
        sig = np.sqrt(LHS.sum(axis=1))
        sig_band = np.sqrt(LHS[:, :_NF].sum(axis=1))
        
        _band_ix = np.where(_ok)[0]
        
        for i in range(_NF):
            ix = indices[i]
            msg = '{0:>10} {1:.3f}  {2:.2f} {3:.2f}'
            print(msg.format(self.flux_columns[_band_ix[i]], coeffs[i], 
                             utils.nmad(_r[ix]/sig[ix]), 
                             utils.nmad(_r[ix]/sig_band[ix])))
        
        # Normalize residuals, uncertainties by model flux
        E2 = ((F-M)/M)**2 - (sigma/M)**2 
        clip = (sigma.flatten() > 0) & np.isfinite(M.flatten()) 
        clip &= np.isfinite(F.flatten())
        clip &= (np.isfinite(E2.flatten())) & (np.abs(E2.flatten()) < 2)
        
        SN = (self.fnu[selection,:]/sigma).flatten()
        clip &= (SN > sn_limits[0]) & (SN < sn_limits[1])
        
        M = M.flatten()[clip]
        F = F.flatten()[clip]
        sigma = sigma.flatten()[clip]
        lcz = lcz.flatten()[clip]
        E2 = ((F-M)/M)**2 - (sigma/M)**2 
        
        so = np.argsort(lcz)
        
        wave_inp = lcz
            
        Aspl = bspline_templates(wave_inp, degree=3, df=7, 
                                       get_matrix=True, log=log_wave)
        
        # Sampled
        wave_samp = np.linspace(wave_inp.min(), wave_inp.max(), 1024)
        Asamp = bspline_templates(wave_samp, degree=3, df=7, 
                                        get_matrix=True, log=log_wave)
        
        _lsq = optimizer(Aspl, E2, **optimizer_args)
        coeffs = _lsq[0]
        spline_samp = Asamp.dot(coeffs)
        
        if te_wave is None:
            te_wave = self.TEF.te_x*1
        
        te_y = np.interp(te_wave, wave_samp, np.maximum(spline_samp, min_err), 
                         left=spline_samp[0], right=spline_samp[-1])
        
        if in_place:
            self.TEF = templates_module.TemplateError(file='internal', 
                                                      arrays=(te_wave, te_y),
                                                      lc=self.lc, scale=1.0)

            self.TEFgrid = np.zeros((self.NZ, self.NFILT),
                                    dtype=self.ARRAY_DTYPE)
            for i in range(self.NZ):
                self.TEFgrid[i,:] = self.TEF(self.zgrid[i])
            
            
        return te_wave, te_y


    def residuals(self, selection=None, minsn=3, resid_sig_clip=5, update_zeropoints=False, update_templates=False, ref_filter=None, correct_zp=True, n_knots=-1, use_bspline=False, logspline=True, wlimits=[1000, 3.e4], runmed_kwargs={'NBIN':16}, zpanel_kwargs={'zmin':0, 'zmax':4, 'catastrophic_limit':0.15}, full_label=None, ignore_zeropoint=False, ignore_spline=False, run_iterative=True, skip_filters=[], iterative_nsteps=3, **kwargs):
        """
        Show residuals and compute zeropoint offsets

        selection=None
        update_zeropoints=False
        update_templates=False
        ref_filter=226
        correct_zp=True
        NBIN=None

        """
        import os
        import matplotlib as mpl
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import scipy.interpolate

        #import threedhst
        #from astroML.sum_of_norms import sum_of_norms, norm

        #izbest = np.argmin(self.chi2_fit, axis=1)
        izbest = self.izbest*1
        zbest = self.zgrid[izbest]

        fnu_i = self.fnu*self.ext_redden*self.zp
        resid = (self.fmodel - fnu_i)/self.fmodel+1

        valid = (zbest > self.zgrid[0]) & (zbest < self.zgrid[-1])
        if selection is not None:
            valid &= selection

        idx = self.idx[valid]

        ## Full variance
        teff_err = self.TEF(np.maximum(self.zbest[:,None], self.zgrid[1]))         
        var = fnu_i**2*(self.param.params['SYS_ERR']**2 + teff_err**2)
        var += (self.efnu*self.ext_redden*self.zp)**2
        var *= self.ok_data
        inv_sig = 1/np.sqrt(var)
        del(var)

        ## Data to use
        sn = self.fnu/self.efnu_orig
        clip = (sn > minsn) & np.isfinite(inv_sig)
        clip &= (self.fnu > self.param['NOT_OBS_THRESHOLD']) 
        clip &= (resid > 0) & (self.fmodel != 0)
        clip &= np.isfinite(self.fnu) & np.isfinite(self.efnu) 
        clip &= np.abs(self.fmodel - fnu_i)*inv_sig < resid_sig_clip
        clip[~valid,:] = False

        ## Helper arrays
        # Redshifted filter wavelengths
        lcz = np.dot(1/(1+self.zgrid[izbest][:, np.newaxis]),
                     self.lc[np.newaxis,:])

        so = np.argsort(lcz[clip])
        lczso = lcz[clip][so]

        yp, xp = np.indices(lcz.shape)
        xpc = xp[clip][so]
        del(xp)
        del(yp)

        ## Splines for template correction
        if n_knots < 0:
            n_knots = self.NFILT

        wlim = np.clip(wlimits, lcz[clip][so][0], lcz[clip][so][-1])

        if logspline:
            wfunc = np.log
        else:
            wfunc = np.abs # Dummy

        if use_bspline:
            bspl = bspline_templates(wfunc(lcz[clip][so]), df=n_knots, 
                                    log=False, get_matrix=True, 
                                    minmax=wfunc(wlim))
        else:
            # Gaussian knots
            wclip = (lczso > wlim[0]) & (lczso < wlim[1])

            wi = np.interp(np.linspace(0, 1, n_knots),
                           np.cumsum(np.ones(wclip.sum()))/wclip.sum(), 
                           lczso[wclip])
            dw = np.gradient(wi)

            bspl = gaussian_templates(lczso, centers=wi, widths=dw)

        # Pedestal
        #bspl = np.hstack([np.ones((clip.sum(), 1)), bspl])

        ## Design matrix for lstsq optimization
        sh = bspl.shape
        _A = np.hstack([np.zeros((sh[0], self.NFILT)), bspl])

        for i in range(self.NFILT):
            if self.flux_columns[i] not in skip_filters:
                _A[xpc == i, i] = 1

        # Iterative
        if run_iterative:
            print('Iterative correction - zeropoint / template')

            fmodel = self.fmodel[clip][so]*1.
            mask_zeropoint = np.arange(_A.shape[1]) < self.NFILT
            mask_spline = np.arange(_A.shape[1]) > self.NFILT
            _Af = _A[:,:self.NFILT].T*inv_sig[clip][so]
            nzf = _Af.sum(axis=1) != 0
            _Af = _Af[nzf]
            _As = _A[:,self.NFILT:].T*inv_sig[clip][so]
            nzs = _As.sum(axis=1) != 0
            _As = _As[nzs]

            #_f, _a = plt.subplots(1,1,figsize=(6,4))

            corr = np.ones_like(fmodel)
            zcorr = np.ones(nzf.sum())
            scorr = np.ones(nzs.sum())

            for _iter in range(iterative_nsteps):

                # Fit zeropoint
                _yi = (fnu_i[clip][so] - fmodel*corr)*inv_sig[clip][so]

                _res = np.linalg.lstsq((_Af*fmodel*corr).T, _yi, rcond=None)
                zcorr_i = _A[:,:self.NFILT][:,nzf].dot(_res[0])
                zcorr *= (1+_res[0])
                corr *= 1 + zcorr_i
                #print('Iterative zeropoints: ', _iter, _res[0])

                # Fit spline
                _yi = (fnu_i[clip][so] - fmodel*corr)*inv_sig[clip][so]
                _res = np.linalg.lstsq((_As*fmodel*corr).T, _yi, rcond=None)
                scorr_i = _A[:,self.NFILT:][:,nzs].dot(_res[0])
                scorr *= (1+_res[0])
                corr *= 1 + scorr_i

                #_a.plot(lcz[clip][so], 1 + scorr_i, label=f'iter {_iter}')

            _yx = (fnu_i[clip][so] - fmodel*corr)*inv_sig[clip][so]

            _N = _A.shape[1]
            coeffs = np.zeros(_N)
            coeffs[np.arange(_N)[:self.NFILT][nzf]] += zcorr - 1
            coeffs[np.arange(_N)[self.NFILT:][nzs]] += scorr - 1
            #_a.semilogx()
            #_a.set_ylim(0.8, 1.2)

        else:

            if ref_filter is None:
                print('!! Warning - normalization may not be constrained with `ref_filter = None` !!')

            if ignore_zeropoint:
                _A[:, :self.NFILT] = 0.
            elif ignore_spline:
                _A[:, self.NFILT:] = 0

            # weighted by uncertainties
            _Ax = _A.T*(self.fmodel*inv_sig)[clip][so]
            _yx = ((fnu_i - self.fmodel)*inv_sig)[clip][so]

            nonzero = _Ax.sum(axis=1) != 0

            ### Least squares
            _res = np.linalg.lstsq(_Ax[nonzero,:].T, _yx, rcond=None)
            coeffs = np.zeros(len(nonzero))
            coeffs[nonzero] = _res[0]

        # Zeropoints
        zp_i = 1 + coeffs[:self.NFILT]
        if ref_filter is None:
            #ref_filter = self.param['PRIOR_FILTER']
            zp_ref = 1.
        else:
            zp_ref = zp_i[self.f_numbers == ref_filter][0]
            zp_i /= zp_ref

        # Spline template correction
        templ_wave = self.templates[0].wave
        if use_bspline:
            templ_spl = bspline_templates(wfunc(templ_wave), df=n_knots, 
                                      log=False, get_matrix=True, 
                                      minmax=wfunc(wlim))
        else:
            templ_spl = gaussian_templates(templ_wave, centers=wi, 
                                                 widths=dw)

        #templ_spl = np.hstack([np.ones_like(templ_wave), templ_spl])

        templ_corr = (templ_spl.dot(coeffs[self.NFILT:])+1) * zp_ref
        if use_bspline & (not run_iterative):
            templ_corr[(templ_wave < wlim[0]) | (templ_wave > wlim[1])] = 1.

        # residuals with zeropoints
        _y = (fnu_i/self.fmodel)[clip][so]

        ####### Figure
        fig = plt.figure(figsize=[16,4])
        gs = gridspec.GridSpec(1, 4)

        # Spectrum
        ax = fig.add_subplot(gs[:,:3])
        ax.plot(templ_wave, templ_corr, color='k', linewidth=2, alpha=0.5, 
                zorder=10)

        ax.text(0.9, -0.05, f'N={len(idx)}', ha='left', va='top', 
                fontsize=8, transform=ax.transAxes)

        if full_label is not None:
            ax.text(0.95, 0.05, full_label, ha='right', va='bottom', 
                fontsize=8, transform=ax.transAxes)

        cmap = cm.rainbow
        cnorm = mpl.colors.Normalize(vmin=0, vmax=self.NFILT-1)

        self.zp_delta = zp_i
        if correct_zp:
            image_corrections = self.zp*0.+1
        else:
            image_corrections = 1/self.zp

        # Filters  
        for i, ifilt in enumerate(np.argsort(self.lc)):
            ix = xpc == ifilt
            if ix.sum() == 0:
                self.zp_delta[ifilt] = 1.
                continue

            color = cmap(cnorm(i))

            if correct_zp:
                delta_i = self.zp_delta[ifilt]
            else:
                delta_i = 1.

            #fname = os.path.basename(self.filters[ifilt].name.split()[0])
            #if fname.count('.') > 1:
            #    fname = '.'.join(fname.split('.')[:-1])
            fname = self.flux_columns[ifilt]

            label = '{0:30s} {1:.3f}'
            label = label.format(fname, delta_i/image_corrections[ifilt])

            _ = utils.running_median(lczso[ix], _y[ix], 
                                          **runmed_kwargs)
            xm, ym, ys, yn = _

            ax.plot(xm, ym/delta_i*image_corrections[ifilt], color=color, 
                    alpha=0.8, label=label, linewidth=2)

            yy = ym/delta_i*image_corrections[ifilt]        
            ax.fill_between(xm, yy-ys/np.sqrt(yn), yy+ys/np.sqrt(yn), 
                            color=color, alpha=0.1) 

        try:
            ax.plot(self.TEF.te_x, self.TEF.te_y*self.TEF.scale+1, 
                    color='pink', zorder=1001, label='TEF')
        except:
            pass

        ax.semilogx()
        ax.set_ylim(0.8,1.2)
        ax.set_xlim(800,8.e4)
        l = ax.legend(fontsize=6, ncol=5, loc='upper right', handlelength=0.4)
        l.set_zorder(-20)
        ax.grid()
        ax.vlines([2175, 3727, 5007, 6563.], 0.8, 1.0, linestyle='--', 
                  color='k', zorder=-18)
        ax.set_xlabel(r'$\lambda_\mathrm{rest}$')
        ax.set_ylabel('data / template')

        ## zphot-zspec
        dz = (self.zbest-self.cat['z_spec'])/(1+self.cat['z_spec'])
        clip = (izbest > 0) & (self.cat['z_spec'] > 0)

        ax = fig.add_subplot(gs[:,-1])
        utils.zphot_zspec(self.zbest, self.cat['z_spec'], axes=[ax], 
                          selection=valid, **zpanel_kwargs)

        fig.tight_layout(pad=0.1)

        # update zeropoints in self.zp
        if update_zeropoints:
            ref_ix = self.f_numbers == ref_filter
            self.zp /= self.zp_delta/self.zp_delta[ref_ix]
            self.zp[self.zp_delta == 1] = 1.

        # corrected templates
        if update_templates:
            print('Reprocess corrected templates')
            #w_best, locs, widths, xmin, xmax = self.tnorm
            for templ in self.templates:
                #templ = self.templates[itemp]

                templ_wave = templ.wave
                if use_bspline:
                    templ_spl = bspline_templates(wfunc(templ_wave), 
                                              df=n_knots, 
                                              log=False, get_matrix=True, 
                                              minmax=wfunc(wlim))
                else:
                    templ_spl = gaussian_templates(templ_wave, 
                                                       centers=wi, widths=dw)

                #templ_spl = np.hstack([np.ones_like(templ_wave), templ_spl])
                templ_corr = (templ_spl.dot(coeffs[self.NFILT:]) + 1) * zp_ref
                templ_corr[templ.wave < wlim[0]] = 1.
                templ_corr[templ.wave > wlim[1]] = 1.

                #templ_tweak[(templ.wave < xmin) | (templ.wave > xmax)] = 1
                templ.flux *= templ_corr
                #templ.flux_fnu /= templ_tweak

            # Recompute filter fluxes from tweaked templates    
            self.tempfilt = TemplateGrid(self.zgrid, self.templates, RES=self.param['FILTERS_RES'], f_numbers=self.f_numbers, add_igm=self.param['IGM_SCALE_TAU'], galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'], n_proc=0, cosmology=self.cosmology, array_dtype=self.ARRAY_DTYPE)

        return fig


    def write_zeropoint_file(self, file='zphot.zeropoint.x'):
        fp = open(file,'w')
        for i in range(self.NFILT):
            fp.write('F{0:<3d}  {1:.6f}  # {2}\n'.format(self.f_numbers[i], self.zp[i], self.flux_columns[i]))
        
        fp.close()
    
    def full_sed(self, z, coeffs_i):
        import astropy.units as u
        global IGM_OBJECT
        
        templ = self.templates[0]
        tempflux = np.zeros((self.NTEMP, templ.wave.shape[0]),
                            dtype=self.ARRAY_DTYPE)
        for i in range(self.NTEMP):
            iz = self.templates[i].zindex(z=z, redshift_type='nearest')
            tempflux[i, :] = self.templates[i].flux_fnu(iz)
            
        templz = templ.wave*(1+z)
        
        if self.tempfilt.add_igm:
            igmz = templ.wave*0.+1
            lyman = templ.wave < 1300
            igmz[lyman] = IGM_OBJECT.full_IGM(z, templz[lyman])
        else:
            igmz = 1.
        
        templf = np.dot(coeffs_i, tempflux)*igmz
        return templz, templf
                
    def show_fit(self, id, show_fnu=False, xlim=[0.3, 9], get_spec=False, id_is_idx=False, show_components=False, show_redshift_draws=False, draws_cmap=None, zshow=None, ds9=None, ds9_sky=True, add_label=True, showpz=0.6, logpz=False, zr=None, axes=None, template_color='#1f77b4', figsize=[8,4], NDRAW=100, fitter='nnls', show_missing=True, maglim=None, show_prior=False, show_stars=False, delta_chi2_stars=0, show_upperlimits=True, snr_thresh=2.):
        """
        Make plot of SED and p(z) of a single object
        
        Parameters
        ----------
        id : int
            Object ID corresponding to columns in `self.cat['id']`.  Or if
            `id_is_idx` is set to True, then is zero-index of the desired 
            object in the catalog array.
        
        show_fnu : bool, int
            If False, then make plots in f-lambda units of 1e-19 erg/s/cm2/A.
            
            If `show_fnu == 1`, then plot f-nu units of uJy
            
            If `show_fnu == 2`, then plot "nu-Fnu" units of uJy/micron.
        
        xlim : list
            Wavelength limits to plot
        
        get_spec : bool
            If True, return the SED data rather than make a plot
        
        id_is_idx : bool
            See `id`.
        
        show_components : bool
            Show all of the individual SED components, along with their 
            combination.
        
        show_redshift_draws: bool
            Show templates at different redshifts drawn from the PDF
        
        draws_cmap: color map
            Color map for `show_reshift_draws=True`, defaults to 
            `~matplotlib.pyplot.cm.rainbow`.
            
        zshow : None, float
            If a value is supplied, compute the best-fit SED at this redshift,
            rather than the value in the `self.zbest` array.
        
        showpz : bool, float
            Include p(z) panel.  If a float, then scale the p(z) panel by 
            a factor of `showpz` relative to half of the full plot width.
        
        zr : None or [z0, z1]
            Range of redshifts to show in p(z) panel.  If None, then show
            the full range in `self.zgrid`.
        
        axes : None or list
            If provided, draw the SED and p(z) panels into the provided axes.
            If just one axis is provided, then just plot the SED.

        show_upper_limits: bool
            If False, the upper limit errorbar measurements will not be shown.

        snr_thresh: float
            Sets the threshold in SNR required for a detection.
            Default is 2.
            
        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object
        
        data : dict
            Dictionary of fit data (photometry, best-fit template, etc.).
            
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from scipy.integrate import cumtrapz
        
        import astropy.units as u
        from cycler import cycler
        
        global IGM_OBJECT
        
        if False:
            ids = self.cat['id'][(self.zbest > 1.7)]
            j = -1
            j+=1; id = ids[j]; self.show_fit(id, show_fnu=show_fnu)
            
        if id_is_idx:
            ix = id
            z = self.zbest[ix]
        else:
            ix = self.idx[self.cat['id'] == id][0]
            z = self.zbest[ix]
        
        if zshow is not None:
            z = zshow
        
        if ds9 is not None:
            pan = ds9.get('pan fk5')
            if pan == '0 0':
                ds9_sky = False
                
            if ds9_sky:
                #for c in ['ra','RA','x_world']:
                ds9.set('pan to {0} {1} fk5'.format(self.cat['x_world'][ix], self.cat['y_world'][ix]))
            else:
                ds9.set('pan to {0} {1}'.format(self.cat['x_image'][ix], self.cat['y_image'][ix]))
                
        ## SED
        fnu_i = np.squeeze(self.fnu[ix, :])*self.ext_redden*self.zp
        efnu_i = np.squeeze(self.efnu[ix,:])*self.ext_redden*self.zp
        ok_band = (fnu_i/self.zp > self.param['NOT_OBS_THRESHOLD']) 
        ok_band &= (efnu_i/self.zp > 0)
        efnu_i[~ok_band] = self.param['NOT_OBS_THRESHOLD'] - 9.
        
        ## Evaluate coeffs at specified redshift
        tef_i = self.TEF(z)
        A = np.squeeze(self.tempfilt(z))
        chi2_i, coeffs_i, fmodel, draws = _fit_obj(fnu_i, efnu_i, A, 
                                                   tef_i, self.zp, 
                                                   NDRAW, fitter)
        if draws is None:
            efmodel = 0
        else:
            efmodel = np.percentile(np.dot(draws, A), [16,84], axis=0)
            efmodel = np.squeeze(np.diff(efmodel, axis=0)/2.)
            
        #templz, templf = self.full_sed(self.zbest[ix][0], coeffs_i)
        if True:
            templ = self.templates[0]
            tempflux = np.zeros((self.NTEMP, templ.wave.shape[0]),
                                dtype=self.ARRAY_DTYPE)
            for i in range(self.NTEMP):
                iz = self.templates[i].zindex(z=z, redshift_type='nearest')
                try:
                    tempflux[i, :] = self.templates[i].flux_fnu(iz)#[iz,:]
                except:
                    tempflux[i, :] = np.interp(templ.wave, self.templates[i].wave, self.templates[i].flux_fnu(iz)) #[iz,:])
                    
            templz = templ.wave*(1+z)

            if self.tempfilt.add_igm:
                igmz = templ.wave*0.+1
                lyman = templ.wave < 1300
                igmz[lyman] = IGM_OBJECT.full_IGM(z, templz[lyman])
            else:
                igmz = 1.

            templf = np.dot(coeffs_i, tempflux)*igmz
            if draws is not None:
                templf_draws = np.dot(draws, tempflux)*igmz
                
        fnu_factor = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
        
        if show_fnu:
            if show_fnu == 2:
                templz_power = -1
                flam_spec = 1.e29/(templz/1.e4)
                flam_sed = 1.e29/self.ext_corr/(self.lc/1.e4)
                ylabel = (r'$f_\nu / \lambda$ [$\mu$Jy / $\mu$m]')
                flux_unit = u.uJy / u.micron
            else:
                templz_power = 0
                flam_spec = 1.e29
                flam_sed = 1.e29
                ylabel = (r'$f_\nu$ [$\mu$Jy]')    
                flux_unit = u.uJy
            
        else:
            templz_power = -2
            flam_spec = utils.CLIGHT*1.e10/templz**2/1.e-19
            flam_sed = utils.CLIGHT*1.e10/self.lc**2/self.ext_corr/1.e-19
            ylabel = (r'$f_\lambda [10^{-19}$ erg/s/cm$^2$]')
            
            flux_unit = 1.e-19*u.erg/u.s/u.cm**2/u.AA
                        
        try:
            data = OrderedDict(ix=ix, id=self.cat['id'][ix], z=z,
                           lc=self.lc, 
                           model=fmodel*fnu_factor*flam_sed,
                           emodel=efmodel*fnu_factor*flam_sed,
                           fobs=fnu_i*fnu_factor*flam_sed, 
                           efobs=efnu_i*fnu_factor*flam_sed,
                           templz=templz,
                           templf=templf*fnu_factor*flam_spec,
                           unit=show_fnu*1,
                           flux_unit=flux_unit,
                           wave_unit=u.AA)
        except:
            data = None
        
        ## Just return the data    
        if get_spec:            
            return data
        
        ###### Make the plot
        
        if axes is None:
            fig = plt.figure(figsize=figsize)
            if showpz:
                fig_axes = GridSpec(1,2,width_ratios=[1,showpz])
            else:    
                fig_axes = GridSpec(1,1,width_ratios=[1])
                
            ax = fig.add_subplot(fig_axes[0])
        else:
            fig = None
            fig_axes = None
            ax = axes[0]
                        
        ax.scatter(self.lc/1.e4, fmodel*fnu_factor*flam_sed, 
                   color='w', label=None, zorder=1, s=120, marker='o')
        
        ax.scatter(self.lc/1.e4, fmodel*fnu_factor*flam_sed, marker='o',
                  color=template_color, label=None, zorder=2, s=50, 
                  alpha=0.8)

        if draws is not None:
            ax.errorbar(self.lc/1.e4, fmodel*fnu_factor*flam_sed,
                        efmodel*fnu_factor*flam_sed, alpha=0.8,
                        color=template_color, zorder=2,
                        marker='None', linestyle='None', label=None)
        
        # Missing data
        missing = (fnu_i < self.param.params['NOT_OBS_THRESHOLD']) 
        missing |= (efnu_i < 0)
        
        # Detection
        sn2_detection = (~missing) & (fnu_i/efnu_i > snr_thresh)
        
        # S/N < 2
        sn2_not = (~missing) & (fnu_i/efnu_i <= snr_thresh)
        
        ax.errorbar(self.lc[sn2_detection]/1.e4, 
                    (fnu_i*fnu_factor*flam_sed)[sn2_detection], 
                    (efnu_i*fnu_factor*flam_sed)[sn2_detection], 
                    color='k', marker='s', linestyle='None', label=None, 
                    zorder=10)

        if show_upperlimits:
            ax.errorbar(self.lc[sn2_not]/1.e4, 
                        (fnu_i*fnu_factor*flam_sed)[sn2_not], 
                        (efnu_i*fnu_factor*flam_sed)[sn2_not], color='k', 
                        marker='s', alpha=0.4, linestyle='None', label=None)

        if show_missing:
            ax.errorbar(self.lc[missing]/1.e4, 
                        (fnu_i*fnu_factor*flam_sed)[missing]*0, 
                        (efnu_i*fnu_factor*flam_sed)[missing], 
                        color='0.7', marker='x', linestyle='None', 
                        alpha=0.4, label=None)
        
        pl = ax.plot(templz/1.e4, templf*fnu_factor*flam_spec, alpha=0.5, 
                     zorder=-1, color=template_color, 
                     label='z={0:.2f}'.format(z))
        
        if show_components:
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for i in range(self.NTEMP):
                if coeffs_i[i] != 0:
                    pi = ax.plot(templz/1.e4, 
                        coeffs_i[i]*tempflux[i,:]*igmz*fnu_factor*flam_spec, 
                              alpha=0.5, zorder=-1, 
                              label=self.templates[i].name.split('.dat')[0], 
                              color=colors[i % len(colors)])
                            
        elif show_redshift_draws:
            
            if draws_cmap is None:
                draws_cmap = plt.cm.rainbow
                
            # Draw random values from p(z)
            pz = np.exp(self.lnp[ix,:]).flatten()
            pzcum = cumtrapz(pz, x=self.zgrid)
            
            if show_redshift_draws == 1:
                nzdraw = 100
            else:
                nzdraw = show_redshift_draws*1
            
            rvs = np.random.rand(nzdraw)
            zdraws = np.interp(rvs, pzcum, self.zgrid[1:])
            
            for zi in zdraws:
                Az = np.squeeze(self.tempfilt(zi))
                chi2_zi, coeffs_zi, fmodelz, __ = _fit_obj(fnu_i, efnu_i, Az, 
                                                       self.TEF(zi), self.zp, 
                                                       0, fitter)
                c_i = np.interp(zi, self.zgrid, np.arange(self.NZ)/self.NZ)
                
                templzi = templ.wave*(1+zi)
                if self.tempfilt.add_igm:
                    igmz = templ.wave*0.+1
                    lyman = templ.wave < 1300
                    igmz[lyman] = IGM_OBJECT.full_IGM(zi, templzi[lyman])
                else:
                    igmz = 1.

                templfz = np.dot(coeffs_zi, tempflux)*igmz                
                templfz *=  flam_spec * (templz / templzi)**templz_power
                
                plz = ax.plot(templzi/1.e4, templfz*fnu_factor,
                             alpha=np.maximum(0.1, 1./nzdraw), 
                             zorder=-1, color=draws_cmap(c_i))
                
        if draws is not None:
            templf_width = np.percentile(templf_draws*fnu_factor*flam_spec, 
                                         [16,84], axis=0)
            ax.fill_between(templz/1.e4, templf_width[0,:], templf_width[1,:], 
                            color=pl[0].get_color(), alpha=0.1, label=None)
                                                       
        if show_stars & (not hasattr(self, 'star_chi2')):
            print('`star_chi2` attribute not found, run `fit_phoenix_stars`.')
            
        elif show_stars & hasattr(self, 'star_chi2'):
            # if __name__ == '__main__':
            #     # debug
            #     ix = _[1]['ix']
            #     chi2_i = self.chi2_noprior[ix]  
            #     ax = _[0].axes[0]
                
            delta_chi2 = self.star_chi2[ix,:] - chi2_i    
            good_stars = delta_chi2 < delta_chi2_stars
            good_stars &= (self.star_chi2[ix,:] - self.star_chi2[ix,:].min() < 100)
            
            if good_stars.sum() == 0:
                msg = 'Min delta_chi2 = {0:.1f} ({1})'
                sname = self.star_templates[np.argmin(delta_chi2)].name
                print(msg.format(delta_chi2.min(), sname))
                
            else:
                # dummy for cycler
                ax.plot(np.inf, np.inf)
                star_models  = self.star_tnorm[ix,:] * self.star_flux
                so = np.argsort(self.lc)
                order = np.where(good_stars)[0]
                order = order[np.argsort(delta_chi2[order])]
            
                for si in order:
                    label = self.star_templates[si].name.strip('bt-settl_')
                    label = '{0} {1:5.1f}'.format(label.replace('_', ' '),
                                                 delta_chi2[si])
                    print(label)
                    ax.plot(self.lc[so]/1.e4,
                            (star_models[:,si]*fnu_factor*flam_sed)[so], 
                            marker='o', alpha=0.5, label=label)

                if __name__ == '__main__':
                    ax.legend()
                
        if axes is None:            
            ax.set_ylabel(ylabel)
            
            if sn2_detection.sum() > 0:
                ymax = (fmodel*fnu_factor*flam_sed)[sn2_detection].max()
            else:
                ymax = (fmodel*fnu_factor*flam_sed).max()
                        
            if np.isfinite(ymax):
                ax.set_ylim(-0.1*ymax, 1.2*ymax)

            ax.set_xlim(xlim)
            xt = np.array([0.1, 0.5, 1, 2, 4, 8, 24, 160, 500])*1.e4

            ax.semilogx()

            valid_ticks = (xt > xlim[0]*1.e4) & (xt < xlim[1]*1.e4)
            if valid_ticks.sum() > 0:
                xt = xt[valid_ticks]
                ax.set_xticks(xt/1.e4)
                ax.set_xticklabels(xt/1.e4)

            ax.set_xlabel(r'$\lambda_\mathrm{obs}$')
            ax.grid()
            
            if add_label:
                txt = '{0}\nID={1}, mag={2:.1f}'
                txt = txt.format(self.param.params['MAIN_OUTPUT_FILE'], 
                                 self.cat['id'][ix], self.prior_mag_cat[ix])
                                 
                ax.text(0.95, 0.95, txt, ha='right', va='top', fontsize=7,
                        transform=ax.transAxes, 
                        bbox=dict(facecolor='w', alpha=0.5), zorder=10)
                
                ax.legend(fontsize=7, loc='upper left')
        
        # Optional mag scaling if show_fnu = 1 for uJy
        if (maglim is not None) & (show_fnu == 1):
            
            ax.semilogy()
            # Limits
            ax.scatter(self.lc[sn2_not]/1.e4, ((3*efnu_i)*fnu_factor*flam_sed)[sn2_not], color='k', marker='v', alpha=0.4, label=None)
            
            # Mag axes
            axm = ax.twinx()
            ax.set_ylim(10**(-0.4*(np.array(maglim)-23.9)))
            axm.set_ylim(0,1)
            ytv = np.arange(maglim[0], maglim[1], -1, dtype=int)
            axm.set_yticks(np.interp(ytv, maglim[::-1], [1,0]))
            axm.set_yticklabels(ytv)
                        
        ## P(z)
        if not showpz:
            return fig, data
            
        if axes is not None:
            if len(axes) == 1:
                return fig, data
            else:
                ax = axes[1]
        else:
            ax = fig.add_subplot(fig_axes[1])
        
        chi2 = np.squeeze(self.chi2_fit[ix,:])
        prior = np.exp(self.full_logprior[ix,:].flatten())
        #pz = np.exp(-(chi2-chi2.min())/2.)*prior
        #pz /= np.trapz(pz, self.zgrid)
        pz = np.exp(self.lnp[ix,:]).flatten()
        
        ax.plot(self.zgrid, pz, color='orange', label=None)
        if show_prior:
            ax.plot(self.zgrid, prior/prior.max()*pz.max(), color='g',
                label='prior')
        
        ax.fill_between(self.zgrid, pz, pz*0, color='yellow', alpha=0.5, 
                        label=None)
        if self.cat['z_spec'][ix] > 0:
            ax.vlines(self.cat['z_spec'][ix], 1.e-5, pz.max()*1.05, color='r',
                      label='zsp={0:.3f}'.format(self.cat['z_spec'][ix]))
        
        if zshow is not None:
            ax.vlines(zshow, 1.e-5, pz.max()*1.05, color='purple', 
                      label='z={0:.3f}'.format(zshow))
            
        if axes is None:
            ax.set_ylim(0,pz.max()*1.05)
            
            if logpz:
                ax.semilogy()
                ymax = np.minimum(ax.get_ylim()[1], 100)
                ax.set_ylim(1.e-3*ymax, 1.8*ymax)
                
            if zr is None:
                ax.set_xlim(0,self.zgrid[-1])
            else:
                ax.set_xlim(zr)
                
            ax.set_xlabel('z'); ax.set_ylabel('p(z)')
            ax.grid()
            ax.set_yticklabels([])
            
            fig_axes.tight_layout(fig, pad=0.5)
            
            if add_label & (self.cat['z_spec'][ix] > 0):
                ax.legend(fontsize=7, loc='upper left')
                
            return fig, data
        else:
            return fig, data


    def observed_frame_fluxes(self, f_numbers=[325], filters=None, verbose=True, n_proc=-1, percentiles=[2.5,16,50,84,97.5]):
        """
        Observed-frame fluxes in additional (e.g., unobserved) filters
        
        Parameters
        ----------
        f_numbers: list
            Unit-index of filters specified in``FILTER_FILE``.
        
        filters: list, optional
            Manually-specified `~eazy.filtesr.FilterDefinition` objects.
            If specified, then supercedes `f_numbers`.
        
        n_proc: int
            Number of processors passed to `~eazy.photoz.TemplateGrid`.
        
        percentiles: list or None
            If specified, compute percentiles of the template fluxes based 
            on the random template coefficient draws in ``self.coeffs_draws``.
        
        
        Returns
        -------
        tab: `~astropy.table.Table`
            Table of the observed-frame flux densities with metadata 
            describing the filters.
            
        """        
        from astropy.table import Table
        
        if verbose:
            if filters is None:
                msg = 'Observed-frame f_numbers: {0}'
                print(msg.format(f_numbers))
            else:
                fnames = '\n'.join([f'{i:>4} {f.name}'
                                for i, f in enumerate(filters)])
                print('Observed-frame filters:\n~~~~~~~~~~~~~~~~~~~ ')
                print(fnames)
                
        _tempfilt = TemplateGrid(self.zgrid, self.templates, 
                                   RES=self.param['FILTERS_RES'], 
                                   f_numbers=np.array(f_numbers), 
                                   add_igm=self.param['IGM_SCALE_TAU'],
                                   galactic_ebv=self.param['MW_EBV'], 
                                   Eb=self.param['SCALE_2175_BUMP'], 
                                   n_proc=n_proc, verbose=verbose, 
                                   cosmology=self.cosmology,
                                   array_dtype=self.ARRAY_DTYPE, 
                                   filters=filters)
        
        if filters is None:
            NOBS = len(f_numbers)            
        else:
            NOBS = len(filters)
            f_numbers = [i for i, f in enumerate(filters)]

            
        #izbest = np.argmax(self.pz, axis=1)
        izbest = self.izbest*1
                       
        templ_fluxes = _tempfilt.tempfilt[izbest, :, :]
        
        if percentiles is not None:
            draws_resh = np.transpose(self.coeffs_draws, axes=(1,0,2))
                                 
        tab = Table()
        for i in range(NOBS):
            flux_i = (self.coeffs_best*templ_fluxes[:,:,i]).sum(axis=1) 
            
            tab['obs{0}'.format(f_numbers[i])] = flux_i

            if percentiles is not None:
                draws_i = (draws_resh*templ_fluxes[:,:,i]).sum(axis=2) 
                perc = np.percentile(draws_i, percentiles, axis=0)
                tab['obs{0}_p'.format(f_numbers[i])] = perc.T
                
            key = 'name{0}'.format(f_numbers[i])
            tab.meta[key] = _tempfilt.filter_names[i]
            key = 'pivw{0}'.format(f_numbers[i])
            tab.meta[key] = _tempfilt.lc[i]
        
        return tab


    def rest_frame_fluxes(self, f_numbers=DEFAULT_UBVJ_FILTERS, pad_width=0.5, max_err=0.5, percentiles=[2.5,16,50,84,97.5], simple=False, verbose=1, fitter='nnls', n_proc=-1, par_skip=10000):
        """
        Rest-frame fluxes, refit by down-weighting bands far away from 
        the desired RF band.
        
        Parameters
        ==========
        f_numbers : list
            List of either unit-indices of filters in ``self.RES`` read 
            from FILTER_FILE or `~eazy.filters.FilterDefinition` objects.
        
        pad_width : float
            Padding around rest-frame wavelength to down-weight observed 
            filters.
            
        max_err : float
            Increased uncertainty outside of ``pad_width``.
            
            The modified uncertainties are computed as follows:
            
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> pad_width = 0.5
            >>> max_err = 0.5
            >>> z = 1.5
            >>> # Observed-frame pivot wavelengths
            >>> lc_obs = np.array([10543.5, 12470.5, 13924.2, 15396.6,  \
                                   7692.3,  8056.9,  9032.7, 4318.8, \
                                   5920.8, 3353.6, 35569.3, 45020.3, \
                                   57450.3, 79157.5])       
            >>> lc_rest = 5500. # e.g., rest V
            >>> x = np.log(lc_rest/(lc_obs/(1+z)))
            >>> grow = np.exp(-x**2/2/np.log(1/(1+pad_width))**2)
            >>> TEFz = (2/(1+grow/grow.max())-1)*max_err
            >>> so = np.argsort(lc_obs)
            >>> _ = plt.plot(lc_obs[so], TEFz[so])
            >>> _ = plt.xlabel(r'$\lambda_\mathrm{rest}$')
            >>> _ = plt.ylabel('Fractional uncertainty')
            
            
        percentiles : list
            Percentiles to return of the computed rest-frame fluxes drawn 
            from the fits including the observed uncertainties
        
        simple : bool
            If ``True`` then just return the rest-frame fluxes of the currrent
            best fits rather than doing the filter reweighting.
        
        fitter : str
            Fitting method passed to `~eazy.photoz._fit_obj`.
            
        Returns
        =======
        rf_tempfilt : `~numpy.ndarray`
            Array of the integrated template fluxes with dimensions 
            ``[NZGRID,NTEMP,len(f_numbers)]``.
        
        lc_rest : `~numpy.ndarray`
            Rest-frame filter pivot wavelengths
        
        rf_fluxes : `~numpy.ndarray`
            Rest-frame fluxes, with dimensions 
            ``[NOBJ, len(f_numbers), len(percentiles)]``.
            
        """
        import multiprocessing as mp
        import time
                
        NREST = len(f_numbers)
        if isinstance(f_numbers[0], int):
            f_list = [self.RES[fn] for fn in f_numbers]
        else:
            f_list = [fn for fn in f_numbers]
        
        if verbose:
            fnames = '\n'.join([f'{i:>4} {f.name}'
                                for i, f in enumerate(f_list)])
            print('Rest-frame filters:\n~~~~~~~~~~~~~~~~~~~ ')
            print(fnames)
            
        rf_tempfilt = np.zeros((self.NZ, self.NTEMP, NREST), 
                       dtype=self.ARRAY_DTYPE)
        
        rf_lc = np.array([f_i.pivot for f_i in f_list])
        
        for i_t, templ in enumerate(self.templates):
            # Redshift dependent templates
            iz = np.maximum(templ.zindex(self.zgrid), 0)
            _rf = [templ.integrate_filter(f_list, z=0, iz=i) 
                     for i in range(templ.NZ)]
            
            rf_tempfilt[:, i_t, :] = np.array(_rf)[iz,:]
            
            
        # Grid index of best redshfit
        izbest = self.izbest*1

        f_rest = np.zeros((self.NOBJ, NREST, len(percentiles)),
                          dtype=self.ARRAY_DTYPE)
        f_rest += self.param['NOT_OBS_THRESHOLD'] - 9.
        
        if simple:
            # Don't refit reweighting filters and get straight 
            # template fluxes
            if verbose:
                print(' ... (simple=True) no filter reweighting')
                
            coeffsT = np.transpose(self.coeffs_draws, axes=(1,0,2))

            rf_iz = rf_tempfilt[izbest,:,:]

            for i in range(NREST):
                f_vals = (coeffsT*rf_iz[:,:,i]).sum(axis=2)
                f_rest[:,i,:] = np.percentile(f_vals, percentiles, axis=0).T
            
            return rf_tempfilt, rf_lc, f_rest  
                        
        fnu_corr = self.fnu*self.ext_redden*self.zp
        efnu_corr = self.efnu*self.ext_redden*self.zp
        efnu_corr[~self.ok_data] = self.param['NOT_OBS_THRESHOLD'] - 9.
                
        idx = self.idx[self.zbest > self.zgrid[0]]
        
        # Set seed
        np.random.seed(self.random_seed)
        
        if (n_proc >= 0) & (len(idx) > par_skip):            
            if n_proc == 0:
                n_proc = mp.cpu_count() - 2
            
            skip = np.maximum(len(idx)//par_skip, 1)
            
            n_proc = np.minimum(n_proc, skip)
            
            t0 = time.time()

            pool = mp.Pool(processes=n_proc)
            results = [pool.apply_async(_fit_rest_group, 
                                          (idx[i::skip], 
                                           fnu_corr[idx[i::skip],:], 
                                           efnu_corr[idx[i::skip],:], 
                                           izbest[idx[i::skip]], 
                                           self.zbest[idx[i::skip]], 
                                           self.zp*1, True, 
                                           fitter, self.tempfilt,
                                           self.ARRAY_DTYPE, 
                                           rf_tempfilt, percentiles, 
                                           rf_lc, pad_width, max_err))
                        for i in range(skip)]

            pool.close()
            pool.join()

            for res in results:
                _ = res.get(timeout=1)
                _ix, _frest = _                
                f_rest[_ix,:,:] = _frest
            #
            t1 = time.time()
            print(f' ... rest-frame flux: {t1-t0:.1f} s (n_proc={n_proc}, '
                  f' NOBJ={len(idx)})')
            
        else:            
            for ix in idx:

                fnu_i = fnu_corr[ix,:]*1
                efnu_i = efnu_corr[ix,:]*1
                z = self.zbest[ix]
                if (z < 0) | (~np.isfinite(z)):
                    continue
            
                A = self.tempfilt(z)   
                iz = izbest[ix]
            
                for i in range(NREST):
                    ## Grow uncertainties away from RF band
                    #lc_i = rf_tempfilt.lc[i]
                    lc_i = rf_lc[i]
                
                    # Normal in log wavelength
                    x = np.log(lc_i/(self.lc/(1+z)))
                    grow = np.exp(-x**2/2/np.log(1/(1+pad_width))**2)

                    TEFz = (2/(1+grow/grow.max())-1)*max_err
            
                    _ = _fit_obj(fnu_i, efnu_i, A, TEFz, self.zp, 100, fitter)
                    chi2_i, coeffs_i, fmodel_i, draws = _
                
                    if draws is None:
                        f_rest[ix,i,:] = np.zeros(len(percentiles),
                                                  dtype=self.ARRAY_DTYPE) - 1
                    else:
                        dval = np.dot(draws, rf_tempfilt[iz,:,i])
                        f_rest[ix,i,:] = np.percentile(dval, percentiles,
                                                       axis=0)
                        del(dval)
                    
        return rf_tempfilt, rf_lc, f_rest  


    def compute_tef_lnp(self, in_place=True):
        """
        Uncertainty + TEF component of the log likelihood
        """
        try:
            import tqdm
            iters = tqdm.tqdm(enumerate(self.zgrid))
        except:
            iters = enumerate(self.zgrid)
            
        tef_lnp = np.zeros((self.NOBJ, self.NZ), dtype=self.ARRAY_DTYPE)
        for i, z in iters:
            TEFz = self.TEF(z)
            var = self.efnu**2 + (TEFz*self.fnu)**2
            tef_lnp[:,i] = -1/2.*(np.log(var)*self.ok_data).sum(axis=1)
        
        if in_place:
            self.tef_lnp = tef_lnp
            
        return tef_lnp


    def compute_lnp(self, prior=False, beta_prior=False, 
                    clip_wavelength=1100, in_place=True):
        """
        Compute log-likelihood from chi2, prior, and TEF terms
        
        Parameters
        ==========
        
        prior : bool
            Apply apparent magnitude prior
        
        beta_prior : bool
            Apply UV-slope beta prior
        
        clip_wavelength : float or None
            If specified, set pz = 0 at redshifts beyond where 
            `clip_wavelength*(1+z)` is greater than the reddest valid filter
            for a given object.
         
        Returns
        =======
        Updates ``lnp``, ``lnpmax`` attributes.
            
        """
        import time
        
        has_chi2 = (self.chi2_fit != 0).sum(axis=1) > 0 
        #min_chi2 = self.chi2_fit[has_chi2,:].min(axis=1)
        
        loglike = -self.chi2_fit[has_chi2,:]/2.
        #pz = np.exp(-(self.chi2_fit[has_chi2,:].T-min_chi2)/2.).T
        
        if self.param['VERBOSITY'] >= 2:
            print('compute_lnp ({0})'.format(time.ctime()))
            
        if hasattr(self, 'tef_lnp'):
            if self.param['VERBOSITY'] >= 2:
                print(' ... tef_lnp')
            
            loglike += self.tef_lnp[has_chi2,:]
            
        if prior:
            if self.param['VERBOSITY'] >= 2:
                print(' ... full_logprior')
            
            loglike += self.full_logprior[has_chi2,:]
        
        if clip_wavelength is not None:
            # Set pz=0 at redshifts where clip_wavelength beyond reddest 
            # filter
            clip_wz = clip_wavelength*(1+self.zgrid)
            red_mask = (clip_wz[:,None] > self.lc_reddest[None, has_chi2]).T
            
            loglike[red_mask] = -np.inf
            self.lc_zmax = self.lc_reddest/clip_wavelength - 1
            self.clip_wavelength = clip_wavelength
            
        if beta_prior:
            if self.param['VERBOSITY'] >= 2:
                print(' ... beta lnp_beta')
                
            p_beta = self.prior_beta(w1=1350, w2=1800, sample=has_chi2)
            self.lnp_beta[has_chi2,:] = np.log(p_beta)
            self.lnp_beta[~np.isfinite(self.lnp_beta)] = -np.inf
            loglike += self.lnp_beta[has_chi2,:]
        
        # Optional extra prior
        if hasattr(self, 'extra_lnp'):
            loglike += self.extra_lnp[has_chi2,:]
            
        loglike[~np.isfinite(loglike)] = -1e20
        
        lnpmax = loglike.max(axis=1)
        pz = np.exp(loglike.T - lnpmax).T
        log_norm = np.log(pz.dot(self.trdz))
        
        lnp = (loglike.T - lnpmax - log_norm).T
        #lnpmax = -log_norm
        
        lnp[~np.isfinite(lnp)] = -1e20
        
        if in_place:
            self.lnp[has_chi2,:] = lnp
            self.lnpmax[has_chi2] = -log_norm
        
            self.lnp_with_prior = prior
            self.lnp_with_beta_prior = beta_prior
        else:
            return has_chi2, lnp, -log_norm
            
        del(lnpmax)
        del(pz)
        del(log_norm)
        del(loglike)
        del(lnp)


    def get_maxlnp_redshift(self, prior=False, beta_prior=False, clip_wavelength=1100):
        """Fit parabola to ``lnp`` to get continuous max(lnp) redshift.
        
        Returns
        -------
        zml: ``NOBJ`` array
            Redshift where lnp is maximized
        
        maxlnp: ``NOBJ`` array
            Maximum of lnp
            
        """
        #from scipy import polyfit, polyval
        from numpy import polyfit, polyval
        
        self.compute_lnp(prior=prior, beta_prior=beta_prior, 
                         clip_wavelength=clip_wavelength)
                         
        # Objects that have been fit
        has_chi2 = (self.chi2_fit != 0).sum(axis=1) > 0 
                                   
        #izbest0 = np.argmin(self.chi2_fit, axis=1)
        izmax = np.argmax(self.lnp, axis=1)*has_chi2
        
        zbest = self.zgrid[izmax]
        lnpmax = np.zeros_like(zbest)
        
        zbest[izmax == 0] = -1
        
        mask = (izmax > 0) & (izmax < self.NZ-1) & has_chi2

        #####
        # Analytic parabola fit
        iz_ = izmax[self.idx[mask]]
        
        _x = np.array([self.zgrid[iz-1:iz+2] for iz in iz_])
        _y = np.array([self.lnp[iobj, iz-1:iz+2] 
                       for iz, iobj in zip(iz_, self.idx[mask])])

        dx = np.diff(_x, axis=1).T
        dx2 = np.diff(_x**2, axis=1).T
        dy = np.diff(_y, axis=1).T

        c2 = (dy[1]/dx[1] - dy[0]/dx[0]) / (dx2[1]/dx[1] - dx2[0]/dx[0])
        c1 = (dy[0] - c2 * dx2[0])/dx[0]
        c0 = _y.T[0] - c1*_x.T[0] - c2*_x.T[0]**2
        
        _m = self.idx[mask]
        zbest[_m] = -c1/2/c2
        lnpmax[_m] = c2*zbest[_m]**2+c1*zbest[_m]+c0
        
        del(_x)
        del(_y)
        del(iz_)
        del(dx)
        del(dx2)
        del(dy)
        del(c2)
        del(c1)
        del(c0)
        del(_m)
        
        return zbest, lnpmax


    @property
    def izbest(self):
        """
        index of nearest ``zgrid`` value to ``zbest``.
        """   
        iz = np.argmin(np.abs(self.zgrid[:,None]-self.zbest[None,:]), axis=0)
        return iz


    def lcz(self, zbest=None):
        """
        Redshifted filter wavelengths using ``zbest``.
        """
        if zbest is None:
            zbest = self.zbest
            
        _lcz = np.dot(1/(1+zbest[:, np.newaxis]), self.lc[np.newaxis,:])
        return _lcz


    @property
    def izchi2(self):
        """
        ``zgrid`` index where ``chi2_fit`` maximized
        """
        return np.argmin(self.chi2_fit, axis=1)


    @property 
    def zchi2(self):
        """
        Redshift at `~eazy.photoz.PhotoZ.izchi2` index.
        """
        return self.zgrid[self.izchi2]


    @property 
    def izml(self):
        """
        ``zgrid`` index where ``lnp`` maximized
        """    
        return np.argmax(self.lnp)


    def compute_full_risk(self):
        """
        Full "risk" profile from Tanaka et al. 2017
        """
        zsq = np.dot(self.zgrid[:,None], np.ones_like(self.zgrid)[None,:])
        L = self._loss((zsq-self.zgrid)/(1+self.zgrid))
        
        Rz = self.lnp*0.
        
        has_chi2 = (self.chi2_fit != 0).sum(axis=1) > 0 
        hasz = self.zbest > self.zgrid[0]
        idx = self.idx[hasz & (has_chi2)]
        
        for i in idx:
            Rz[i,:] = np.dot(np.exp(self.lnp[i,:])*L, self.trdz)
        
        del(zsq)
        del(L)
        del(has_chi2)
        del(hasz)
        del(idx)    
        return Rz
        
        #self.full_risk = Rz
        #self.min_risk = self.zgrid[np.argmin(Rz, axis=1)]


    def compute_best_risk(self):
        """
        "Risk" function from Tanaka et al. 2017
        """
        has_chi2 = (self.chi2_fit != 0).sum(axis=1) > 0 
        mask = (has_chi2) & (self.zbest > self.zgrid[0])
        
        zbest_grid = np.dot(self.zbest[mask, None],
                            np.ones_like(self.zgrid)[None,:])
        L = self._loss((zbest_grid-self.zgrid)/(1+self.zgrid))
        #dz = np.gradient(self.zgrid)
        
        zbest_risk = np.zeros(self.NOBJ, dtype=self.ARRAY_DTYPE)-1
        zbest_risk[mask] = np.dot(np.exp(self.lnp[mask,:])*L, self.trdz)
        
        del(has_chi2)
        del(mask)
        del(zbest_grid)
        del(L)
        
        return zbest_risk


    @staticmethod    
    def _loss(dz, gamma=0.15):
        """
        Loss for risk function
        """
        return 1-1/(1+(dz/gamma)**2)


    def PIT(self, zspec):
        """
        PIT function for evaluating the calibration of p(z), 
        as described in Tanaka (2017).
        """
        zspec_grid = np.dot(zspec[:,None], np.ones_like(self.zgrid)[None,:])
        zlim = zspec_grid >= self.zgrid
        #dz = np.gradient(self.zgrid)
        PIT = np.dot(np.exp(self.lnp)*zlim, self.trdz)
        del(zspec_grid)
        
        return PIT


    def pz_percentiles(self, percentiles=[2.5,16,50,84,97.5], oversample=5,
                       selection=None):
        """
        Compute percentiles of the final PDF(z)
        """
        import scipy.interpolate 
        from scipy.integrate import cumtrapz
        
        interpolator = scipy.interpolate.Akima1DInterpolator
        
        p100 = np.array(percentiles)/100.
        zlimits = np.zeros((self.NOBJ, p100.size), dtype=self.ARRAY_DTYPE)
            
        zr = [self.param['Z_MIN'], self.param['Z_MAX']]
        zgrid_zoom = utils.log_zgrid(zr=zr,dz=self.param['Z_STEP']/oversample)
                
        ok = self.zbest > self.zgrid[0]      
        if selection is not None:
            ok &= selection
        
        if ok.sum() == 0:
            print('pz_percentiles: No objects in selection')
            return zlimits
            
        spl = interpolator(self.zgrid, self.lnp[ok,:], axis=1)
        pzcum = cumtrapz(np.exp(spl(zgrid_zoom)), x=zgrid_zoom, axis=1)

        # Akima1DInterpolator can get some NaNs at the end?
        valid = np.isfinite(pzcum)
        pzcum[~valid] = 0.
        pzcmax = pzcum.max(axis=1)
        pzcum = (pzcum.T / pzcmax).T
        pzcum[~valid] = 1.
        
        #pzcum /= pzcum[-1]
        #pzcum = cumtrapz(self.pz[ok,:], x=self.zgrid, axis=1)
        
        for j, i in enumerate(self.idx[ok]):
            zlimits[i,:] = np.interp(p100, pzcum[j, :], zgrid_zoom[1:])
        
        del(pzcum)
        del(p100)
        del(spl)
        del(zgrid_zoom)
        del(valid)
        del(pzcmax)
        
        return zlimits
    
    
    def cdf_percentiles(self, cdf_sigmas=CDF_SIGMAS, **kwargs):
        """
        Redshifts of PDF percentiles in terms of ``\sigma`` for a normal
        distribution, useful for compressing the PDF.
        """
        import scipy.stats
        cdf_percentiles = scipy.stats.norm.cdf(cdf_sigmas)*100
        zlimits = self.pz_percentiles(percentiles=cdf_percentiles, **kwargs)
        return zlimits


    def find_peaks(self, thres=0.8, min_dist_dz=0.1):
        """
        Find discrete peaks in ``lnp`` with `~peakutils`.
        
        Parameters
        ----------
        thres: float
            Threshold passed to `~peakutils.indexes`.
        
        min_dist_dz: float
            Peak separation in units of dz*(1+z)
            
        """
        import peakutils
        
        ok = self.zbest > self.zgrid[0]      

        peaks = [0]*self.NOBJ
        numpeaks = np.zeros(self.NOBJ, dtype=int)
        
        min_dist = int(min_dist_dz/self.param['Z_STEP'])
        
        for j, i in enumerate(self.idx[ok]):
            indices = peakutils.indexes(np.exp(self.lnp[i,:]), thres=thres,
                                       min_dist=min_dist)
            peaks[i] = indices
            numpeaks[i] = len(indices)
        
        return peaks, numpeaks


    def abs_mag(self, f_numbers=[271, 272, 274], cosmology=None, rest_kwargs={'percentiles':[2.5,16,50,84,97.5], 'pad_width':0.5, 'max_err':0.5, 'verbose':False, 'simple':False}):
        """
        Get absolute mags (e.g., UV).
        
        Parameters
        ==========
        f_numbers : list
            List of either unit-indices of filters in ``self.RES`` read 
            from FILTER_FILE or `~eazy.filters.FilterDefinition` objects.
            
        cosmology : `~astropy.cosmology` object
            If ``None``, default to ``self.cosmology``.
        
        rest_kwargs : dict
            Arguments passed to `~eazy.photoz.PhotoZ.rest_frame_fluxes`
        
        Returns
        =======
        tab : `~astropy.table.Table`
           Table with rest-frame luminosities.  `tab.meta` includes the filter
           information.
           
        """    
        if cosmology is None:
            #from astropy.cosmology import WMAP9 as cosmology
            cosmology = self.cosmology
            
        _rf = self.rest_frame_fluxes(f_numbers=f_numbers, **rest_kwargs) 
        rf_tf, rf_lc, rf = _rf
        
        zdm = self.zgrid #utils.log_zgrid([0.01, 13], 0.01)
        dm = cosmology.distmod(zdm).value - 2.5*np.log10(1+zdm)
        DM = np.interp(self.zbest, zdm, dm, left=0, right=0)
        
        #lc_round = ['{0}'.format(int(np.round(lc/100))*100) 
        #            for lc in rf_tf.lc]
        
        tab = Table()
        tab['DISTMOD'] = DM
        
        for i, fn in enumerate(f_numbers):
            tab.meta['MNAME{0}'.format(fn)] = (self.RES[fn].name, 
                                                     'Filter name')

            tab.meta['MWAVE{0}'.format(fn)] = (self.RES[fn].pivot, 
                                                 'Pivot wavelength, Angstrom')
                                                        
            obsm = self.param.params['PRIOR_ABZP'] - 2.5*np.log10(rf[:,i,:])
            tab['ABSM_{0}'.format(fn)] = (obsm.T - DM).T
        
        return tab


    def sps_parameters(self, UBVJ=DEFAULT_UBVJ_FILTERS, LIR_wave=[8,1000], cosmology=None, extra_rf_filters=DEFAULT_RF_FILTERS, rf_pad_width=0.5, rf_max_err=0.5, percentile_limits=[2.5, 16, 50, 84, 97.5], template_fnu_units=(1*u.solLum / u.Hz), simple=False):
        """
        Rest-frame colors and population parameters
        
        template_fnu_units: Units of templates when converted to ``flux_fnu``, 
                            e.g., L_sun / Hz for fsps templates.
        
        """        
        if cosmology is None:
            #from astropy.cosmology import WMAP9 as cosmology
            cosmology = self.cosmology
            
        _ubvj = self.rest_frame_fluxes(f_numbers=UBVJ, pad_width=rf_pad_width, max_err=rf_max_err, percentiles=[2.5,16,50,84,97.5], verbose=False, simple=simple) 
        self.ubvj_tempfilt, self.ubvj_lc, self.ubvj = _ubvj
        self.ubvj_f_numbers = UBVJ
        
        restU = self.ubvj[:,0,2]
        restB = self.ubvj[:,1,2]
        restV = self.ubvj[:,2,2]
        restJ = self.ubvj[:,3,2]

        errU = (self.ubvj[:,0,3] - self.ubvj[:,0,1])/2.
        errB = (self.ubvj[:,1,3] - self.ubvj[:,1,1])/2.
        errV = (self.ubvj[:,2,3] - self.ubvj[:,2,1])/2.
        errJ = (self.ubvj[:,3,3] - self.ubvj[:,3,1])/2.
                    
        template_params_file = self.param['TEMPLATES_FILE']+'.fits'
        if os.path.exists(template_params_file):
            tab_temp = Table.read(template_params_file)
            has_template_params = True
            if len(tab_temp) != self.NTEMP:
                NADD = self.NTEMP - len(tab_temp)
                msg = 'Warning: adding {0} empty rows to {1} to match NTEMP={2}'
                print(msg.format(NADD, template_params_file, self.NTEMP))
                for j in range(NADD):
                    tab_temp.add_row(vals=None)
        else:
            # Dummy
            msg = """
 Couldn't find template parameters file {0} for population synthesis 
 calculations.
            """
            print(msg.format(template_params_file))
            has_template_params = False
            tab_temp = Table()
            
            cols = ['Av', 'mass', 'Lv', 'sfr', 'formed_100', 'formed_total',
                    'LIR', 'energy_abs', 
                    'line_EW_Ha', 'line_C_Ha', 'line_flux_Ha', 
                    'line_EW_O3', 'line_C_O3', 'line_flux_O3', 
                    'line_EW_Hb', 'line_C_Hb', 'line_flux_Hb', 
                    'line_EW_O2', 'line_C_O2', 'line_flux_O2', 
                    'line_EW_Lya', 'line_C_Lya', 'line_flux_Lya']
                    
            for c in cols:
                tab_temp[c] = np.ones(self.NTEMP)*np.nan
                
        # Normalize fit coefficients to template V-band
        iz = np.argmin(np.abs(self.zgrid[:,None] - self.zbest[None,:]), 
                       axis=0)
        
        coeffs_norm = self.coeffs_best*self.ubvj_tempfilt[iz,:,2]        
        
        # Normalize fit coefficients to unity sum
        coeffs_norm = (coeffs_norm.T/coeffs_norm.sum(axis=1)).T

        # Convert observed maggies to fnu
        fnu_units = u.erg/u.s/u.cm**2/u.Hz
        uJy_to_cgs = u.microJansky.to(u.erg/u.s/u.cm**2/u.Hz)
        fnu_scl = 10**(-0.4*(self.param.params['PRIOR_ABZP']-23.9))*uJy_to_cgs
        
        dL = np.zeros(self.NOBJ, dtype=np.float64)*u.cm
        mask = self.zbest > 0
        dL[mask] = cosmology.luminosity_distance(self.zbest[mask]).to(u.cm)
        
        par_table = {}        
        
        if template_fnu_units is not None:
            to_physical = fnu_scl*fnu_units*4*np.pi*dL**2/(1+self.zbest)
            to_physical /= (1*template_fnu_units).to(u.erg/u.second/u.Hz)
        else:
            to_physical = None
            
        if self.get_err:
            par_draws_table = {}
        
            coeffs_draws = np.maximum(self.coeffs_draws, 0)
            if to_physical is not None:
                rest_draws = np.transpose((coeffs_draws.T*to_physical).T, 
                                          axes=(1,0,2))
                                     
                # Remove unit (which should be null)
                rest_draws = np.array(rest_draws)
                draws_norm = None
                
            else:
                #  Renorm in rest V band
                _draws = np.transpose(coeffs_draws, axes=(1,0,2)) 
                coeffs_draws = np.transpose(_draws*self.ubvj_tempfilt[iz,:,2],
                                            axes=(0,1,2))
                draws_norm = (coeffs_draws.T/coeffs_draws.sum(axis=2).T).T
        else:
            draws_norm = None
            coeffs_draws = None
                              
        ##### Use physical units
        if to_physical is not None:
            
            if self.param['VERBOSITY'] >= 2:
                print(f' ... Physical quantities directly from coeffs and templates ({template_fnu_units})')
            
            coeffs_rest = (self.coeffs_best.T*to_physical).T
            
            # Remove unit (which should be null)
            coeffs_rest = np.array(coeffs_rest)

            # mass = coeffs_rest.dot(tab_temp['mass'])*u.solMass
            # SFR = coeffs_rest.dot(tab_temp['sfr'])*u.solMass/u.yr
            # Lv = coeffs_rest.dot(tab_temp['Lv'])*u.solLum
            # LIR = coeffs_rest.dot(tab_temp['LIR'])*u.solLum
            # energy_abs = coeffs_rest.dot(tab_temp['energy_abs'])*u.solLum
            
            iz0 = np.zeros(mask.sum(), dtype=int)
            
            #par_table = {}
            
            table_units = {'mass':u.solMass, 'sfr':u.solMass/u.yr,
                           'Lv':u.solLum, 'LIR':u.solLum, 
                           'energy_abs':u.solLum, 
                           'Lu':u.solLum, 'Lj':u.solLum, 
                           'L1400':u.solLum, 'L2800':u.solLum, 
                           'lwAgeV':u.Gyr, 'lw_age_V':u.Gyr,
                           'LHa':u.solLum, 'LOIII':u.solLum, 
                           'LHb':u.solLum, 'LOII':u.solLum}
                        
            for par in ['mass', 'sfr', 'Lv', 'LIR', 'energy_abs', 
                        'Lu', 'Lj', 'L1400', 'L2800', 
                        'LHa', 'LOIII', 'LHb', 'LOII']:
                
                if par not in tab_temp.colnames:
                    #par_table[par] = np.zeros(self.NOBJ) - 99.
                    continue
                    
                temp_par = tab_temp[par]
                                
                if temp_par.ndim == 1:
                    par_value = coeffs_rest.dot(temp_par)
                    
                    if self.get_err:
                        par_draws = rest_draws.dot(temp_par)
                        
                else:
                    # Redshift-dependent parameters
                    temp_matrix = np.zeros_like(coeffs_rest)
                    zb = self.zbest*1
                    zb[~np.isfinite(zb)] = -1.
                    
                    for _i, templ in enumerate(self.templates):
                        if templ.NZ == 0:
                            iz = iz0
                        else:
                            iz = templ.zindex(zb)
                            
                        temp_matrix[:, _i] = temp_par[_i, iz]
                    
                    par_value = (coeffs_rest*temp_matrix).sum(axis=1)
                    if self.get_err:
                        par_draws = (rest_draws*temp_matrix).sum(axis=2)
                                
                par_table[par] = par_value
                if par in table_units:
                    par_table[par] *= table_units[par]
                elif hasattr(tab_temp[par], 'unit'):
                    if tab_temp[par].unit is not None:
                        par_table[par] *= tab_temp[par].unit
                
                if self.get_err:
                    par_draws_table[par] = par_draws
                    
            par_table['MLv'] = par_table['mass']/par_table['Lv']
            
            # Light-weighted (V) parameters
            for par in ['Av', 'dust1', 'dust2', 'lwAgeV', 'lw_Age_V']:
                
                if par not in tab_temp.colnames:
                    continue
                
                if par in ['Av', 'dust1', 'dust2']:
                    if par == 'Av':
                        Av_tau = 0.4*np.log(10)
                    else:
                        Av_tau = 1.
                        
                    temp_par = np.exp(tab_temp[par]*Av_tau)
                    is_dust = True
                else:
                    is_dust = False
                    temp_par = tab_temp[par]
                                
                if temp_par.ndim == 1:
                    par_value = coeffs_norm.dot(temp_par) 
                    if is_dust:
                        par_denom = coeffs_norm.dot(temp_par*0+1) 
                        par_value = np.log(par_value/par_denom) / Av_tau
                        
                else:
                    # Redshift-dependent parameters
                    temp_matrix = np.zeros_like(coeffs_rest)
                    zb = self.zbest*1
                    zb[~np.isfinite(zb)] = -1.
                
                    for _i, templ in enumerate(self.templates):
                        if templ.NZ == 0:
                            iz = iz0
                        else:
                            iz = templ.zindex(zb)
                            #pass
                            
                        temp_matrix[:, _i] = temp_par[_i, iz]
                    
                    par_value = (coeffs_norm*temp_matrix).sum(axis=1)
                    if is_dust:
                        temp_ones = np.ones_like(temp_matrix)
                        par_denom = (coeffs_norm*temp_ones).sum(axis=1)
                        par_value = np.log(par_value/par_denom) / Av_tau
                        
                par_table[par] = par_value
                if par in table_units:
                    par_table[par] *= table_units[par]
            
            if self.get_err:
                del(rest_draws)
        else:
            
            # Mass & SFR, normalize to V band and then scale by V luminosity
            #MLv = (coeffs_norm*temp_MLv).sum(axis=1)*u.solMass/u.solLum
            mass_norm = (coeffs_norm*tab_temp['mass']).sum(axis=1)*u.solMass
            Lv_norm = (coeffs_norm*tab_temp['Lv']).sum(axis=1)*u.solLum
            MLv = mass_norm / Lv_norm

            LIR_norm = (coeffs_norm*tab_temp['LIR']).sum(axis=1)*u.solLum
            LIRv = LIR_norm / Lv_norm

            # Absorbed energy 
            if 'energy_abs' in tab_temp.colnames:
                energy_abs_norm = (coeffs_norm*tab_temp['energy_abs']).sum(axis=1)*u.solLum
                energy_abs_v = energy_abs_norm / Lv_norm
            else:
                energy_abs_v = LIRv*0.

            # Comute LIR directly from templates as tab_temp['LIR']
            templ_LIR = np.zeros(self.NTEMP)
            for j in range(self.NTEMP):
                templ = self.templates[j]
                clip = (templ.wave > LIR_wave[0]*1e4) & (templ.wave < LIR_wave[1]*1e4)
                templ_LIR[j] = np.trapz(templ.flux[0,clip], templ.wave[clip])

            LIR_norm = (coeffs_norm*templ_LIR).sum(axis=1)*u.solLum
            LIRv = LIR_norm / Lv_norm

            SFR_norm = (coeffs_norm*tab_temp['sfr']).sum(axis=1)*u.solMass/u.yr
            SFRv = SFR_norm / Lv_norm
            
            fnu = restV*fnu_scl*(fnu_units)
            Lnu = fnu*4*np.pi*dL**2
            pivotV = self.ubvj_lc[2]*u.Angstrom*(1+self.zbest)
            nuV = (const.c/pivotV).to(u.Hz) 
            Lv = (nuV*Lnu).to(u.L_sun)

            # Av, compute based on linearized extinction corrections
            # NB: dust1 is tau, not Av, which differ by a factor of log(10)/2.5
            Av_tau = 0.4*np.log(10)
            if 'dust1' in tab_temp.colnames:
                tau_corr = np.exp(tab_temp['dust1'])
            elif 'dust2' in tab_temp.colnames:
                tau_corr = np.exp(tab_temp['dust2'])
            else:
                tau_corr = 1.

            # Force use Av if available
            if 'Av' in tab_temp.colnames:
                tau_corr = np.exp(tab_temp['Av']*Av_tau)

            tau_num = np.dot(coeffs_norm, tau_corr)
            tau_den = np.dot(coeffs_norm, tau_corr*0+1)
            tau_dust = np.log(tau_num/tau_den)
            Av = tau_dust / Av_tau

            if 'ageV' in tab_temp.colnames:
                age_norm = (coeffs_norm*tab_temp['ageV']).sum(axis=1)*u.Gyr
                lw_age_V = age_norm
            else:
                lw_age_V = -99*u.Gyr

            par_table['mass'] = MLv*Lv
            par_table['SFR'] = SFRv*Lv
            par_table['LIR'] = LIRv*Lv
            par_table['energy_abs'] = energy_abs_v*Lv
            par_table['Av'] = Av
            par_table['lw_age_V'] = lw_age_V
            par_table['MLv'] = MLv
        
        # Make the full table
        tab = Table()
        
        tab['restU'] = restU
        tab['restU_err'] = errU
        tab['restB'] = restB
        tab['restB_err'] = errB
        tab['restV'] = restV
        tab['restV_err'] = errV
        tab['restJ'] = restJ
        tab['restJ_err'] = errJ

        tab['dL'] = dL.to(u.Mpc)

        column_formats = {'dL':'.1e',
                         'mass':'.2e',
                         'sfr': '.3f',
                         'Lv':  '.2e',
                         'LIR': '.2e',
                         'energy_abs':  '.2e',
                         'Lu': '.2e',
                         'Lj': '.2e',
                         'L1400': '.2e',
                         'L2800': '.2e',
                         'lw_age_V':'.2f',
                         'lwAgeV':'.2f',
                         'MLv':'.2f',
                         'Av':'.2f',
                         'ssfr':'.2e',
                         'LHa':'.2e',
                         'LOIII':'.2e',
                         'LHb':'.2e',
                         'LOII':'.2e'}

        for col in par_table:
            tab[col] = par_table[col]
        
        for col in tab.colnames:
            if col in column_formats:
                tab[col].format = column_formats[col]
            else:
                tab[col].format = '.3f'
        
        if self.get_err:
            # Propagate coeff covariance to parameters
            if self.param['VERBOSITY'] >= 2:
                print(' ... Get uncertainties')
            
            if to_physical is None:
        
                massv_draws = (draws_norm*tab_temp['mass']).sum(axis=2)*u.solMass
                Lv_draws= (draws_norm*tab_temp['Lv']).sum(axis=2)*u.solLum
                LIR_draws = (draws_norm*templ_LIR).sum(axis=2)*u.solLum
                SFR_draws = (draws_norm*tab_temp['sfr']).sum(axis=2)*u.solMass/u.yr
                
                par_draws_table['Lv'] = Lv_draws
                par_draws_table['mass'] = ((massv_draws / Lv_draws).T*Lv).T
                par_draws_table['LIR'] = ((LIR_draws / Lv_draws).T*Lv).T
                par_draws_table['sfr'] = ((massv_draws / Lv_draws).T*Lv).T
            
            else:
                pass
                # Computed earlier
                    
            par_draws_table['ssfr'] = (par_draws_table['sfr'] / 
                                       par_draws_table['mass'])
                                                   
            for col in par_draws_table:
                pcol = col+'_p'
                #print('xxx', col, par_draws_table[col].shape)
                tab[pcol] = np.percentile(par_draws_table[col],
                                          percentile_limits, axis=0).T
                
                if col in column_formats:
                    tab[pcol].format = column_formats[col]
                else:
                    tab[pcol].format = '.3f'
        
        if 'sfr' in tab.colnames:
            tab['sfr'].description = 'SFR over last 100 Myr'
        
        if 'LIR' in tab.colnames:
            tab['LIR'].description = 'IR luminosity = energy_abs'

        for c in ['lw_Age_V', 'lwAgeV']:
            if c in tab.colnames:
                tab[c].description = 'Light-weighted age (V band)'

        for col in tab.colnames:
            bad = ~np.isfinite(tab[col])
            tab[col][bad] = -9e29
        
        for k in ['SYS_ERR', 'TEMP_ERR_FILE', 'TEMP_ERR_A2', 
                  'PRIOR_FILTER', 'PRIOR_ABZP', 
                  'IGM_SCALE_TAU', 'APPLY_IGM', 'TEMPLATES_FILE']:
            tab.meta[k] = self.param[k]
        
        for i, templ in enumerate(self.templates):
            tab.meta[f'TEMPL{i:03d}'] = templ.name
            
        tab.meta['ZBEST_USER'] = self.ZPHOT_USER
        tab.meta['ZBEST_AT_ZSPEC'] = self.ZPHOT_AT_ZSPEC
        tab.meta['ZML_WITH_PRIOR'] = self.ZML_WITH_PRIOR
        tab.meta['ZML_WITH_BETA_PRIOR'] = self.ZML_WITH_BETA_PRIOR
        tab.meta['RFSIMPLE'] = simple, 'RF fluxes without reweighting'
        
        for i in range(self.NFILT):
            f_i = self.f_numbers[i]
            c_i = self.flux_columns[i]
            
            comment = 'ZP offset in filter {0} ({1})'.format(f_i, c_i)
            tab.meta['ZP{0}'.format(f_i)] = self.zp[i], comment
            
        tab.meta['FNUSCALE'] = (fnu_scl, 'Scale factor to f-nu CGS')
        tab.meta['COSMOL'] = (cosmology.name, 'Cosmological model')
        tab.meta['COS_OM'] = (cosmology.Om0, 'Omega matter')
        tab.meta['COS_OL'] = (cosmology.Ode0, 'Omega lambda')
        tab.meta['COS_H0'] = (cosmology.H0.value, 'Hubble constant')
        
        tab.meta['RF_PADW'] = (rf_pad_width, 'pad_width for RF fluxes')
        tab.meta['RF_PADM'] = (rf_max_err, 'max_err for RF fluxes')
        
        # Additional Rest-frame filters
        if len(extra_rf_filters) > 0:
            extra_tempfilt, extra_lc, extra_rest = self.rest_frame_fluxes(f_numbers=extra_rf_filters, pad_width=rf_pad_width, max_err=rf_max_err, percentiles=[16,50,84], verbose=False, simple=simple) 
            for ir, f_n in enumerate(extra_rf_filters):
                tab['rest{0}'.format(f_n)] = extra_rest[:,ir,1]
                tab['rest{0}_err'.format(f_n)] = (extra_rest[:,ir,2]-extra_rest[:,ir,0])/2.
                tab['rest{0}'.format(f_n)].format = '.3f'
                tab['rest{0}_err'.format(f_n)].format = '.3f'
                
                tab.meta['name{0}'.format(f_n)] = (self.RES[f_n].name.split(' lambda_c')[0], 'Filter name')
                tab.meta['pivot{0}'.format(f_n)] = (self.RES[f_n].pivot, 'Pivot wavelength, Angstrom')
            
            del(extra_tempfilt)
            del(extra_rest)
            
        del(coeffs_norm)
        del(coeffs_draws)
        del(draws_norm)
        
        return tab


    def standard_output(self, zbest=None, prior=False, beta_prior=False, UBVJ=DEFAULT_UBVJ_FILTERS, extra_rf_filters=DEFAULT_RF_FILTERS, cosmology=None, LIR_wave=[8,1000], rf_pad_width=0.5, rf_max_err=0.5, save_fits=True, get_err=True, percentile_limits=[2.5, 16, 50, 84, 97.5], fitter='nnls', n_proc=0, clip_wavelength=1100, absmag_filters=[271, 272, 274], run_find_peaks=False, simple=False):#
        """
        Full output to ``zout.fits`` file.  
        
        First refits the coefficients at ``zml`` and optionally ``zbest``.
        
        Computes redshift statistics and then sends arguments to 
        `~eazy.photoz.PhotoZ.sps_parameters` for rest-frame colors, masses, 
        etc.
        
        """
        import astropy.io.fits as pyfits
        from .version import __version__
        
        if self.param['VERBOSITY'] >= 1:
            print('Get best fit coeffs & best redshifts')
                        
        tab = Table()
        tab['id'] = self.cat['id']
        for col in ['ra', 'dec']:
            if col in self.cat.colnames:
                tab[col] = self.cat[col]
                
        tab['z_spec'] = self.cat['z_spec']
        tab['nusefilt'] = self.nusefilt

        # Fit at max-lnp (default if zbest = None) first and record this 
        # information no matter what.          
        self.fit_at_zbest(zbest=None, prior=prior, beta_prior=beta_prior, 
                      get_err=get_err, fitter=fitter, n_proc=0, 
                      clip_wavelength=clip_wavelength)
        
        tab['z_ml'] = self.zbest
        tab['z_ml_chi2'] = self.chi2_best 
        tab['z_ml_risk'] = self.zbest_risk
            
        # Fit at the user's requested zbest, which defaults to 
        # z_pdf if zbest is None
        if zbest is not None:
            self.fit_at_zbest(zbest=zbest, prior=prior, beta_prior=beta_prior, 
                          get_err=get_err, fitter=fitter, n_proc=0, 
                          clip_wavelength=clip_wavelength)
               
        try:
            zlimits = self.pz_percentiles(percentiles=[2.5,16,50,84,97.5],
                                          oversample=5)
        except:
            print('Couldn\'t compute pz_percentiles')
            zlimits = np.zeros((self.NOBJ, 5), dtype=self.ARRAY_DTYPE) - 1
        
        # min/max observed wavelengths of valid data
        lc_full = np.dot(np.ones((self.NOBJ, 1)), self.lc[np.newaxis,:])
        tab['lc_min'] = (lc_full*(self.ok_data +
                                  1e10*(~self.ok_data))).min(axis=1)
        tab['lc_max'] = (lc_full*self.ok_data).max(axis=1)
        tab['lc_max'].format = tab['lc_min'].format = '.1f'
        
        if run_find_peaks:
            peaks, numpeaks = self.find_peaks()
            tab['numpeaks'] = numpeaks
            
        tab['z_phot'] = self.zbest
        tab['z_phot_chi2'] = self.chi2_best #chi2_fit.min(axis=1)
        tab['z_phot_risk'] = self.zbest_risk
        
        self.Rz = self.compute_full_risk()
        tab['z_min_risk'] = self.zgrid[np.argmin(self.Rz, axis=1)]
        tab['min_risk'] = self.Rz.min(axis=1)
                
        tab['z_raw_chi2'] = self.zchi2
        tab['raw_chi2'] = self.chi2_fit.min(axis=1)
        
        tab['z025'] = zlimits[:,0]
        tab['z160'] = zlimits[:,1]
        tab['z500'] = zlimits[:,2]
        tab['z840'] = zlimits[:,3]
        tab['z975'] = zlimits[:,4]
        
        for col in tab.colnames[-6:]:
            tab[col].format='8.4f'
            
        tab.meta['version'] = (__version__, 'Eazy-py version')
        tab.meta['prior'] = (prior, 'Prior applied ({0})'.format(self.param.params['PRIOR_FILE']))
        tab.meta['betprior'] = (beta_prior, 'Beta prior applied')
        tab.meta['fitter'] = (fitter, 'Optimization method for template fits')
        
        if self.param['VERBOSITY'] >= 1:
            print('Get parameters (UBVJ={0}, LIR={1})'.format(UBVJ, LIR_wave))
            
        sps_tab = self.sps_parameters(UBVJ=UBVJ, 
                          extra_rf_filters=extra_rf_filters, 
                          cosmology=cosmology, LIR_wave=LIR_wave, 
                          rf_pad_width=rf_pad_width, rf_max_err=rf_max_err, 
                          percentile_limits=percentile_limits, 
                          simple=simple)
                          
        for col in sps_tab.colnames:
            tab[col] = sps_tab[col]
        
        for key in sps_tab.meta:
            tab.meta[key] = sps_tab.meta[key]
        
        if len(absmag_filters) > 0:
            print('Abs Mag filters', absmag_filters)
            absm = self.abs_mag(f_numbers=absmag_filters, cosmology=cosmology, 
                               rest_kwargs={'percentiles':percentile_limits, 
                                            'pad_width':rf_pad_width, 
                                            'max_err':rf_max_err,
                                            'simple':simple})
            
            for c in absm.colnames:
                tab[c] = absm[c]
            
            for key in absm.meta:
                tab.meta[key] = absm.meta[key]
                
        if not save_fits:
            return tab, None
        
        root = self.param.params['MAIN_OUTPUT_FILE']
        if os.path.exists('{0}.zout.fits'.format(root)):
            os.remove('{0}.zout.fits'.format(root))
        
        tab.write('{0}.zout.fits'.format(root), format='fits')
        
        self.param.write('{0}.zphot.param'.format(root))
        self.write_zeropoint_file('{0}.zphot.zeropoint'.format(root))
        self.translate.write('{0}.zphot.translate'.format(root))
        
        hdu = pyfits.HDUList(pyfits.PrimaryHDU())
        #hdu.append(pyfits.ImageHDU(self.cat['id'].astype(np.uint32), name='ID'))
        hdu.append(pyfits.ImageHDU(self.zbest.astype(np.float32), name='ZBEST'))
        hdu.append(pyfits.ImageHDU(self.zgrid.astype(np.float32), name='ZGRID'))
        hdu.append(pyfits.ImageHDU(self.chi2_fit.astype(np.float32), name='CHI2'))
        hdu[-1].header['PRIOR'] = (prior, 'Prior applied ({0})'.format(self.param.params['PRIOR_FILE']))
        
        # Template coefficients 
        hdu.append(pyfits.ImageHDU(self.coeffs_best, name='COEFFS'))
        h = hdu[-1].header
        h['ABZP'] = (self.param['PRIOR_ABZP'], 'AB zeropoint')
        h['NTEMP'] = (self.NTEMP, 'Number of templates')
        for i, t in enumerate(self.templates):
            h['TEMP{0:04d}'.format(i)] = t.name
        
        if save_fits == 1:
            hdu.writeto('{0}.data.fits'.format(root), overwrite=True)
            
        return tab, hdu


    def get_match_index(self, id=None, rd=None, verbose=True):
        """
        Get object index of closest match based either on ``id`` (exact) or 
        closest to specified ``(ra, dec) = rd``.
        """
        import astropy.units as u
        
        if id is not None:
            ix = np.where(self.cat['id'] == id)[0][0]
            if verbose:
                print('ID={0}, idx={1}'.format(id, ix))
                
            return ix
        
        # From RA / DEC
        idx, dr = self.cat.match_to_catalog_sky([rd[0]*u.deg, rd[1]*u.deg])
        if verbose:
            msg = 'ID={0}, idx={1}, dr={2:.3f}'
            print(msg.format(self.cat['id'][idx[0]], idx[0], dr[0]))
            
        return idx[0]


    def to_prospector(self, id=None, rd=None):
        """
        Get the photometry and filters in a format that 
        Prospector can use
        """
        from sedpy.observate import Filter
        
        ix = self.get_match_index(id, rd)
        
        ZP = self.param['PRIOR_ABZP']
        maggies = self.fnu[ix,:]*10**(-0.4*ZP)
        maggies_unc = self.efnu[ix,:]*10**(-0.4*ZP)
        dq = (self.fnu[ix,:] > self.param['NOT_OBS_THRESHOLD']) & (self.efnu[ix,:] > 0) & np.isfinite(self.fnu[ix,:]) & np.isfinite(self.efnu[ix,:])
        
        sedpy_filters = []
        for i in range(self.NFILT):
            if dq[i]:
                filt = self.filters[i]
                sedpy_filters.append(Filter(data=[filt.wave, filt.throughput], kname=filt.name.split()[0]))
        
        pivot = np.array([f.wave_pivot for f in sedpy_filters])
        lightspeed = 2.998e18  # AA/s
        conv = pivot**2 / lightspeed * 1e23 / 3631 
        
        pdict = {'maggies': maggies[dq],
                 'maggies_unc': maggies_unc[dq],
                 'maggies_toflam':1/conv,
                 'filters':sedpy_filters,
                 'wave_pivot':pivot,
                 'phot_catalog_id':self.cat['id'][ix]}
        
        return pdict


    def get_grizli_photometry(self, id=1, rd=None, grizli_templates=None):
        """
        Get photometry dictionary of a given object that can be used with 
        `~grizli` fits.
        
        """
        from collections import OrderedDict
        import astropy.units as u
        
        if grizli_templates is not None:
            template_list = [templates_module.Template(arrays=(grizli_templates[k].wave, grizli_templates[k].flux), name=k) for k in grizli_templates]
            
            tempfilt = TemplateGrid(self.zgrid, template_list, RES=self.param['FILTERS_RES'], f_numbers=self.f_numbers, add_igm=self.param['IGM_SCALE_TAU'], galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'], cosmology=self.cosmology, array_dtype=self.ARRAY_DTYPE)
        else:
            tempfilt = None
        
        if rd is not None:
            ti = Table()
            ti['ra'] = [rd[0]]
            ti['dec'] = [rd[1]]
            
            idx, dr = self.cat.match_to_catalog_sky(ti)
            idx = idx[0]
            dr = dr[0]
        else:
            idx = np.where(self.cat['id'] == id)[0][0]
            dr = 0
        
        notobs_mask =  self.fnu[idx,:] < self.param['NOT_OBS_THRESHOLD']
        sed = self.show_fit(idx, show_fnu=False, xlim=[0.3, 9], get_spec=True, id_is_idx=True)
        sed['fobs'][notobs_mask] = self.param['NOT_OBS_THRESHOLD'] - 9.
        sed['efobs'][notobs_mask] = self.param['NOT_OBS_THRESHOLD'] - 9.
        
        photom = OrderedDict()
        photom['flam'] = sed['fobs']*1.e-19
        photom['eflam'] = sed['efobs']*1.e-19
        photom['filters'] = self.filters
        photom['tempfilt'] = tempfilt
        photom['pz'] = self.zgrid, np.exp(self.lnp[idx,:])
        
        return photom, self.cat['id'][idx], dr


    def rest_frame_SED(self, idx=None, norm_band=155, c='k', min_sn=3, median_args=dict(NBIN=50, use_median=True, use_nmad=True, reverse=False), get_templates=True, make_figure=True, scatter_args=None, show_uvj=True, axes=None, **kwargs):
        """
        Make Rest-frame SED plot
        
        idx: selection array
        
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        if isinstance(norm_band, int):
            init_sed_data = True
            if hasattr(self, 'rf_sed_data'):
                rf_tempfilt, rf_lc, f_rest = self.rf_sed_data
                if rf_lc == self.RES[norm_band].pivot:
                    init_sed_data = False
                    
            if init_sed_data:
                rf_tempfilt, rf_lc, f_rest = self.rest_frame_fluxes(f_numbers=[norm_band], pad_width=0.5, percentiles=[2.5,16,50,84,97.5], simple=True) 
                self.rf_sed_data = (rf_tempfilt, rf_lc, f_rest)
        else:
            rf_tempfilt, rf_lc, f_rest = norm_band
            
        norm_flux = f_rest[:,0,2]
        fnu_norm = (self.fnu[idx,:].T/norm_flux[idx]).T
        fmodel_norm = (self.fmodel[idx,:].T/norm_flux[idx]).T
        
        output_data = {}
        
        lcz = np.dot(1/(1+self.zbest[:, np.newaxis]), self.lc[np.newaxis,:])[idx,:]
        
        clip = (self.efnu[idx,:] > 0) & (self.fnu[idx,:] > self.param['NOT_OBS_THRESHOLD']) & np.isfinite(self.fnu[idx,:]) & np.isfinite(self.efnu[idx,:])
        clip *= self.fnu[idx,:]/self.efnu[idx,:] > min_sn
        
        wave = lcz[clip]
        flam = fnu_norm[clip]/(wave/rf_lc[0])**2
        flam_obs = fmodel_norm[clip]/(wave/rf_lc[0])**2
        
        output_data['phot_wave'] = wave
        output_data['phot_flam'] = flam
        output_data['phot_flam_model'] = fmodel_norm
        
        # Running median
        xm, ym, ys, N = utils.running_median(wave, flam, **median_args)
        
        output_data['sed_wave'] = fmodel_norm
        output_data['sed_flam'] = xm
        output_data['sed_nmad'] = ys

        if get_templates:
            sp = self.show_fit(self.cat['id'][idx][0], get_spec=True)
            templf = []
            for i in self.idx[idx]:
                sp = self.show_fit(self.cat['id'][i], get_spec=True)
                sp_i = sp['templf']/(norm_flux[i]/(1+self.zbest[i])**2)
                templf.append(sp_i)
                                
            med = np.median(np.array(templf), axis=0)/3.6
            tmin = np.percentile(np.array(templf), 16, axis=0)/3.6
            tmax = np.percentile(np.array(templf), 84, axis=0)/3.6
        
            output_data['templ_wave'] = self.templates[0].wave
            output_data['templ_flam'] = np.array(templf)
            output_data['templ_med'] = med
            output_data['templ_p16'] = tmin
            output_data['templ_p84'] = tmax
        
        if not make_figure:
            return output_data
            
        # Make figure
        if axes is None:
            fig = plt.figure(figsize=[10,4])
        else:
            fig = None
            
        # UVJ?
        if (self.ubvj is not None) & show_uvj:
            UV = -2.5*np.log10(self.ubvj[:,0,2]/self.ubvj[:,2,2])
            VJ = -2.5*np.log10(self.ubvj[:,2,2]/self.ubvj[:,3,2])
            ok = np.isfinite(UV) & np.isfinite(VJ)
            
            gs = GridSpec(1,2, width_ratios=[2,3])
        
            if axes is None:
                ax = fig.add_subplot(gs[0,0])
            else:
                ax = axes[0]
                
            sc = ax.scatter(VJ[ok], UV[ok], c='k', vmin=-1, vmax=1.,
                            alpha=0.01, marker='.', edgecolor='k', zorder=-1)
            sc = ax.scatter(VJ[idx], UV[idx], c=c, vmin=-1, vmax=1., 
                            alpha=0.2, marker='o', edgecolor='k', zorder=1)
        
            if axes is None:
                ax.set_xlabel(r'$V-J$ (rest)')
                ax.set_ylabel(r'$U-V$ (rest)')
                ax.set_xlim(-0.2,2.8); ax.set_ylim(-0.2,2.8)
                ax.grid()

                ax = fig.add_subplot(gs[0,1])
            else:
                ax = axes[1]
        else:
            gs = GridSpec(1,1, width_ratios=[1])
        
            if axes is None:
                ax = fig.add_subplot(gs[0,0])
            else:
                ax = axes[1]
                
        ax.plot(xm, np.maximum(ym, 0.01), color=c, linewidth=2, alpha=0.4)
        ax.fill_between(xm, np.maximum(ym+ys, 0.001), np.maximum(ym-ys, 0.001), color=c, alpha=0.4)
        if scatter_args is not None:
            ax.scatter(wave, flam, **scatter_args)
            
        if get_templates:
            ax.plot(self.templates[0].wave[::5], med[::5], color=c, 
                linewidth=1, zorder=2)
    
            ax.fill_between(self.templates[0].wave[::5], tmin[::5], tmax[::5], 
                            color=c, linewidth=1, zorder=2, alpha=0.1)
        
        # MIPS
        if hasattr(self, 'mips_scaled'):
            # MIPS flux in this catalog zeropoint
            # mips_scaled = kate_sfr['f24tot']*10**(0.4*(self.param.params['PRIOR_ABZP']-23.9))
            
            mips_obs = self.mips_scaled/norm_flux/(24.e4/(1+self.zbest)/rf_lc[0])**2#/2
            ok_mips = (mips_obs > 0)
        
            xm, ym, ys, N = utils.running_median(24.e4/(1+self.zbest[idx & ok_mips]), np.log10(mips_obs[idx & ok_mips]), NBIN=10, use_median=True, use_nmad=True, reverse=False)
            ax.fill_between(xm, np.maximum(10**(ym+ys), 1.e-4), np.maximum(10**(ym-ys), 1.e-4), color=c, alpha=0.4)
        
            ax.scatter(24.e4/(1+self.zbest[idx & ok_mips]), mips_obs[idx & ok_mips], color=c, marker='.', alpha=0.1)

            ax.set_xlim(2000,120.e5)
        
        ax.scatter(rf_lc[0], 1, marker='x', color=c, zorder=1000)
                
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
        
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])
        else:
            ax.set_ylim(9.e-4,4)
            
        if axes is None:
            ax.set_xlabel(r'$\lambda_\mathrm{rest}$')
            ax.set_ylabel(r'$f_\lambda\ /\ f_V$')
            ax.loglog()
            ax.grid()
            gs.tight_layout(fig)

        #fig.tight_layout()
        return output_data, fig
        
        #fig.savefig('gs_eazypy.RF_{0}.png'.format(label))


    def spatial_statistics(self, band_indices=None, xycols=('ra','dec'), is_sky=True, nbin=(50,50), bins=None, apply=False, min_sn=10, catalog_mask=None, statistic='median', zrange=[0.05, 4], verbose=True, vm=(0.92, 1.08), output_suffix='', save_results=True, make_plot=True, cmap='plasma', figsize=5, plot_format='png', close=True, scale_by_uncertainties=False):
        """
        Show statistics as a function of position
        
        band_indices : list of int, None
            Indices of the bands to process, in the order of the 
            `self.lc`, `self.filters`, etc. lists.  If None, do all of them.
        
        statistic : str
            See `~scipy.stats.binned_statistic_2d`.
            
        
        """
        from scipy.stats import binned_statistic_2d
        import matplotlib.pyplot as plt
        import astropy.stats
        
        # Coordinate things
        xc = self.cat[xycols[0]]
        yc = self.cat[xycols[1]]
        
        xr = [xc.min(), xc.max()]
        yr = [yc.min(), yc.max()]
        dx, dy = xr[1]-xr[0], yr[1]-yr[0]
        aspect = dy/dx

        if bins is None:
            px, py = 1/nbin[0], 1/nbin[1]
            xbins = np.linspace(xr[0]-px*dx, xr[1]+px*dx, nbin[0])
            ybins = np.linspace(yr[0]-py*dy, yr[1]+py*dy, nbin[1])
        else:
            xbins, ybins = bins
            
        if is_sky:
            xr = xr[::-1]
            cosd = np.cos(np.mean(yc)/180*np.pi)
            aspect /= cosd

        # Residuals
        if scale_by_uncertainties:
            resid = (self.fmodel/self.ext_redden/self.zp - self.fnu)/self.efnu
        else:
            resid = (self.fmodel - self.fnu*self.ext_redden*self.zp)/self.fmodel
        
        # Data mask
        mask = (self.fnu/self.efnu_orig > min_sn) & (self.efnu > 0) & (self.fnu > self.param['NOT_OBS_THRESHOLD']) & np.isfinite(resid)
        
        object_mask = (self.zbest > zrange[0]) & (self.zbest < zrange[1])
        if catalog_mask is not None:
            object_mask &= catalog_mask

        mask = (mask.T & object_mask).T
        
        # By filter
        if band_indices is None:
            band_indices = range(self.NFILT)
        
        for i in band_indices:
            col_i = self.flux_columns[i]
            f_name = self.tempfilt.filter_names[i].split()[0]
            msum = mask[:,i].sum()
            
            label = '{0} / {1} (N={2:>6d})'.format(col_i, f_name, msum)
            
            if msum == 0:
                continue
                
            ret = binned_statistic_2d(xc[mask[:,i]], yc[mask[:,i]], resid[mask[:,i],i]+1, bins=(xbins, ybins), statistic=statistic)
            
            if apply & (statistic in ['mean', 'median', astropy.stats.biweight_location]):
                self.apply_spatial_offset(i, ret, xycols=xycols)
                if verbose:
                    print(label + ' (applied)')
            else:
                if verbose:
                    print(label)
                
            if save_results:
                save_file = '{0}_{1}{2}.{3}'.format(self.param.params['MAIN_OUTPUT_FILE'], col_i, output_suffix, 'npy')
                np.save(save_file, [[i, col_i, f_name, msum, min_sn], ret])
                
            if make_plot:
                fig = plt.figure(figsize=[figsize, figsize*aspect])
                ax = fig.add_subplot(111)
                ax.imshow(ret.statistic.T, vmin=vm[0], vmax=vm[1], extent=(ret.x_edge[0], ret.x_edge[-1], ret.y_edge[0], ret.y_edge[-1]), origin='lower', cmap=cmap)
                
                ax.set_xlim(xr)
                ax.set_ylim(yr)
                #ax.set_title(label)
                ax.text(0.05, 0.97, label, ha='left', va='top', transform=ax.transAxes, fontsize=8, zorder=2000, bbox=dict(facecolor='w', alpha=0.8))
                ax.set_xlabel(xycols[0])
                ax.set_ylabel(xycols[1])
                ax.grid(zorder=500)
                ax.set_aspect(1./cosd)
                
                fig.tight_layout(pad=0.1)
                fig_file = '{0}_{1}{2}.{3}'.format(self.param.params['MAIN_OUTPUT_FILE'], col_i, output_suffix, plot_format)
                fig.savefig(fig_file)
                if close:
                    plt.close()
        
    def apply_spatial_offset(self, f_ix, bin2d, xycols=('ra','dec')):
        """
        Apply a spatial zeropoint offset determined from    
        spatial_statistics.
        """
        xc = self.cat[xycols[0]]
        yc = self.cat[xycols[1]]
        
        nx = len(bin2d.x_edge)
        ny = len(bin2d.y_edge)
        
        ex = np.arange(nx)
        ey = np.arange(ny)
        
        ix = np.cast[int](np.interp(xc, bin2d.x_edge, ex))
        iy = np.cast[int](np.interp(yc, bin2d.y_edge, ey))
        
        corr = bin2d.statistic[ix, iy]
        corr[~np.isfinite(corr)] = 1
        
        if self.spatial_offset is not None:
            self.spatial_offset[:,f_ix] *= corr
        else:
            self.spatial_offset = np.ones_like(self.fnu)
            self.spatial_offset[:,f_ix] *= corr
            
        mask = (self.fnu[:,f_ix] > self.param['NOT_OBS_THRESHOLD']) 
        mask &= (self.efnu_orig[:,f_ix] > 0)
        self.fnu[mask,f_ix] *= corr[mask]
        self.efnu_orig[mask,f_ix] *= corr[mask]
        self.efnu = np.sqrt(self.efnu_orig**2+(self.param['SYS_ERR']*self.fnu)**2)


    def fit_phoenix_stars(self, wave_lim=[3000, 4.e4], apply_extcorr=False, sys_err=None):
        """
        Fit grid of Phoenix stars
        
        `apply_extcorr` defaults to False because stars not necessarily 
        "behind" MW extinction
        
        """
        stars = templates_module.load_phoenix_stars(add_carbon_star=False)
        self.star_templates = stars
        tflux = [t.integrate_filter(self.filters, z=0, include_igm=False)
                     for t in self.star_templates]
        
        templ_params = [[float(s[1:]) for s in t.name.split('_')[1:]] 
                         for t in stars]
        self.star_teff = np.array(templ_params)[:,0]
        self.star_logg = np.array(templ_params)[:,1]
        self.star_zmet = np.array(templ_params)[:,2]

        if apply_extcorr:
            self.star_flux = (np.array(tflux)*self.ext_corr).T
        else:
            self.star_flux = np.array(tflux).T
            
        self.NSTAR = self.star_flux.shape[1]
        
        # Least squares normalization of stellar templates
        if sys_err is None:
            _wht = 1/(self.efnu**2+(self.param.params['SYS_ERR']*self.fnu)**2)
        else:
            _wht = 1/(self.efnu**2+(sys_err*self.fnu)**2)
            
        _wht[(~self.ok_data) | (self.efnu <= 0)] = 0
        
        clip_filter = (self.lc < wave_lim[0]) | (self.lc > wave_lim[1])
         
        _wht[:, clip_filter] = 0
            
        _num = np.dot(self.fnu*_wht, self.star_flux)
        _den= np.dot(1*_wht, self.star_flux**2)
        _den[_den == 0] = 0
        self.star_tnorm = _num/_den
        
        # Chi-squared
        self.star_chi2 = np.zeros(self.star_tnorm.shape, dtype=np.float32)
        for i in range(self.NSTAR):
            _m = self.star_tnorm[:,i:i+1]*self.star_flux[:,i]
            self.star_chi2[:,i] = ((self.fnu - _m)**2*_wht).sum(axis=1)
        
        self.star_min_ix = np.argmin(self.star_chi2, axis=1)
        
        self.star_min_chi2 = self.star_chi2.min(axis=1)
        self.star_min_chinu = self.star_min_chi2 / (self.nusefilt - 1)


    def _redshift_pairs(self, rix=None):
        """
        Redshift differences of pairs
        
        TBD
        """
        import itertools
        
        if rix is None:
            rix = self.zbest > 0
            
        r0, d0 = np.mean(self.cat['ra'][rix]), np.mean(self.cat['dec'][rix])
        cosd = np.cos(d0/180*np.pi)
        r0 = (self.cat['ra'][rix]-r0)*cosd*3600
        d0 = (self.cat['dec'][rix]-d0)*3600
        zix = self.zbest[rix]

        pair_inds = np.array(list(itertools.combinations(range(rix.sum()),2)))


def _obj_nnls(coeffs, A, fnu_i, efnu_i):
    fmodel = np.dot(coeffs, A)
    return -0.5*np.sum((fmodel-fnu_i)**2/efnu_i**2)


class TemplateGrid(object):
    def __init__(self, zgrid, templates, RES='FILTERS.latest', f_numbers=[156], add_igm=True, galactic_ebv=0, n_proc=4, Eb=0, interpolator=None, filters=None, verbose=2, cosmology=None, array_dtype=np.float32):
        """
        Integrate filters through filters on a redshift grid
        """
        import multiprocessing as mp
        import astropy.units as u
        from astropy.cosmology import WMAP9
        
        self.templates = templates
        self.RES = RES
        self.f_numbers = f_numbers
        self.add_igm = add_igm
        self.galactic_ebv = galactic_ebv
        self.Eb = Eb
        self.ARRAY_DTYPE = array_dtype
        
        if cosmology is None:
            cosmology = WMAP9
        
        self.cosmology = cosmology
        
        #self.NTEMP = len(templates)
        #self.NZ = len(zgrid)
        
        self.zgrid = zgrid
        self.trdz = utils.trapz_dx(self.zgrid)
        
        self.dz = np.diff(zgrid)
        self.idx = np.arange(self.NZ, dtype=int)
                
        if filters is None:
            all_filters = np.load(RES+'.npy', allow_pickle=True)[0]
            filters = [all_filters[fnum] for fnum in f_numbers]
        
        #self.lc = np.array([f.pivot for f in filters])
        self.filter_names = np.array([f.name for f in filters])
        self.filters = filters
        #self.NFILT = len(self.filters)
        
        self.tempfilt = np.zeros((self.NZ, self.NTEMP, self.NFILT), 
                                 self.ARRAY_DTYPE)
        
        if n_proc >= 0:
            # Parallel            
            if n_proc == 0:
                pool = mp.Pool(processes=mp.cpu_count())
            else:
                pool = mp.Pool(processes=n_proc)
                
            results = [pool.apply_async(_integrate_tempfilt,
                                        (itemp, templates[itemp], zgrid, RES,
                                         f_numbers, add_igm, galactic_ebv, Eb,
                                         filters))
                       for itemp in range(self.NTEMP)]

            pool.close()
            pool.join()
                
            for res in results:
                itemp, tf_i = res.get(timeout=1)
                if verbose > 1:
                    print('Process template {0} (NZ={1}).'.format(templates[itemp].name, templates[itemp].NZ))
                self.tempfilt[:,itemp,:] = tf_i        
        else:
            # Serial
            for itemp in range(self.NTEMP):
                itemp, tf_i = _integrate_tempfilt(itemp, templates[itemp],
                                                  zgrid, RES, f_numbers, 
                                                  add_igm, galactic_ebv, Eb, 
                                                  filters)
                if verbose > 1:
                    msg = 'Process template {0}.'
                    print(msg.format(templates[itemp].name))
                    
                self.tempfilt[:,itemp,:] = tf_i        
        
        # Check for bad values.  Not sure where they're coming from?
        bad = ~np.isfinite(self.tempfilt)
        if bad.sum():
            print('Fix bad values in `tempfilt` (N={0})'.format(bad.sum()))
            self.tempfilt[bad] = 0
            
        self.interpolator_function = interpolator
        self.init_interpolator(interpolator=interpolator)


    @property 
    def NOBJ(self):
        """
        Number of objects in catalog
        """
        if not hasattr(self, 'cat'):
            return 0
        else:
            return len(self.cat)


    @property
    def NFILT(self):
        """
        Number of filters
        """
        if hasattr(self, 'filters'):
            return len(self.filters)
        else:
            return 0


    @property 
    def lc(self):
        """
        Filter pivot wavelengths
        """        
        if hasattr(self, 'filters'):
            return np.array([f.pivot for f in self.filters])
        else:
            return None


    @property 
    def NTEMP(self):
        """
        Number of templates
        """
        if hasattr(self, 'templates'):
            return len(self.templates)
        else:
            return 0


    @property
    def NZ(self):
        """
        Number of redshift grid points
        """
        if hasattr(self, 'zgrid'):
            return len(self.zgrid)
        else:
            return 0


    def init_interpolator(self, interpolator=None):
        """
        Initialize filter flux interpolator
        
        Parameters
        ----------
        interpolator: None or a `~scipy.interpolate` class.
            Defaults to `~scipy.interpolate.Akima1DInterpolator` which has 
            desirable smooth behavior robust to large curvature in the 
            interpolated data.
            
        """
        import scipy.interpolate 
        
        # Spline interpolator        
        if interpolator is None:
            self.spline = scipy.interpolate.Akima1DInterpolator(self.zgrid, self.tempfilt, axis=0)
            #self.spline = scipy.interpolate.CubicSpline(self.zgrid, self.tempfilt)
            self.interpolator_function = scipy.interpolate.Akima1DInterpolator
        else:
            self.spline = interpolator(self.zgrid, self.tempfilt, axis=0)
            self.interpolator_function = interpolator
    
    def apply_SFH_constraint(self, max_mass_frac=0.5, cosmology=None, sfh_file='templates/fsps_full/fsps_QSF_12_v3.sfh.fits'):
        """
        Set interpolated template fluxes to zero for a given redshift/tmeplate
        combination if the accumulated stellar mass fraction at ages older
        than the age of the universe is greater than `max_mass_frac`.
        
        Requires the "sfh.fits" file.
        
        .. warning::
            
            The implementation seems to work but the results when applied to 
            running on a full catalog don't seem very reliable, probably 
            caused by the effect of clipping out templates at discrete
            redshifts.
            
        """
        from astropy.table import Table
        import astropy.units as u
        
        if cosmology is None:
            #from astropy.cosmology import WMAP9 as cosmology
            cosmology = self.cosmology
            
        sfh = Table.read(sfh_file)
        mass_accum = np.cumsum(sfh['SFH'][::-1,:], axis=0)
        mass_accum = (mass_accum / mass_accum[-1,:])[::-1,:]
        t_accum = sfh['time']
        
        t_lb = cosmology.age(self.zgrid)
        for iz in range(self.NZ):
            t_ix = np.where(t_accum.data*u.Gyr <= t_lb[iz])[0][-1]
            mass_ix = mass_accum[t_ix,:]
            clip = mass_ix > max_mass_frac
            
            self.tempfilt[iz,clip,:] *= 0
        
        # Reinit interpolator
        self.init_interpolator(interpolator=self.interpolator_function)
        
    def __call__(self, z):
        """
        Return interpolated filter fluxes
        """
        
        return self.spline(z)
    
    
def _integrate_tempfilt(itemp, templ, zgrid, RES, f_numbers, add_igm, galactic_ebv, Eb, filters):
    """
    For multiprocessing filter integration
    """
    import astropy.units as u
    global IGM_OBJECT
    if filters is None:
        all_filters = np.load(RES+'.npy', allow_pickle=True)[0]
        filters = [all_filters[fnum] for fnum in f_numbers]
    
    NZ = len(zgrid)
    NFILT = len(filters)
        
    if add_igm:
        igm = IGM_OBJECT #igm_module.Inoue14(scale_tau=add_igm)
    else:
        igm = 1.

    f99 = utils.GalacticExtinction(EBV=galactic_ebv, Rv=3.1)
    
    # Add bump with Drude profile in template rest frame
    width = 350
    l0 = 2175
    tw = templ.wave
    Abump = Eb/4.05*(tw*width)**2/((tw**2-l0**2)**2+(tw*width)**2)
    Fbump = 10**(-0.4*Abump)
    
    tempfilt = np.zeros((NZ, NFILT))
    for iz in range(NZ):
        lz = templ.wave*(1+zgrid[iz])
        
        # IGM absorption
        if add_igm:
            igmz = templ.wave*0.+1
            lyman = templ.wave < 1300
            igmz[lyman] = igm.full_IGM(zgrid[iz], lz[lyman])
        else:
            igmz = 1.
            
        # Galactic Redenning        
        red = (lz > 910.) & (lz < 6.e4)
        A_MW = templ.wave*0.        
        A_MW[red] = f99(lz[red])
        
        F_MW = 10**(-0.4*A_MW)
        
        for ifilt in range(NFILT):
            fnu = templ.integrate_filter(filters[ifilt], 
                                         scale=igmz*F_MW*Fbump, 
                                         z=zgrid[iz], 
                                         include_igm=False)
                                         
            tempfilt[iz, ifilt] = fnu
    
    return itemp, tempfilt


def _fit_vertical(iz, z, A, fnu_corr, efnu_corr, TEF, zp, verbose, fitter):
    """
    Fit all objects at a given reshift for redshift-fit parallelization
    """
    NOBJ, NFILT = fnu_corr.shape#[0]
    NTEMP = A.shape[0]
    chi2 = np.zeros(NOBJ)
    coeffs = np.zeros((NOBJ, NTEMP))
    TEFz = TEF(z)
    
    if verbose > 1:
        print('z={0:7.3f}'.format(z))
    
    for iobj in range(NOBJ):
        
        fnu_i = fnu_corr[iobj, :]
        efnu_i = efnu_corr[iobj,:]
        ok_band = (efnu_i > 0)
        
        if ok_band.sum() < 2:
            continue
        
        chi2[iobj], coeffs[iobj], fmodel, draws = _fit_obj(fnu_i, efnu_i, A, TEFz, zp, False, fitter)
            
    return iz, chi2, coeffs


def _fit_at_zbest_group(ix, fnu_corr, efnu_corr, zbest, zp, get_err, fitter, tempfilt, TEF, ARRAY_DTYPE, _self):
    """
    Standalone function for fitting individual objects and getting 
    coefficients and random draws
    """
    #TEF, tempfilt = np.load(savefile, allow_pickle=True)
    NOBJ = len(ix)
    
    NTEMP = tempfilt.NTEMP
    if _self is None:
        coeffs_best = np.zeros((NOBJ, tempfilt.NTEMP), dtype=ARRAY_DTYPE)
        fmodel = np.zeros((NOBJ, tempfilt.NFILT), dtype=ARRAY_DTYPE)
        efmodel = np.zeros((NOBJ, tempfilt.NFILT), dtype=ARRAY_DTYPE)
        chi2_best = np.zeros(NOBJ, dtype=ARRAY_DTYPE)
        coeffs_draws = np.zeros((NOBJ, 100, tempfilt.NTEMP),
                                dtype=ARRAY_DTYPE)
    else:
        # In place, avoid making copies
        coeffs_best = _self.coeffs_best
        fmodel = _self.fmodel
        efmodel = _self.efmodel
        chi2_best = _self.chi2_best
        coeffs_draws = _self.coeffs_draws
        
    idx = np.where((zbest > tempfilt.zgrid[0]) & 
                   (zbest < tempfilt.zgrid[-1]))[0]
    
    for iobj in idx:
        
        zi = zbest[iobj]
        A = tempfilt(zi)
        TEFz = TEF(zi)

        fnu_i = fnu_corr[iobj, :]
        efnu_i = efnu_corr[iobj,:]
        if get_err:
            _ = _fit_obj(fnu_i, efnu_i, A, TEFz, zp, 100, fitter)
            chi2, coeffs_best[iobj,:], fmodel[iobj,:], draws = _
            if draws is None:
                efmodel[iobj,:] = -1
            else:
                #tf = self.tempfilt(zi)
                efm = np.diff(np.percentile(np.dot(draws, A), [16,84], 
                                            axis=0), axis=0)/2.
                efmodel[iobj,:] = efm
                coeffs_draws[iobj, :, :] = draws
        else:
            _ = _fit_obj(fnu_i, efnu_i, A, TEFz, zp, False, fitter)
            chi2, coeffs_best[iobj,:], fmodel[iobj,:], draws = _

        chi2_best[iobj] = chi2
    
    if _self is None:
        return ix, coeffs_best, fmodel, efmodel, chi2_best, coeffs_draws
    else:
        return True


def _fit_rest_group(ix, fnu_corr, efnu_corr, izbest, zbest, zp, get_err, fitter, tempfilt, ARRAY_DTYPE, rf_tempfilt, percentiles, rf_lc, pad_width, max_err):
    """
    Standalone function for fitting rest-frame fluxes for individual objects
    """
    #TEF, tempfilt = np.load(savefile, allow_pickle=True)
    NOBJ = len(ix)
    NTEMP = tempfilt.NTEMP    
    NREST = rf_tempfilt.shape[2]
    f_rest = np.zeros((NOBJ, NREST, len(percentiles)),
                      dtype=ARRAY_DTYPE)
    
    idx = np.where((zbest > tempfilt.zgrid[0]) & 
                   (zbest < tempfilt.zgrid[-1]))[0]
                
    for iobj in idx:

        fnu_i = fnu_corr[iobj,:]*1
        efnu_i = efnu_corr[iobj,:]*1
        z = zbest[iobj]
        if (z < 0) | (~np.isfinite(z)):
            continue
        
        A = tempfilt(z)   
        iz = izbest[iobj]
        
        for i in range(NREST):
            ## Grow uncertainties away from RF band
            #lc_i = rf_tempfilt.lc[i]
            lc_i = rf_lc[i]
            
            # Normal in log wavelength
            x = np.log(lc_i/(tempfilt.lc/(1+z)))
            grow = np.exp(-x**2/2/np.log(1/(1+pad_width))**2)

            TEFz = (2/(1+grow/grow.max())-1)*max_err
        
            _ = _fit_obj(fnu_i, efnu_i, A, TEFz, zp, 100, fitter)
            chi2_i, coeffs_i, fmodel_i, draws = _
            
            if draws is None:
                f_rest[iobj,i,:] = -1.
            else:
                dval = np.dot(draws, rf_tempfilt[iz,:,i])
                f_rest[iobj,i,:] = np.percentile(dval, percentiles, axis=0)
                del(dval)
    
    return ix, f_rest


#BOUNDED_DEFAULTS = {'bounds':(1.e3, 1.e18), 'method': 'bvls', 'tol': 1.e-8, 'verbose': 0}
BOUNDED_DEFAULTS = {'bound_range':[0.05, 20], 'method': 'trf', 'tol': 1.e-8, 'verbose': 0, 'normalize_type':0}

def _fit_obj(fnu_i, efnu_i, A, TEFz, zp, get_err, fitter):
    """
    Main optimization / fitting functino
    """
    from scipy.optimize import nnls
    import scipy.optimize
    
    global MIN_VALID_FILTERS
    global BOUNDED_DEFAULTS
    
    sh = A.shape

    # Valid fluxes
    ok_band = (efnu_i/zp > 0) & np.isfinite(fnu_i) & np.isfinite(efnu_i)
    if ok_band.sum() < MIN_VALID_FILTERS:
        coeffs_i = np.zeros(sh[0])
        fmodel = np.dot(coeffs_i, A)
        return np.inf, np.zeros(A.shape[0]), fmodel, None
        
    var = efnu_i**2 + (TEFz*fnu_i)**2
    rms = np.sqrt(var)
    
    # Nonzero templates
    ok_temp = (np.sum(A, axis=1) > 0)
    if ok_temp.sum() == 0:
        coeffs_i = np.zeros(sh[0])
        fmodel = np.dot(coeffs_i, A)
        return np.inf, np.zeros(A.shape[0]), fmodel, None
    
    # Least-squares fit    
    Ax = (A/rms).T[ok_band,:]*1
    try:
        if fitter == 'nnls':
            coeffs_x, rnorm = nnls(Ax[:,ok_temp], (fnu_i/rms)[ok_band])
            
        elif fitter == 'bounded':
            func = scipy.optimize.lsq_linear
            
            if 'bound_range' in BOUNDED_DEFAULTS:
                br = BOUNDED_DEFAULTS['bound_range']
            else:
                br = [0.05, 20]
            
            #print('xxx BR: ', br)
            
            ### Normalize templates
            normalize_type = 0
            if 'normalize_type' in BOUNDED_DEFAULTS:
                normalize_type = BOUNDED_DEFAULTS['normalize_type']
            
            bound_kwargs = {}
            for k in BOUNDED_DEFAULTS:
                if k not in ['bound_range','normalize_type']:
                    bound_kwargs[k] = BOUNDED_DEFAULTS[k]
                    
            if normalize_type == 0:
                # Fit templates individually
                norm = (Ax[:,ok_temp].T*(fnu_i/rms)[ok_band]).sum(axis=1)
                norm /= (Ax[:,ok_temp].T**2).sum(axis=1)
                norm = norm[norm > 0]
                bounds = (br[0]*norm.min(), br[1]*norm.max())
                A0 = 1
            else:
                A0 = A[ok_temp,0]
                A0nn = A0 > 0
                A0t = (A[ok_temp,:].T/A0).T[A0nn]
                bounds = (br[0]*A0t.min(), br[1]*A0t.max())

            lsq_out = func(Ax[:,ok_temp]/A0, (fnu_i/rms)[ok_band], 
                           bounds=bounds, **bound_kwargs)
            coeffs_x = lsq_out.x/A0
            
        elif fitter.startswith('normal'):
            # print('Fit with normal equations')
            # Transform A to normal equations
                            
            ivmat = np.diag(1./var[ok_band])
            Ai = A[ok_temp,:][:,ok_band].T
            An = np.median(Ai)
            Ai /= An
            LHS = Ai.T.dot(ivmat).dot(Ai)
            
            # Regularization
            if '_' in fitter:
                lamb = float(fitter.split('_')[1])
                n_col = ok_temp.sum()
                LHS += lamb*np.identity(n_col)
                
            RHS = Ai.T.dot(ivmat).dot(fnu_i[ok_band])
            if '+' in fitter:
                coeffs_x, rnorm = nnls(LHS, RHS)
            else:
                coeffs_x = np.linalg.solve(LHS, RHS)
            
            coeffs_x /= An
            
        elif fitter.startswith('regularized'):
            # With regularization
            if '_' in fitter:
                lamb = float(fitter.split('_')[1])
            else:
                lamb = 0.5
            
            Ai = Ax[:,ok_temp]*1
            An = np.median(Ai)
            # Ai /= An
            n_col = Ai.shape[1]
            y = (fnu_i/rms)[ok_band]
            LHS, RHS = Ai.T.dot(Ai) + lamb * np.identity(n_col), Ai.T.dot(y)
            #coeffs_x, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)
            coeffs_x = np.linalg.solve(LHS, RHS)#, rcond=None)
            coeffs_x /= An
            
            # coeffs_x, _, _, _ = np.linalg.lstsq(Ax[:,ok_temp], (fnu_i/rms)[ok_band],
            #                                     rcond=None)
        else:
            raise ValueError(f'fitter {fitter} not recognized')
            
        coeffs_i = np.zeros(sh[0])
        coeffs_i[ok_temp] = coeffs_x
    except:
        coeffs_i = np.zeros(sh[0])
        
    fmodel = np.dot(coeffs_i, A)
    chi2_i = np.sum((fnu_i-fmodel)**2/var*ok_band)
    
    coeffs_draw = None
    if get_err > 0:
        if fitter == 'nnls':
            ok_temp = coeffs_i > 0
            LHS = Ax[:,ok_temp]*1
            An = 1.
        else:
            ok_temp = coeffs_i != 0
        
        coeffs_draw = np.zeros((get_err, A.shape[0]))
        try:
            covar = np.matrix(np.dot(LHS.T, LHS)).I/An**2
            #covar = utils.safe_invert(arr)/An**2
            #covar = np.matrix(np.dot(Ax[:,ok_temp].T, Ax[:,ok_temp])).I
            coeffs_draw[:, ok_temp] = np.random.multivariate_normal(coeffs_i[ok_temp], covar, size=get_err)
        except:
            coeffs_draw = None
            
    return chi2_i, coeffs_i, fmodel, coeffs_draw


