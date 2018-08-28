import os
import time
import numpy as np

from collections import OrderedDict

try:
    from grizli.utils import GTable as Table
except:
    from astropy.table import Table

import astropy.io.fits as pyfits

from . import filters
from . import param 
from . import igm as igm_module
from . import templates as templates_module 
from . import utils 

__all__ = ["PhotoZ", "TemplateGrid"]

class PhotoZ(object):
    def __init__(self, param_file='zphot.param', translate_file='zphot.translate', zeropoint_file=None, load_prior=True, load_products=True, params={}, random_seed=0, n_proc=0):
                
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
        self.param = param.EazyParam(param_file, read_templates=False, read_filters=False)
        self.translate = param.TranslateFile(translate_file)
                
        if 'MW_EBV' not in self.param.params:
            self.param.params['MW_EBV'] = 0.0354 # MACS0416
            #self.param.params['MW_EBV'] = 0.0072 # GOODS-S
            #self.param['SCALE_2175_BUMP'] = 0.4 # Test
        
        for key in params:
            self.param.params[key] = params[key]
            
        ### Read templates
        self.templates = self.param.read_templates(templates_file=self.param['TEMPLATES_FILE'])
        self.NTEMP = len(self.templates)
        
        ### Set redshift fit grid
        #self.param['Z_STEP'] = 0.003
        self.get_zgrid()
        
        ### Read catalog and filters
        self.read_catalog()
        
        ### Read prior file
        self.full_prior = np.ones((self.NOBJ, self.NZ))
        if load_prior:
            self.read_prior()
        
        if zeropoint_file is not None:
            self.read_zeropoint(zeropoint_file)
        else:
            self.zp = self.f_numbers*0+1.
            
        self.fit_chi2 = np.zeros((self.NOBJ, self.NZ))
        self.fit_coeffs = np.zeros((self.NOBJ, self.NZ, self.NTEMP))
        
        ### Interpolate templates
        #self.tempfilt = TemplateGrid(self.zgrid, self.templates, self.filters, add_igm=True, galactic_ebv=0.0354)

        t0 = time.time()
        self.tempfilt = TemplateGrid(self.zgrid, self.templates, RES=self.param['FILTERS_RES'], f_numbers=self.f_numbers, add_igm=True, galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'], n_proc=n_proc)
        t1 = time.time()
        print('Process templates: {0:.3f} s'.format(t1-t0))
        
        ### Template Error
        self.get_template_error()
        
        self.ubvj = None
        
        ### Load previous products?
        if load_products:
            self.load_products()
        
        ### Flam conversion factors
        self.to_flam = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))*3.e18/1.e-19/self.lc**2/self.ext_corr
        
            
        #### testing
        if False:
            
            idx = np.arange(self.NOBJ)[self.cat['z_spec'] > 0]
            i=37
            fig = eazy.plotExampleSED(idx[i], MAIN_OUTPUT_FILE='m0416.uvista.full')
         
            obj_ix = 2480
            obj_ix = idx[i]
    
    def load_products(self):
        zout_file = '{0}.zout.fits'.format(self.param['MAIN_OUTPUT_FILE'])
        if os.path.exists(zout_file):
            print('Load products: {0}'.format(zout_file))

            data_file = '{0}.data.fits'.format(self.param['MAIN_OUTPUT_FILE'])
            data = pyfits.open(data_file)
            self.fit_chi2 = data['CHI2'].data*1
            self.compute_pz()
            self.ubvj = data['REST_UBVJ'].data*1
            
            self.zout = Table.read(zout_file)
            for iter in range(2):
                self.best_fit(zbest=self.zout['z_phot'].data, prior=False)
                self.error_residuals()
               
    def read_catalog(self, verbose=True):
        #from astropy.table import Table
               
        if verbose:
            print('Read CATALOG_FILE:', self.param['CATALOG_FILE'])
             
        if 'fits' in self.param['CATALOG_FILE'].lower():
            self.cat = Table.read(self.param['CATALOG_FILE'], format='fits')
        else:
            self.cat = Table.read(self.param['CATALOG_FILE'], format='ascii.commented_header')
        self.NOBJ = len(self.cat)
        
        all_filters = filters.FilterFile(self.param['FILTERS_RES'])
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
                
                self.filters.append(all_filters.filters[f_number-1])
                self.flux_columns.append(k)
                self.err_columns.append(ke)
                self.f_numbers.append(f_number)
                print('{0} {1} ({2:3d}): {3}'.format(k, ke, f_number, self.filters[-1].name.split()[0]))
                
                
        for k in self.translate.trans:
            fcol = self.translate.trans[k]
            if fcol.startswith('F') & ('FTOT' not in fcol):
                f_number = int(fcol[1:])
                for ke in self.translate.trans:
                    if self.translate.trans[ke] == 'E{0}'.format(f_number):
                        break
                                 
                if (k in self.cat.colnames) & (ke in self.cat.colnames):
                    self.filters.append(all_filters.filters[f_number-1])
                    self.flux_columns.append(k)
                    self.err_columns.append(ke)
                    self.f_numbers.append(f_number)
                    print('{0} {1} ({2:3d}): {3}'.format(k, ke, f_number, self.filters[-1].name.split()[0]))
                        
        self.f_numbers = np.array(self.f_numbers)
        
        self.lc = np.array([f.pivot() for f in self.filters])
        self.ext_corr = np.array([10**(0.4*f.extinction_correction(self.param.params['MW_EBV'])) for f in self.filters])
        self.zp = self.ext_corr*0+1
        
        self.NFILT = len(self.filters)
        self.fnu = np.zeros((self.NOBJ, self.NFILT))
        self.efnu = np.zeros((self.NOBJ, self.NFILT))
        
        for i in range(self.NFILT):
            self.fnu[:,i] = self.cat[self.flux_columns[i]]
            self.efnu[:,i] = self.cat[self.err_columns[i]]
            if self.err_columns[i] in self.translate.error:
                self.efnu[:,i] *= self.translate.error[self.err_columns[i]]
                
        self.efnu_orig = self.efnu*1.
        self.efnu = np.sqrt(self.efnu**2+(self.param['SYS_ERR']*self.fnu)**2)
        
        ok_data = (self.efnu > 0) & (self.fnu > self.param['NOT_OBS_THRESHOLD']) & np.isfinite(self.fnu) & np.isfinite(self.efnu)
        self.fnu[~ok_data] = -99
        self.efnu[~ok_data] = -99
        
        self.nusefilt = ok_data.sum(axis=1)
        
        # Translate the file itself    
        for k in self.translate.trans:
            if k in self.cat.colnames:
                #self.cat.rename_column(k, self.translate.trans[k])
                self.cat[self.translate.trans[k]] = self.cat[k]
        
    def read_zeropoint(self, zeropoint_file='zphot.zeropoint'):
        lines = open(zeropoint_file).readlines()
        for line in lines:
            if not line.startswith('F'):
                continue
            
            fnum = int(line.strip().split()[0][1:])
            if fnum in self.f_numbers:
                ix = self.f_numbers == fnum
                self.zp[ix] = float(line.split()[1])
                
    def get_template_error(self):
        self.TEF = templates_module.TemplateError(self.param['TEMP_ERR_FILE'], lc=self.lc, scale=self.param['TEMP_ERR_A2'])
        
        self.TEFgrid = np.zeros((self.NZ, self.NFILT))
        for i in range(self.NZ):
            self.TEFgrid[i,:] = self.TEF(self.zgrid[i])
            
    def get_zgrid(self):
        zr = [self.param['Z_MIN'], self.param['Z_MAX']]
        self.zgrid = utils.log_zgrid(zr=zr, dz=self.param['Z_STEP'])
        self.NZ = len(self.zgrid)
    
    def read_prior(self, verbose=True):
        
        if not os.path.exists(self.param['PRIOR_FILE']):
            return False
            
        prior_raw = np.loadtxt(self.param['PRIOR_FILE'])
        prior_header = open(self.param['PRIOR_FILE']).readline()
        
        self.prior_mags = np.cast[float](prior_header.split()[2:])
        self.prior_data = np.zeros((self.NZ, len(self.prior_mags)))
        for i in range(self.prior_data.shape[1]):
            self.prior_data[:,i] = np.interp(self.zgrid, prior_raw[:,0], prior_raw[:,i+1])
            
        # self.prior_z = prior_raw[:,0]
        # self.prior_data = prior_raw[:,1:]
        # self.prior_map_z = np.interp(self.zgrid, self.prior_z, np.arange(len(self.prior_z)))
        self.prior_map_z = np.arange(self.NZ)
                
        ix = self.f_numbers == int(self.param['PRIOR_FILTER'])
        if ix.sum() == 0:
            print('PRIOR_FILTER {0:d} not found in the catalog!')
            self.prior_mag_cat = np.zeros(self.NOBJ)-1
            
        else:
            self.prior_mag_cat = self.param['PRIOR_ABZP'] - 2.5*np.log10(np.squeeze(self.fnu[:,ix]))
            self.prior_mag_cat[~np.isfinite(self.prior_mag_cat)] = -1
            
            for i in range(self.NOBJ):
                if self.prior_mag_cat[i] > 0:
                    #print(i)
                    self.full_prior[i,:] = self._get_prior(self.prior_mag_cat[i])

        if verbose:
            print('Read PRIOR_FILE: ', self.param['PRIOR_FILE'])
    
    def _get_prior(self, mag):
        mag_clip = np.clip(mag, self.prior_mags[0], self.prior_mags[-1]-0.02)
        
        mag_ix = np.interp(mag_clip, self.prior_mags, np.arange(len(self.prior_mags)))
        int_mag_ix = int(mag_ix)
        f = mag_ix-int_mag_ix
        prior = np.dot(self.prior_data[:,int_mag_ix:int_mag_ix+2], [f,1-f])
        return prior
        
    def _x_get_prior(self, mag, **kwargs):
        import scipy.ndimage as nd
        mag_ix = np.interp(mag, self.prior_mags, np.arange(len(self.prior_mags)))*np.ones(self.NZ)
        prior = np.maximum(nd.map_coordinates(self.prior_data, [self.prior_map_z, mag_ix], **kwargs), 1.e-8)
        return prior
    
    def iterate_zp_templates(self, idx=None, update_templates=True, update_zeropoints=True, iter=0, n_proc=4, save_templates=False, error_residuals=False, prior=True):
        
        self.fit_parallel(idx=idx, n_proc=n_proc, prior=prior)        
        #self.best_fit()
        if error_residuals:
            self.error_residuals()
        
        fig = self.residuals(update_zeropoints=update_zeropoints,
                       ref_filter=int(self.param['PRIOR_FILTER']),
                       update_templates=update_templates,
                       Ng=(self.zbest > 0.1).sum() // 50, min_width=500)
        
        fig_file = '{0}_zp_{1:03d}.png'.format(self.param['MAIN_OUTPUT_FILE'], iter)
        fig.savefig(fig_file)
                                        
        if save_templates:
            self.save_templates()
    
    def zphot_zspec(self, zmin=0, zmax=4, axes=None, figsize=[6,7], minor=0.5, skip=2):

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        zlimits = self.pz_percentiles(percentiles=[16,84], oversample=10)
        
        dz = (self.zbest-self.cat['z_spec'])/(1+self.cat['z_spec'])
        #izbest = np.argmin(self.fit_chi2, axis=1)

        clip = (self.zbest > self.zgrid[0]) & (self.cat['z_spec'] > zmin) & (self.cat['z_spec'] <= zmax)
        
        gs = GridSpec(2,1, height_ratios=[6,1])
        if axes is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(gs[0,0])
        else:
            ax = axes[0]
            
        ax.set_title(self.param['MAIN_OUTPUT_FILE'])

        #ax = fig.add_subplot(gs[:,-1])
        #ax.scatter(np.log10(1+self.cat['z_spec'][clip]), np.log10(1+self.zbest[clip]), marker='o', alpha=0.2, color='k')
        
        yerr = np.log10(1+np.abs(zlimits.T-self.zbest))
        ax.errorbar(np.log10(1+self.cat['z_spec'][clip]), np.log10(1+self.zbest[clip]), yerr=yerr[:,clip], marker='.', alpha=0.2, color='k', linestyle='None')

        xt = np.arange(zmin,zmax+0.1, minor)
        xl = np.log10(1+xt)
        ax.plot(xl, xl, color='r', alpha=0.5)
        ax.set_xlim(xl[0], xl[-1]); ax.set_ylim(xl[0],xl[-1])
        xtl = list(xt)
        
        if skip > 0:
            for i in range(1, len(xt), skip):
                xtl[i] = ''

        ax.set_xticks(xl); ax.set_xticklabels([]);
        ax.set_yticks(xl); ax.set_yticklabels(xtl);
        ax.grid()
        ax.set_ylabel(r'$z_\mathrm{phot}$')

        sample_nmad = utils.nmad(dz[clip])
        ax.text(0.05, 0.925, r'N={0}, $\sigma$={1:.4f}'.format(clip.sum(), sample_nmad), ha='left', va='top', fontsize=10, transform=ax.transAxes)
        #ax.axis('scaled')
        
        if axes is None:
            ax = fig.add_subplot(gs[1,0])
        else:
            ax = axes[1]
            
        yerr = np.abs(zlimits.T-self.zbest)#/(1+self.cat['z_spec'])
        
        ax.errorbar(np.log10(1+self.cat['z_spec'][clip]), dz[clip], yerr=yerr[:,clip], marker='.', alpha=0.2, color='k', linestyle='None')
        
        ax.set_xticks(xl); ax.set_xticklabels(xtl);
        ax.set_xlim(xl[0], xl[-1])
        ax.set_ylim(-6*sample_nmad, 6*sample_nmad)
        ax.set_yticks([-3*sample_nmad, 0, 3*sample_nmad])
        ax.set_yticklabels([r'$-3\sigma$',r'$0$',r'$+3\sigma$'])
        ax.set_xlabel(r'$z_\mathrm{spec}$')
        ax.set_ylabel(r'$\Delta z / 1+z_\mathrm{spec}$')
        ax.grid()
        
        fig.tight_layout(pad=0.1)

        return fig
    
    def save_templates(self):
        path = os.path.dirname(self.param['TEMPLATES_FILE'])
        for templ in self.templates:
            templ_file = os.path.join('{0}/tweak_{1}'.format(path,templ.name))
            
            print('Save tweaked template {0}'.format(templ_file))
                                      
            np.savetxt(templ_file, np.array([templ.wave, templ.flux]).T,
                       fmt='%.6e')
                           
    def fit_parallel(self, idx=None, n_proc=4, verbose=True, get_best_fit=True, prior=False):

        import numpy as np
        import matplotlib.pyplot as plt
        import time
        import multiprocessing as mp
        
        if idx is None:
            idx = np.arange(self.NOBJ)
        
        # Setup
        # if False:
        #     hmag = 25-2.5*np.log10(self.cat['f_F160W'])
        #     idx = (hmag < 25) | (self.cat['z_spec'] > 0)
        #     #idx &= self.cat['bandtotal'] != 'bcg'
        # 
        #     #### GOODS-S
        #     idx = (hmag < 23) | (self.cat['z_spec'] > 0)
        #     idx = (hmag < 25) #| (self.cat['z_spec'] > 0)
        #     if np.sum(self.zp) == len(self.zp):
        #         self.zp[-3] = 0.75
        #         self.zp[38] = 0.88
        #         self.zp[37] = 0.87
        #         self.zp[19] = 1.1
        # 
        #     idx = np.arange(self.NOBJ)[idx]

        fnu_corr = self.fnu[idx,:]*self.ext_corr*self.zp
        efnu_corr = self.efnu[idx,:]*self.ext_corr*self.zp
            
        t0 = time.time()
        pool = mp.Pool(processes=n_proc)
        
        results = [pool.apply_async(_fit_vertical, (iz, self.zgrid[iz],  self.tempfilt(self.zgrid[iz]), fnu_corr, efnu_corr, self.TEF, self.zp, self.param.params['VERBOSITY'])) for iz in range(self.NZ)]

        pool.close()
        pool.join()
                
        for res in results:
            iz, chi2, coeffs = res.get(timeout=1)
            self.fit_chi2[idx,iz] = chi2
            self.fit_coeffs[idx,iz,:] = coeffs
        
        self.compute_pz(prior=prior)
        
        t1 = time.time()
        if verbose:
            print('Fit {1:.1f} s (n_proc={0}, NOBJ={2})'.format(n_proc, t1-t0, len(idx)))
        
        if get_best_fit:
            self.best_fit()
            
    def fit_object(self, iobj=0, z=0, show=False):
        """
        Fit on the redshift grid
        """
        from scipy.optimize import nnls
        #import np.linalg
                
        fnu_i = self.fnu[iobj, :]*self.ext_corr
        efnu_i = self.efnu[iobj,:]*self.ext_corr
        ok_band = (fnu_i > -90) & (efnu_i > 0)
        
        A = self.tempfilt(z)
        var = (0.0*fnu_i)**2 + efnu_i**2 + (self.TEF(z)*fnu_i)**2
        
        chi2 = np.zeros(self.NZ)
        coeffs = np.zeros((self.NZ, self.NTEMP))
        
        for iz in range(self.NZ):
            A = self.tempfilt(self.zgrid[iz])
            var = (0.0*fnu_i)**2 + efnu_i**2 + (self.TEF(zgrid[iz])*fnu_i)**2
            rms = np.sqrt(var)
            try:
                coeffs_i, rnorm = nnls((A/rms).T[ok_band,:], (fnu_i/rms)[ok_band])
            except:
                coeffs_i = np.zeros(A.shape[0])
                
            fmodel = np.dot(coeffs_i, A)
            chi2[iz] = np.sum((fnu_i-fmodel)**2/var*ok_band)
            coeffs[iz, :] = coeffs_i
        
        return iobj, chi2, coeffs
        
        if show:
            pz = np.exp(-(chi2-chi2.min())/2.)
            pz /= np.trapz(pz, self.zgrid)
            fig.axes[1].plot(self.zgrid, pz)
            fig.axes[1].set_ylim(0,pz.max()*1.05)
        
            izbest = np.argmax(pz)
            A = self.tempfilt(self.zgrid[izbest])
            ivar = 1/((0.0*fnu_i)**2 + efnu_i**2 + (self.TEF(zgrid[izbest])*fnu_i)**2)
            rms = 1./np.sqrt(ivar)
            model = np.dot(coeffs[izbest,:], A)
            flam_factor = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))*3.e18/1.e-17/self.lc**2/self.ext_corr
            fig.axes[0].scatter(self.lc, model*flam_factor, color='orange')
            fig.axes[0].errorbar(self.lc, fnu_i*flam_factor, rms*flam_factor, color='g', marker='s', linestyle='None')
    
    def best_fit(self, zbest=None, prior=False, get_err=False):
        self.fmodel = self.fnu*0.
        self.efmodel = self.fnu*0.
        
        izbest = np.argmin(self.fit_chi2, axis=1)

        self.zbest_grid = self.zgrid[izbest]
        if zbest is None:
            self.zbest, self.chi_best = self.best_redshift(prior=prior)
            # No prior, redshift at minimum chi-2
            self.zchi2, self.chi2_noprior = self.best_redshift(prior=False)
        else:
            self.zbest = zbest
        
        if (self.param['FIX_ZSPEC'] in [True, 'y', 'yes', 'Y', 'Yes']) & ('z_spec' in self.cat.colnames):
            #print('USE ZSPEC!')
            has_zsp = self.cat['z_spec'] > 0
            self.zbest[has_zsp] = self.cat['z_spec'][has_zsp]
            
        # Compute Risk function at z=zbest
        self.zbest_risk = self.compute_best_risk()
        
        fnu_corr = self.fnu*self.ext_corr*self.zp
        efnu_corr = self.efnu*self.ext_corr*self.zp
        
        self.coeffs_best = np.zeros((self.NOBJ, self.NTEMP))
        
        self.coeffs_draws = np.zeros((self.NOBJ, 100, self.NTEMP))
        
        idx = np.arange(self.NOBJ)[self.zbest > self.zgrid[0]]
        
        # Set seed
        np.random.seed(self.random_seed)
        
        for iobj in idx:
            #A = self.tempfilt(self.zgrid[izbest[iobj]])
            #self.fmodel[iobj,:] = np.dot(self.fit_coeffs[iobj, izbest[iobj],:], A) 
            zi = self.zbest[iobj]
            A = self.tempfilt(zi)
            TEFz = self.TEF(zi)
            
            fnu_i = fnu_corr[iobj, :]
            efnu_i = efnu_corr[iobj,:]
            if get_err:
                chi2, self.coeffs_best[iobj,:], self.fmodel[iobj,:], draws = _fit_obj(fnu_i, efnu_i, A, TEFz, self.zp, 100)
                if draws is None:
                    self.efmodel[iobj,:] = -1
                else:
                    #tf = self.tempfilt(zi)
                    self.efmodel[iobj,:] = np.diff(np.percentile(np.dot(draws, A), [16,84], axis=0), axis=0)/2.
                    self.coeffs_draws[iobj, :, :] = draws
            else:
                chi2, self.coeffs_best[iobj,:], self.fmodel[iobj,:], draws = _fit_obj(fnu_i, efnu_i, A, TEFz, self.zp, False)
                
    def best_redshift(self, prior=True):
        """Fit parabola to chi2 to get best minimum
        
        TBD: include prior
        """
        from scipy import polyfit, polyval
        
        if prior:
            test_chi2 = self.fit_chi2-2*np.log(self.full_prior)
        else:
            test_chi2 = self.fit_chi2
            
        #izbest0 = np.argmin(self.fit_chi2, axis=1)
        izbest = np.argmin(test_chi2, axis=1)
             
        zbest = self.zgrid[izbest]
        zbest[izbest == 0] = self.zgrid[0]
        chi_best = self.fit_chi2.min(axis=1)
        
        for iobj in range(self.NOBJ):
            iz = izbest[iobj]
            if (iz == 0) | (iz == self.NZ-1):
                continue
            
            #c = polyfit(self.zgrid[iz-1:iz+2], test_chi2[iobj, iz-1:iz+2], 2)
            c = polyfit(self.zgrid[iz-1:iz+2], self.fit_chi2[iobj, iz-1:iz+2], 2)
            
            zbest[iobj] = -c[1]/(2*c[0])
            chi_best[iobj] = polyval(c, zbest[iobj])
        
        return zbest, chi_best
    
    def error_residuals(self, verbose=True):
        """
        Force error bars to touch the best-fit model
        """
        
        if verbose:
            print('`error_residuals`: force uncertainties to match residuals')
            
        self.efnu = self.efnu_orig*1

        # residual
        r = np.abs(self.fmodel - self.fnu*self.ext_corr*self.zp)
        
        # Update where residual larger than uncertainty
        upd = (self.efnu > 0) & (self.fnu > self.param['NOT_OBS_THRESHOLD'])
        upd &= (r > self.efnu) & (self.fmodel > 0)
        upd &= np.isfinite(self.fnu) & np.isfinite(self.efnu)
        
        self.efnu[upd] = r[upd] #np.sqrt(var_new[upd])
    
    
    def check_uncertainties(self):
        import astropy.stats
        from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
        
        TEF_scale = 1.
        
        izbest = np.argmin(self.fit_chi2, axis=1)
        zbest = self.zgrid[izbest]
        
        full_err = self.efnu*0.
        teff_err = self.efnu*0
        
        teff_err = self.TEF(self.zbest[:,None])
        
        # for i in range(self.NOBJ):
        #     teff_err[i,:] = self.TEF(self.zbest[i])*TEF_scale
        
        full_err = np.sqrt(self.efnu_orig**2+(self.fnu*teff_err)**2)
            
        resid = (self.fmodel - self.fnu*self.ext_corr*self.zp)/self.fmodel
        #eresid = np.clip(full_err/self.fmodel, 0.02, 0.2)
        
        self.efnu_i = self.efnu_orig*1
        
        eresid = np.sqrt((self.efnu_i/self.fmodel)**2+self.param.params['SYS_ERR']**2)
        
        okz = (self.zbest > 0.1) & (self.zbest < 3)
        scale_errors = self.lc*0.
        
        for ifilt in range(self.NFILT):
            iok = okz & (self.efnu_orig[:,ifilt] > 0) & (self.fnu[:,ifilt] > self.param['NOT_OBS_THRESHOLD'])
            
            # Spline interp
            xw = self.lc[ifilt]/(1+self.zbest[iok])
            so = np.argsort(xw)
            #spl = UnivariateSpline(xw[so], resid[iok,ifilt][so], w=1/np.clip(eresid[iok,ifilt][so], 0.002, 0.1), s=iok.sum()*4)
            spl = LSQUnivariateSpline(xw[so], resid[iok,ifilt][so], np.exp(np.arange(np.log(xw.min()+100), np.log(xw.max()-100), 0.05)), w=1/eresid[iok,ifilt][so])#, s=10)
            
            nm = utils.nmad((resid[iok,ifilt]-spl(xw))/eresid[iok,ifilt])
            print('{3}: {0} {1:d} {2:.2f}'.format(self.flux_columns[ifilt], self.f_numbers[ifilt], nm, ifilt))
            scale_errors[ifilt] = nm
            self.efnu_i[:,ifilt] *= nm
            
            #plt.hist(resid[iok,ifilt], bins=100, range=[-3,3], alpha=0.5)
        
        # Overall average
        lcz = np.dot(1/(1+self.zbest[:, np.newaxis]), self.lc[np.newaxis,:])
        
    def residuals(self, selection=None, update_zeropoints=True, update_templates=True, ref_filter=205, Ng=40, correct_zp=True, min_width=500, NBIN=None):
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
        from astroML.sum_of_norms import sum_of_norms, norm
               
        izbest = np.argmin(self.fit_chi2, axis=1)
        zbest = self.zgrid[izbest]
                
        if selection is not None:
            idx = np.arange(self.NOBJ)[(izbest > 0) & selection]
        else:
            idx = np.arange(self.NOBJ)[izbest > 0]
            
        resid = (self.fmodel - self.fnu*self.ext_corr*self.zp)/self.fmodel+1
        eresid = (self.efnu_orig*self.ext_corr*self.zp)/self.fmodel

        sn = self.fnu/self.efnu
                
        fig = plt.figure(figsize=[16,4])
        gs = gridspec.GridSpec(1, 4)
        ax = fig.add_subplot(gs[:,:3])
        
        cmap = cm.rainbow
        cnorm = mpl.colors.Normalize(vmin=0, vmax=self.NFILT-1)
        
        so = np.argsort(self.lc)
        
        lcz = np.dot(1/(1+self.zgrid[izbest][:, np.newaxis]), self.lc[np.newaxis,:])
        clip = (sn > 3) & (self.efnu > 0) & (self.fnu > self.param['NOT_OBS_THRESHOLD']) & (resid > 0) & np.isfinite(self.fnu) & np.isfinite(self.efnu) & (self.fmodel != 0)
        xmf, ymf, ysf, Nf = utils.running_median(lcz[clip], resid[clip], NBIN=20*(self.NFILT // 2), use_median=True, use_nmad=True)
        
        if NBIN is None:
            NBIN = (self.zbest > self.zgrid[0]).sum() // (100) #*self.NFILT)
        #NBIN = 20*(self.NFILT // 2)
        
        xmf, ymf, ysf, Nf = utils.running_median(lcz[clip], resid[clip], NBIN=NBIN, use_median=True, use_nmad=True)
        
        clip = (xmf > 950) & (xmf < 7.9e4)
        xmf = xmf[clip]
        ymf = ymf[clip]
        
        if correct_zp:
            image_corrections = self.zp*0.+1
        else:
            image_corrections = 1/self.zp
            
        xmf = np.hstack((100, 600, 700, 800, 900, xmf, 8.e4, 9.e4, 1.e5, 1.1e5))
        ymf = np.hstack((1, 1, 1, 1, 1, ymf, 1., 1., 1., 1))
        
        #Ng = 40
        #w_best, rms, locs, widths = sum_of_norms(xmf, ymf, Ng, spacing='log', full_output=True)
        #w_best, rms, locs, widths = sum_of_norms(xmf, ymf, full_output=True, locs=xmf)#, widths=min_width) #, widths=np.maximum(2*np.gradient(xmf), min_width))
        
        #w_best, rms, locs, widths = sum_of_norms([0.8*xmf.min(), xmf.max()/0.8], [1, 1], Ng, spacing='log', full_output=True)
        #print(xmf.min(), xmf.max(), locs, widths)
        
        xsm = np.logspace(2.5,5,500)
        #norms = (w_best * norm(xsm[:, None], locs, widths)).sum(1)
        #ax.plot(xsm, norms, color='k', linewidth=2, alpha=0.5, zorder=10)
        
        #self.tnorm = (w_best, locs, widths, xmf.min(), xmf.max())
        
        AvgSpline = scipy.interpolate.CubicSpline(xmf, ymf)
        #ax.plot(xsm, spl(xsm), color='r')
        ax.plot(xsm, AvgSpline(xsm), color='k', linewidth=2, alpha=0.5, zorder=10)
        
        self.zp_delta = self.zp*0
        
        for i in range(self.NFILT):
            ifilt = so[i]
            clip = (izbest > 0) & (sn[:,ifilt] > 3) & (self.efnu[:,ifilt] > 0) & (self.fnu[:,ifilt] > self.param['NOT_OBS_THRESHOLD']) & (resid[:,ifilt] > 0)
            color = cmap(cnorm(i))
            
            if clip.sum() == 0:
                self.zp_delta[ifilt] = 1.
                continue
                
            xi = self.lc[ifilt]/(1+self.zgrid[izbest][clip])
            xm, ym, ys, N = utils.running_median(xi, resid[clip, ifilt], NBIN=20, use_median=True, use_nmad=True)
            
            # Normalize to overall median
            #xgi = (w_best * norm(xi[:, None], locs, widths)).sum(1)
            xgi = AvgSpline(xi)
            delta = np.median(resid[clip, ifilt]-xgi+1)
            self.zp_delta[ifilt]  = delta
            if correct_zp:
                delta_i = delta
            else:
                delta_i = 1.
                
            fname = os.path.basename(self.filters[ifilt].name.split()[0])
            if fname.count('.') > 1:
                fname = '.'.join(fname.split('.')[:-1])
            
            ax.plot(xm, ym/delta_i*image_corrections[i], color=color, alpha=0.8, label='{0:30s} {1:.3f}'.format(fname, delta_i/image_corrections[i]), linewidth=2)
            ax.fill_between(xm, ym/delta_i*image_corrections[i]-ys/np.sqrt(N), ym/delta_i*image_corrections[i]+ys/np.sqrt(N), color=color, alpha=0.1) 
                    
        ax.semilogx()
        ax.set_ylim(0.8,1.2)
        ax.set_xlim(800,8.e4)
        l = ax.legend(fontsize=8, ncol=5, loc='upper right')
        l.set_zorder(-20)
        ax.grid()
        ax.vlines([2175, 3727, 5007, 6563.], 0.8, 1.0, linestyle='--', color='k', zorder=-18)
        ax.set_xlabel(r'$\lambda_\mathrm{rest}$')
        ax.set_ylabel('(temp - phot) / temp')
        
        ## zphot-zspec
        dz = (self.zbest-self.cat['z_spec'])/(1+self.cat['z_spec'])
        clip = (izbest > 0) & (self.cat['z_spec'] > 0)
        
        ax = fig.add_subplot(gs[:,-1])
        ax.scatter(np.log10(1+self.cat['z_spec'][clip]), np.log10(1+self.zbest[clip]), marker='.', alpha=0.15, color='k')
        xt = np.arange(0,4.1, 0.5)
        xl = np.log10(1+xt)
        ax.plot(xl, xl, color='r', alpha=0.5)
        ax.set_xlim(0, xl[-1]); ax.set_ylim(0,xl[-1])
        xtl = list(xt)
        for i in range(1,len(xt),2):
            xtl[i] = ''
        
        ax.set_xticks(xl); ax.set_xticklabels(xtl);
        ax.set_yticks(xl); ax.set_yticklabels(xtl);
        ax.grid()
        ax.set_xlabel(r'$z_\mathrm{spec}$')
        ax.set_ylabel(r'$z_\mathrm{phot}$')
        
        ax.text(0.05, 0.925, r'N={0}, $\sigma$={1:.4f}'.format(clip.sum(), utils.nmad(dz[clip])), ha='left', va='top', fontsize=10, transform=ax.transAxes)
        fig.tight_layout(pad=0.1)
        
        # update zeropoints in self.zp
        if update_zeropoints:
            ref_ix = self.f_numbers == ref_filter
            self.zp *= self.zp_delta/self.zp_delta[ref_ix]
            self.zp[self.zp_delta == 1] = 1.
            
        # tweak templates
        if update_templates:
            print('Reprocess tweaked templates')
            #w_best, locs, widths, xmin, xmax = self.tnorm
            for itemp in range(self.NTEMP):
                templ = self.templates[itemp]
                #templ_tweak = (w_best * norm(templ.wave[:, None], locs, widths)).sum(1)
                templ_tweak = AvgSpline(templ.wave)
                templ_tweak[templ.wave > 1.e5] = 1.
                
                #templ_tweak[(templ.wave < xmin) | (templ.wave > xmax)] = 1
                templ.flux /= templ_tweak
                templ.flux_fnu /= templ_tweak
            
            # Recompute filter fluxes from tweaked templates    
            self.tempfilt = TemplateGrid(self.zgrid, self.templates, RES=self.param['FILTERS_RES'], f_numbers=self.f_numbers, add_igm=True, galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'], n_proc=0)
        
        return fig
        
    def write_zeropoint_file(self, file='zphot.zeropoint.x'):
        fp = open(file,'w')
        for i in range(self.NFILT):
            fp.write('F{0:<3d}  {1:.4f}\n'.format(self.f_numbers[i], self.zp[i]))
        
        fp.close()
    
    def full_sed(self, z, coeffs_i):
        import astropy.units as u
        import specutils.extinction
        
        templ = self.templates[0]
        tempflux = np.zeros((self.NTEMP, templ.wave.shape[0]))
        for i in range(self.NTEMP):
            tempflux[i, :] = self.templates[i].flux_fnu
            
        templz = templ.wave*(1+z)
        
        if self.tempfilt.add_igm:
            igmz = templ.wave*0.+1
            lyman = templ.wave < 1300
            igmz[lyman] = igm_module.Inoue14().full_IGM(z, templz[lyman])
        else:
            igmz = 1.
        
        templf = np.dot(coeffs_i, tempflux)*igmz
        return templz, templf
        
    def show_fit(self, id, show_fnu=False, xlim=[0.3, 9], get_spec=False, id_is_idx=False, show_components=False, zshow=None):
        """
        Show SED and p(z) of a single object
        
        Parameters
        ----------
        id : int
            Object ID corresponding to columns in `self.cat['id']`.  Or if
            `id_is_idx` is set to True, then is zero-index of the desired 
            object in the catalog array.
        
        show_fnu : bool, int
            If False, then make plots in f-lambda units of 1e-19 erg/s/cm2/A.
            
            If `show_fnu == 1`, then plot f-nu units of micro-Jansky
            
            If `show_fnu == 2`, then plot "nu-Fnu" units of micro-Jansky/micron.
        
        xlim : list
            Wavelength limits to plot
        
        get_spec : bool
            If True, return the SED data rather than make a plot
        
        id_is_idx : bool
            See `id`.
        
        show_components : bool
            Show all of the individual SED components, along with their 
            combination.
        
        zshow : None, float
            If a value is supplied, compute the best-fit SED at this redshift,
            rather than the value in the `self.zbest` array.
        
        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object
            
        """
        import matplotlib.pyplot as plt
        import astropy.units as u
        
        if False:
            ids = self.cat['id'][(self.cat['z_spec'] > 1.9) & (self.zbest > 0.8)]

            ids = self.cat['id'][(self.cat['z_spec'] > 0.2) & (self.zbest > 0.2)]

            ids = self.cat['id'][(self.zbest > 1.7)]

            j = -1
            
            j+=1; id = ids[j]; self.show_fit(id, show_fnu=show_fnu)
            
        if id_is_idx:
            ix = id
            z = self.zbest[ix]
        else:
            ix = self.cat['id'] == id
            z = self.zbest[ix][0]
        
        if zshow is not None:
            z = zshow
            
        ## SED
        A = np.squeeze(self.tempfilt(z))
        fnu_i = np.squeeze(self.fnu[ix, :])*self.ext_corr*self.zp
        efnu_i = np.squeeze(self.efnu[ix,:])*self.ext_corr*self.zp
        
        ok_band = (fnu_i/self.zp > -90) & (efnu_i/self.zp > 0)
        
        chi2_i, coeffs_i, fmodel, draws = _fit_obj(fnu_i, efnu_i, A, self.TEF(self.zbest[ix]), self.zp, 100)
        if draws is None:
            efmodel = None
        else:
            #tf = self.tempfilt(zi)
            efmodel = np.squeeze(np.diff(np.percentile(np.dot(draws, A), [16,84], axis=0), axis=0)/2.)
            
        #templz, templf = self.full_sed(self.zbest[ix][0], coeffs_i)
        if True:
            templ = self.templates[0]
            tempflux = np.zeros((self.NTEMP, templ.wave.shape[0]))
            for i in range(self.NTEMP):
                try:
                    tempflux[i, :] = self.templates[i].flux_fnu
                except:
                    tempflux[i, :] = np.interp(templ.wave, self.templates[i].wave, self.templates[i].flux_fnu)
                    
            templz = templ.wave*(1+z)

            if self.tempfilt.add_igm:
                igmz = templ.wave*0.+1
                lyman = templ.wave < 1300
                igmz[lyman] = igm_module.Inoue14().full_IGM(z, templz[lyman])
            else:
                igmz = 1.

            templf = np.dot(coeffs_i, tempflux)*igmz
            if draws is not None:
                templf_draws = np.dot(draws, tempflux)*igmz
                
        fnu_factor = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
        
        if show_fnu:
            if show_fnu == 2:
                flam_spec = 1.e29/(templz/1.e4)
                flam_sed = 1.e29/self.ext_corr/(self.lc/1.e4)
                ylabel = (r'$f_\nu / \lambda$ [$\mu$Jy / $\mu$m]')
                flux_unit = u.uJy / u.micron
            else:
                flam_spec = 1.e29
                flam_sed = 1.e29
                ylabel = (r'$f_\nu$ [$\mu$Jy]')    
                flux_unit = u.uJy
        else:
            flam_spec = 3.e18/templz**2/1.e-19
            flam_sed = 3.e18/self.lc**2/self.ext_corr/1.e-19
            ylabel = (r'$f_\lambda\times10^{-19}$ cgs')
            
            flux_unit = 1.e-19*u.erg/u.s/u.cm**2/u.AA
            
        if get_spec:
            
            data = OrderedDict(lc=self.lc, model=fmodel*fnu_factor*flam_sed,
                               emodel=efmodel*fnu_factor*flam_sed,
                               fobs=fnu_i*fnu_factor*flam_sed, 
                               efobs=efnu_i*fnu_factor*flam_sed,
                               templz=templz,
                               templf=templf*fnu_factor*flam_spec,
                               unit=show_fnu*1,
                               flux_unit=flux_unit,
                               wave_unit=u.AA)
                               
            return data
        
        
        fig = plt.figure(figsize=[8,4])
        ax = fig.add_subplot(121)
        
        ax.set_ylabel(ylabel)
            
        if efmodel is None:
            ax.scatter(self.lc/1.e4, fmodel*fnu_factor*flam_sed, color='r')
        else:
            ax.errorbar(self.lc/1.e4, fmodel*fnu_factor*flam_sed, efmodel*fnu_factor*flam_sed, color='r', marker='o', linestyle='None')
        
        missing = (fnu_i < -90) | (efnu_i < -90)
        
        ax.errorbar(self.lc[~missing]/1.e4, (fnu_i*fnu_factor*flam_sed)[~missing], (efnu_i*fnu_factor*flam_sed)[~missing], color='k', marker='s', linestyle='None')
        ax.errorbar(self.lc[missing]/1.e4, (fnu_i*fnu_factor*flam_sed)[missing], (efnu_i*fnu_factor*flam_sed)[missing], color='0.5', marker='s', linestyle='None', alpha=0.4)
        
        pl = ax.plot(templz/1.e4, templf*fnu_factor*flam_spec, alpha=0.5, zorder=-1)
        
        if show_components:
            for i in range(self.NTEMP):
                pi = ax.plot(templz/1.e4, coeffs_i[i]*tempflux[i,:]*igmz*fnu_factor*flam_spec, alpha=0.5, zorder=-1)
                
        if draws is not None:
            templf_width = np.percentile(templf_draws*fnu_factor*flam_spec, [16,84], axis=0)
            ax.fill_between(templz/1.e4, templf_width[0,:], templf_width[1,:], color=pl[0].get_color(), alpha=0.1)
            
        ax.set_xlim(xlim)
        xt = np.array([0.5, 1, 2, 4])*1.e4
        
        ymax = (fmodel*fnu_factor*flam_sed).max()
        ax.set_ylim(-0.1*ymax, 1.2*ymax)
        ax.semilogx()

        ax.set_xticks(xt/1.e4)
        ax.set_xticklabels(xt/1.e4)

        ax.set_xlabel(r'$\lambda_\mathrm{obs}$')
        
        ## P(z)
        
        ax = fig.add_subplot(122)
        chi2 = np.squeeze(self.fit_chi2[ix,:])
        prior = self.full_prior[ix,:].flatten()
        #pz = np.exp(-(chi2-chi2.min())/2.)*prior
        #pz /= np.trapz(pz, self.zgrid)
        pz = self.pz[ix,:].flatten()
        
        ax.plot(self.zgrid, pz, color='orange')
        ax.plot(self.zgrid, prior/prior.max()*pz.max(), color='g')
        
        ax.fill_between(self.zgrid, pz, pz*0, color='yellow', alpha=0.5)
        if self.cat['z_spec'][ix] > 0:
            ax.vlines(self.cat['z_spec'][ix], 1.e-5, pz.max()*1.05, color='r')
        
        if zshow is not None:
            ax.vlines(zshow, 1.e-5, pz.max()*1.05, color='purple')
            
        ax.set_ylim(0,pz.max()*1.05)
        ax.set_xlim(0,self.zgrid[-1])
            
        ax.set_xlabel('z'); ax.set_ylabel('p(z)/dz')
        ax.grid()
        
        fig.tight_layout(pad=0.5)
        
        return fig
        
    def rest_frame_fluxes(self, f_numbers=[153,155,161], pad_width=0.5, percentiles=[2.5,16,50,84,97.5]):
        """
        Rest-frame colors
        """
        print('Rest-frame filters: {0}'.format(f_numbers))
        rf_tempfilt = TemplateGrid(np.array([0,0.1]), self.templates, RES=self.param['FILTERS_RES'], f_numbers=np.array(f_numbers), add_igm=False, galactic_ebv=0, Eb=self.param['SCALE_2175_BUMP'], n_proc=-1)
        rf_tempfilt.tempfilt = np.squeeze(rf_tempfilt.tempfilt[0,:,:])
        
        NREST = len(f_numbers)
        
        fnu_corr = self.fnu*self.ext_corr*self.zp
        efnu_corr = self.efnu*self.ext_corr*self.zp
        
        f_rest = np.zeros((self.NOBJ, NREST, len(percentiles)))
        
        fnu_factor = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
        
        ### Testing
        if False:
            j+=1; id = ids[j]; fig = self.show_fit(id, show_fnu=show_fnu, xlim=[0.1,10])
        
            ix = np.arange(self.NOBJ)[self.cat['id'] == id][0]
            z = self.zbest[ix]
            iz = np.argmin(self.fit_chi2[ix,:])
                        
        indices = np.where(self.zbest > self.zgrid[0])[0]
        for ix in indices:

            fnu_i = fnu_corr[ix,:]*1
            efnu_i = efnu_corr[ix,:]*1
            z = self.zbest[ix]
            A = self.tempfilt(z)   
        
            for i in range(NREST):
                ## Grow uncertainties away from RF band
                lc_i = rf_tempfilt.lc[i]
                grow = np.exp(-(lc_i-self.lc/(1+z))**2/2/(pad_width*lc_i)**2)
                TEFz = (2/(1+grow/grow.max())-1)*0.5
            
                chi2_i, coeffs_i, fmodel_i, draws = _fit_obj(fnu_i, efnu_i, A, TEFz, self.zp, 100)
                if draws is None:
                    f_rest[ix,i,:] = np.zeros(len(percentiles))-1
                else:
                    f_rest[ix,i,:] = np.percentile(np.dot(draws, rf_tempfilt.tempfilt), percentiles, axis=0)[:,i]
        
        return rf_tempfilt, f_rest    
        
        flam_sed = 3.e18/(rf_tempfilt.lc*(1+z))**2/1.e-19
        fig.axes[0].errorbar(rf_tempfilt.lc*(1+z)/1.e4, f_rest_pad*fnu_factor*flam_sed, f_rest_err*fnu_factor*flam_sed, color='g', marker='s', linestyle='None')
        
        # covar_rms = np.zeros(NREST)
        # covar_med = np.zeros(NREST)
        # 
        # chain_rms = np.zeros(NREST)
        # chain_med = np.zeros(NREST)
        # 
        # if False:
        #     ### Redo here
        #     ok_band = (fnu_i > -90) & (efnu_i > 0)
        #     var = efnu_i**2 + (TEFz*fnu_i)**2
        #     rms = np.sqrt(var)
        #     
        #     #ok_band &= fnu_i > 2*rms
        #     
        #     coeffs_i, rnorm = nnls((A/rms).T[ok_band,:], (fnu_i/rms)[ok_band])
        #     fmodel = np.dot(coeffs_i, A)
        #     chi2_i = np.sum((fnu_i-fmodel)**2/var*ok_band)
        #     
        #     ok_temp = (np.sum(A, axis=1) > 0) & (coeffs_i != 0)
        #     ok_temp = np.isfinite(coeffs_i)
        #     Ax = A[:, ok_band][ok_temp,:].T
        #     Ax *= 1/rms[ok_band][:, np.newaxis]
        # 
        #     try:
        #         covar = np.matrix(np.dot(Ax.T, Ax)).I
        #         covard = np.array(np.sqrt(covar.diagonal())).flatten()
        #     except:
        #         covard = np.zeros(ok_temp.sum())#-1.
        #     
        #     coeffs_err = coeffs_i*0.
        #     coeffs_err[ok_temp] += covard
        #     
        #     draws = np.random.multivariate_normal(coeffs_i, covar, size=100)
        #     covar_rms[i] = np.std(np.dot(draws, rf_tempfilt.tempfilt), axis=0)[i]
        #     covar_med[i] = np.median(np.dot(draws, rf_tempfilt.tempfilt), axis=0)[i]
        #     
        #     f_rest_err[i] = np.sqrt(np.dot(coeffs_err**2, rf_tempfilt.tempfilt**2))[i]
        #     
        #     fmodel_err = np.sqrt(np.dot(coeffs_err**2, A**2))
        #     
        #     NWALKERS = ok_temp.sum()*4
        #     NSTEP = 1000
        #     p0 = [coeffs_i+np.random.normal(size=ok_temp.sum())*covard for w in range(NWALKERS)]
        #     
        #     obj_fun = _obj_nnls
        #     obj_args = [A[:,ok_band], fnu_i[ok_band], rms[ok_band]]
        # 
        #     ndim = len(coeffs_i)
        # 
        #     NTHREADS, NSTEP = 1, 1000
        #     sampler = emcee.EnsembleSampler(NWALKERS, ndim, obj_fun, args = obj_args, threads=NTHREADS)
        # 
        #     t0 = time.time()
        #     result = sampler.run_mcmc(p0, NSTEP)
        #     t1 = time.time()
        #     print 'Sampler: %.1f s' %(t1-t0)
        # 
        #     chain = unicorn.interlace_fit.emceeChain(chain=sampler.chain)
        #     draws = chain.draw_random(100)
        #     chain_rms[i] = np.std(np.dot(draws, rf_tempfilt.tempfilt), axis=0)[i]
        #     chain_med[i] = np.median(np.dot(draws, rf_tempfilt.tempfilt), axis=0)[i]
    
    def compute_pz(self, prior=False):
        pz = np.exp(-(self.fit_chi2.T-self.fit_chi2.min(axis=1))/2.).T
        if prior:
            pz *= self.full_prior
        
        dz = np.gradient(self.zgrid)
        norm = (pz*dz).sum(axis=1)
        self.pz = (pz.T/norm).T
    
    def compute_full_risk(self):
        
        dz = np.gradient(self.zgrid)
        
        zsq = np.dot(self.zgrid[:,None], np.ones_like(self.zgrid)[None,:])
        L = self._loss((zsq-self.zgrid)/(1+self.zgrid))
        
        pzi = self.pz[0,:]
        Rz = self.pz*0.
        
        hasz = self.zbest > 0
        idx = np.arange(self.NOBJ)[hasz]
        
        for i in idx:
            Rz[i,:] = np.dot(self.pz[i,:]*L, dz)
        
        return Rz
        
        #self.full_risk = Rz
        #self.min_risk = self.zgrid[np.argmin(Rz, axis=1)]
        
    def compute_best_risk(self):
        """
        "Risk" function from Tanaka et al. 2017
        """
        zbest_grid = np.dot(self.zbest[:,None], np.ones_like(self.zgrid)[None,:])
        L = self._loss((zbest_grid-self.zgrid)/(1+self.zgrid))
        dz = np.gradient(self.zgrid)
        
        zbest_risk = np.dot(self.pz*L, dz)
        zbest_risk[self.zbest < 0] = -1
        return zbest_risk
        
    @staticmethod    
    def _loss(dz, gamma=0.15):
        return 1-1/(1+(dz/gamma)**2)
    
    def PIT(self, zspec):
        """
        PIT function for evaluating the calibration of p(z), 
        as described in Tanaka (2017).
        """
        zspec_grid = np.dot(zspec[:,None], np.ones_like(self.zgrid)[None,:])
        zlim = zspec_grid >= self.zgrid
        dz = np.gradient(self.zgrid)
        PIT = np.dot(self.pz*zlim, dz)

        return PIT
        
        
    def pz_percentiles(self, percentiles=[2.5,16,50,84,97.5], oversample=10):
        
        import scipy.interpolate 
        
        zr = [self.param['Z_MIN'], self.param['Z_MAX']]
        zgrid_zoom = utils.log_zgrid(zr=zr,dz=self.param['Z_STEP']/oversample)
         

        ok = self.zbest > self.zgrid[0]      
        
        spl = scipy.interpolate.Akima1DInterpolator(self.zgrid, self.pz[ok,:], axis=1)
        dz_zoom = np.gradient(zgrid_zoom)
        pzcum = np.cumsum(spl(zgrid_zoom)*dz_zoom, axis=1)
        
        zlimits = np.zeros((self.NOBJ, len(percentiles)))
        Np = len(percentiles)
        for j, i in enumerate(np.arange(self.NOBJ)[ok]):
            zlimits[i,:] = np.interp(np.array(percentiles)/100., pzcum[j, :], zgrid_zoom)
        
        return zlimits
    
    def find_peaks(self):
        import peakutils
        
        ok = self.zbest > self.zgrid[0]      

        peaks = [0]*self.NOBJ
        numpeaks = np.zeros(self.NOBJ, dtype=int)

        for j, i in enumerate(np.arange(self.NOBJ)[ok]):
            indices = peakutils.indexes(self.pz[i,:], thres=0.8, min_dist=int(0.1/self.param['Z_STEP']))
            peaks[i] = indices
            numpeaks[i] = len(indices)
        
        return peaks, numpeaks
        
    def sps_parameters(self, UBVJ=[153,154,155,161], LIR_wave=[8,1000], cosmology=None):
        """
        Rest-frame colors, for tweak_fsps_temp_kc13_12_001 templates.
        """        
        import astropy.units as u
        import astropy.constants as const
        
        if cosmology is None:
            from astropy.cosmology import Planck15 as cosmology
            
        self.ubvj_tempfilt, self.ubvj = self.rest_frame_fluxes(f_numbers=UBVJ, pad_width=0.5, percentiles=[2.5,16,50,84,97.5]) 
        
        restU = self.ubvj[:,0,2]
        restB = self.ubvj[:,1,2]
        restV = self.ubvj[:,2,2]
        restJ = self.ubvj[:,3,2]

        errU = (self.ubvj[:,0,3] - self.ubvj[:,0,1])/2.
        errB = (self.ubvj[:,1,3] - self.ubvj[:,1,1])/2.
        errV = (self.ubvj[:,2,3] - self.ubvj[:,2,1])/2.
        errJ = (self.ubvj[:,3,3] - self.ubvj[:,3,1])/2.
        
        #PARAM_FILE = os.path.join(os.path.dirname(__file__), 'data/spectra_kc13_12_tweak.params')
        #temp_MLv, temp_SFRv = np.loadtxt(PARAM_FILE, unpack=True)
        
        tab_temp = Table.read(self.param['TEMPLATES_FILE']+'.fits')
        temp_MLv = tab_temp['mass']/tab_temp['Lv']
        temp_SFRv = tab_temp['sfr']
        
        # Normalize fit coefficients to template V-band
        coeffs_norm = self.coeffs_best*self.ubvj_tempfilt.tempfilt[:,2]
        
        # Normalize fit coefficients to unity sum
        coeffs_norm = (coeffs_norm.T/coeffs_norm.sum(axis=1)).T

        coeffs_orig = (self.coeffs_best.T/self.coeffs_best.sum(axis=1)).T
        
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
        
        # Mass & SFR, normalize to V band and then scale by V luminosity
        #MLv = (coeffs_norm*temp_MLv).sum(axis=1)*u.solMass/u.solLum
        mass_norm = (coeffs_norm*tab_temp['mass']).sum(axis=1)*u.solMass
        Lv_norm = (coeffs_norm*tab_temp['Lv']).sum(axis=1)*u.solLum
        MLv = mass_norm / Lv_norm
        
        LIR_norm = (coeffs_norm*tab_temp['LIR']).sum(axis=1)*u.solLum
        LIRv = LIR_norm / Lv_norm
        
        # Comute LIR directly from templates as tab_temp['LIR'] was 8-100 um
        templ_LIR = np.zeros(self.NTEMP)
        for j in range(self.NTEMP):
            templ = self.templates[j]
            clip = (templ.wave > LIR_wave[0]*1e4) & (templ.wave < LIR_wave[1]*1e4)
            templ_LIR[j] = np.trapz(templ.flux[clip], templ.wave[clip])
        
        LIR_norm = (coeffs_norm*templ_LIR).sum(axis=1)*u.solLum
        LIRv = LIR_norm / Lv_norm
         
        SFR_norm = (coeffs_norm*tab_temp['sfr']).sum(axis=1)*u.solMass/u.yr
        SFRv = SFR_norm / Lv_norm
        
        # Convert observed maggies to fnu
        uJy_to_cgs = u.Jy.to(u.erg/u.s/u.cm**2/u.Hz)*1.e-6
        fnu_scl = 10**(-0.4*(self.param.params['PRIOR_ABZP']-23.9))*uJy_to_cgs
        
        fnu = restV*fnu_scl*(u.erg/u.s/u.cm**2/u.Hz)
        dL = cosmology.luminosity_distance(self.zbest).to(u.cm)
        Lnu = fnu*4*np.pi*dL**2
        pivotV = self.ubvj_tempfilt.filters[2].pivot()*u.Angstrom*(1+self.zbest)
        nuV = (const.c/pivotV).to(u.Hz) 
        Lv = (nuV*Lnu).to(u.L_sun)
                
        mass = MLv*Lv
        SFR = SFRv*Lv
        LIR = LIRv*Lv
        
        # Emission line fluxes
        line_flux = {}
        line_EW = {}
        emission_lines = ['Ha', 'O3', 'Hb', 'O2', 'Lya']
        
        fnu_factor = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
        
        for line in emission_lines:
            line_flux_norm = (self.coeffs_best*tab_temp['line_flux_{0}'.format(line)]).sum(axis=1)
            line_cont_norm = (self.coeffs_best*tab_temp['line_C_{0}'.format(line)]).sum(axis=1)
            line_EW[line] = line_flux_norm/line_cont_norm*u.AA
            line_flux[line] = line_flux_norm*fnu_factor/(1+self.zbest)*u.erg/u.second/u.cm**2
            
        if False:
            BVx = -2.5*np.log10(restB/restV)
            plt.scatter(BVx[sample], np.log10(MLv.data)[sample], alpha=0.02, color='k', marker='s')
            # Taylor parameterization
            x = np.arange(-0.5,2,0.1)
            plt.plot(x, -0.734 + 1.404*(x+0.084), color='r')
            
        #sSFR = SFR/mass
        
        tab = Table()
        
        tab['restU'] = restU
        tab['restU_err'] = errU
        tab['restB'] = restB
        tab['restB_err'] = errB
        tab['restV'] = restV
        tab['restV_err'] = errV
        tab['restJ'] = restJ
        tab['restJ_err'] = errJ

        tab['Lv'] = Lv
        tab['MLv'] = MLv
        tab['Av'] = Av
        
        for col in tab.colnames:
            tab[col].format = '.3f'
            
        tab['Lv'].format = '.3e'
        
        tab['mass'] = mass
        tab['mass'].format = '.3e'

        tab['SFR'] = SFR
        tab['SFR'].format = '.3e'
        
        tab['LIR'] = LIR
        tab['LIR'].format = '.3e'
        
        for line in emission_lines:
            tab['line_flux_{0}'.format(line)] = line_flux[line]
            tab['line_EW_{0}'.format(line)] = line_EW[line]
            
        tab.meta['FNUSCALE'] = (fnu_scl, 'Scale factor to f-nu CGS')
        
        return tab

    def standard_output(self, prior=True, UBVJ=[153,154,155,161], cosmology=None, LIR_wave=[8, 1000]):
        import astropy.io.fits as pyfits
        
        self.compute_pz(prior=prior)
        self.best_fit(prior=prior)
        
        peaks, numpeaks = self.find_peaks()
        zlimits = self.pz_percentiles(percentiles=[2.5,16,50,84,97.5], oversample=10)
        
        tab = Table()
        tab['id'] = self.cat['id']
        tab['z_spec'] = self.cat['z_spec']
        tab['nusefilt'] = self.nusefilt
        
        tab['numpeaks'] = numpeaks
        tab['z_phot'] = self.zbest
        tab['z_phot_chi2'] = self.chi_best #fit_chi2.min(axis=1)
        tab['z_phot_risk'] = self.zbest_risk
        
        self.Rz = self.compute_full_risk()
        tab['z_min_risk'] = self.zgrid[np.argmin(self.Rz, axis=1)]
        tab['min_risk'] = self.Rz.min(axis=1)
                
        tab['z_chi2_noprior'] = self.zchi2
        tab['chi2_noprior'] = self.chi2_noprior
        
        tab['z025'] = zlimits[:,0]
        tab['z160'] = zlimits[:,1]
        tab['z500'] = zlimits[:,2]
        tab['z840'] = zlimits[:,3]
        tab['z975'] = zlimits[:,4]
        
        for col in tab.colnames[-6:]:
            tab[col].format='8.4f'
            
        tab.meta['prior'] = (prior, 'Prior applied ({0})'.format(self.param.params['PRIOR_FILE']))
        
        sps_tab = self.sps_parameters(UBVJ=UBVJ, cosmology=cosmology, LIR_wave=LIR_wave)
        for col in sps_tab.colnames:
            tab[col] = sps_tab[col]
        
        for key in sps_tab.meta:
            tab.meta[key] = sps_tab.meta[key]
        
        root = self.param.params['MAIN_OUTPUT_FILE']
        if os.path.exists('{0}.zout.fits'.format(root)):
            os.remove('{0}.zout.fits'.format(root))
        
        tab.write('{0}.zout.fits'.format(root), format='fits')
        
        self.param.write('{0}.zphot.param'.format(root))
        self.write_zeropoint_file('{0}.zphot.zeropoint'.format(root))
        self.translate.write('{0}.zphot.translate'.format(root))
        
        hdu = pyfits.HDUList(pyfits.PrimaryHDU())
        hdu.append(pyfits.ImageHDU(self.cat['id'].astype(np.uint32), name='ID'))
        hdu.append(pyfits.ImageHDU(self.zbest.astype(np.float32), name='ZBEST'))
        hdu.append(pyfits.ImageHDU(self.zgrid.astype(np.float32), name='ZGRID'))
        hdu.append(pyfits.ImageHDU(self.fit_chi2.astype(np.float32), name='CHI2'))
        hdu[-1].header['PRIOR'] = (prior, 'Prior applied ({0})'.format(self.param.params['PRIOR_FILE']))
        
        # Template coefficients 
        hdu.append(pyfits.ImageHDU(self.coeffs_best, name='COEFFS'))
        h = hdu[-1].header
        h['ABZP'] = (self.param['PRIOR_ABZP'], 'AB zeropoint')
        h['NTEMP'] = (self.NTEMP, 'Number of templates')
        for i, t in enumerate(self.templates):
            h['TEMP{0:04d}'.format(i)] = t.name
        
        hdu.append(pyfits.ImageHDU(self.templates[0].wave, name='TEMPL'))
        temp_flux = np.zeros((self.NTEMP, len(self.templates[0].wave)))
        for i in range(self.NTEMP):
            temp_flux[i,:] = self.templates[i].flux
            
        hdu.append(pyfits.ImageHDU(temp_flux, name='TEMPF'))
        
        # Rest-frame fluxes
        hdu.append(pyfits.ImageHDU(self.ubvj.astype(np.float32), name='REST_UBVJ'))
        hdu[-1].header['RESFILE'] = (self.param['FILTERS_RES'], 'Filter file')
        hdu[-1].header['UFILT'] = (UBVJ[0], 'U-band filter ID')
        hdu[-1].header['BFILT'] = (UBVJ[1], 'B-band filter ID')
        hdu[-1].header['VFILT'] = (UBVJ[2], 'V-band filter ID')
        hdu[-1].header['JFILT'] = (UBVJ[3], 'J-band filter ID')
        
        hdu.writeto('{0}.data.fits'.format(root), clobber=True)
    
    def get_match_index(self, id=None, rd=None, verbose=True):
        import astropy.units as u
        
        if id is not None:
            ix = np.where(self.cat['id'] == id)[0][0]
            if verbose:
                print('ID={0}, idx={1}'.format(id, ix))
                
            return ix
        
        # From RA / DEC
        idx, dr = self.cat.match_to_catalog_sky([rd[0]*u.deg, rd[1]*u.deg])
        if verbose:
            print('ID={0}, idx={1}, dr={2:.3f}'.format(self.cat['id'][idx[0]], 
                                                   idx[0], dr[0]))
            
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
        dq = (self.fnu[ix,:] > -90) & (self.efnu[ix,:] > 0) & np.isfinite(self.fnu[ix,:]) & np.isfinite(self.efnu[ix,:])
        
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
        from collections import OrderedDict
        from grizli import utils
        import astropy.units as u
        
        if grizli_templates is not None:
            template_list = [templates_module.Template(arrays=(grizli_templates[k].wave, grizli_templates[k].flux), name=k) for k in grizli_templates]
            
            tempfilt = TemplateGrid(self.zgrid, template_list, RES=self.param['FILTERS_RES'], f_numbers=self.f_numbers, add_igm=True, galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'])
        else:
            tempfilt = None
        
        if rd is not None:
            ti = utils.GTable()
            ti['ra'] = [rd[0]]
            ti['dec'] = [rd[1]]
            
            idx, dr = self.cat.match_to_catalog_sky(ti)
            idx = idx[0]
            dr = dr[0]
        else:
            idx = np.where(self.cat['id'] == id)[0][0]
            dr = 0
        
        notobs_mask =  self.fnu[idx,:] < -90
        sed = self.show_fit(idx, show_fnu=False, xlim=[0.3, 9], get_spec=True, id_is_idx=True)
        sed['fobs'][notobs_mask] = -99
        sed['efobs'][notobs_mask] = -99
        
        photom = OrderedDict()
        photom['flam'] = sed['fobs']*1.e-19
        photom['eflam'] = sed['efobs']*1.e-19
        photom['filters'] = self.filters
        photom['tempfilt'] = tempfilt
        photom['pz'] = self.zgrid, self.pz[idx,:]
        
        return photom, self.cat['id'][idx], dr
        
    def rest_frame_SED(self, idx=None, norm_band=155, c='k'):
        
        import matplotlib.pyplot as plt
        
        if False:
            ok = (zout['z_phot'] > 0.4) & (zout['z_phot'] < 2)
            col = (VJ < 1.5) & (UV > 1.5)
            # Quiescent
            idx = col & ok & (np.log10(sSFR) < -11.5)
            idx = col & ok & (np.log10(sSFR) > -10.5)
            idx = col & ok & (np.log10(sSFR) > -9.5)

            idx = ok & (VJ > 1.8)
             
            ## Red
            UWise = f_rest[:,0,2]/f_rest[:,2,2]
            idx, label, c = ok & (np.log10(UWise) > -1) & (np.log10(sSFR) > -10), 'U22_blue', 'b'

            idx, label, c = ok & (np.log10(UWise) < -1.8) & (np.log10(UWise) > -2.2) & (np.log10(sSFR) > -10), 'U22_mid', 'g'

            idx, label, c = ok & (np.log10(UWise) < -2.4) & (np.log10(sSFR) > -10), 'U22_red', 'r'
            
            # Quiescent
            idx, label, c = ok & (np.log10(zout['MLv']) > 0.4) & (np.log10(sSFR) < -11.9), 'Q', 'r'
            
            # Dusty
            idx, label, c = ok & (np.log10(zout['MLv']) > 0.6) & (np.log10(sSFR) < -10.5), 'MLv_lo', 'brown'

            idx, label, c = ok & (np.log10(zout['MLv']) > 0.6) & (np.abs(np.log10(sSFR)+10.5) < 0.5), 'MLv_mid', 'k'

            idx, label, c = ok & (np.log10(zout['MLv']) > 0.6) & (np.log10(sSFR) > -9.5), 'MLv_hi', 'green'
            
            # post-SB    
            #idx, label, c = (UV < 1.6) & ok & (np.log10(sSFR) < -11) & (VJ < 1), 'post-SB', 'orange'
            
            # star-forming    
            idx, label, c = ok & (UV < 0.6) & (VJ < 0.5), 'SF0', 'purple'
            
            idx, label, c = ok & (np.abs(UV-0.8) < 0.2) & (np.abs(VJ-0.6) < 0.2), 'SF1', 'b'

            idx, label, c = ok & (np.abs(UV-1.2) < 0.2) & (np.abs(VJ-1.0) < 0.2), 'SF2', 'orange'

            idx, label, c = ok & (np.abs(UV-1.6) < 0.2) & (np.abs(VJ-1.6) < 0.2), 'SF3', 'pink'
        
        rf_tempfilt, f_rest = self.rest_frame_fluxes(f_numbers=[153,norm_band,247], pad_width=0.5, percentiles=[2.5,16,50,84,97.5]) 
        
        norm_flux = f_rest[:,1,2]
        fnu_norm = (self.fnu[idx,:].T/norm_flux[idx]).T
        fmodel_norm = (self.fmodel[idx,:].T/norm_flux[idx]).T
        
        lcz = np.dot(1/(1+self.zbest[:, np.newaxis]), self.lc[np.newaxis,:])[idx,:]
        
        clip = (self.efnu[idx,:] > 0) & (self.fnu[idx,:] > self.param['NOT_OBS_THRESHOLD']) & np.isfinite(self.fnu[idx,:]) & np.isfinite(self.efnu[idx,:])
        
        sp = self.show_fit(self.cat['id'][idx][0], get_spec=True)
        templf = []
        for i in np.arange(self.NOBJ)[idx]:
            sp = self.show_fit(self.cat['id'][i], get_spec=True)
            templf.append(sp['templf']/(norm_flux[i]/(1+self.zbest[i])**2))
            #plt.plot(self.templates[0].wave[::10], sp['templf'][::10]/(norm_flux[i]/(1+self.zbest[i])**2), alpha=0.1, color=c)
        
        #plt.loglog()
        
        fig = plt.figure(figsize=[10,4])
        gs = GridSpec(1,2, width_ratios=[2,3])
        
        ax = fig.add_subplot(gs[0,0])
        sc = ax.scatter(VJ[ok], UV[ok], c='k', vmin=-1, vmax=1., alpha=0.01, marker='.', edgecolor='k', cmap='spectral')
        sc = ax.scatter(VJ[idx], UV[idx], c=c, vmin=-1, vmax=1., alpha=0.2, marker='o', edgecolor='k')

        ax.set_xlabel(r'$V-J$ (rest)')
        ax.set_ylabel(r'$U-V$ (rest)')

        ax.set_xlim(-0.2,2.8); ax.set_ylim(-0.2,2.8)
        ax.grid()
        
        ax = fig.add_subplot(gs[0,1])
        
        wave = lcz[clip]
        flam = fnu_norm[clip]/(wave/rf_tempfilt.filters[1].pivot())**2
        flam_obs = fmodel_norm[clip]/(wave/rf_tempfilt.filters[1].pivot())**2
        
        xm, ym, ys, N = utils.running_median(wave, flam, NBIN=50, use_median=True, use_nmad=True, reverse=False)
        #c = 'r'
        ax.plot(xm, np.maximum(ym, 0.01), color=c, linewidth=2, alpha=0.4)
        ax.fill_between(xm, np.maximum(ym+ys, 0.001), np.maximum(ym-ys, 0.001), color=c, alpha=0.4)
        
        med = np.median(np.array(templf), axis=0)/3.6
        min = np.percentile(np.array(templf), 16, axis=0)/3.6
        max = np.percentile(np.array(templf), 84, axis=0)/3.6
        ax.plot(self.templates[0].wave[::5], med[::5], color=c, linewidth=1, zorder=2)
    
        ax.fill_between(self.templates[0].wave[::5], min[::5], max[::5], color=c, linewidth=1, zorder=2, alpha=0.1)

        # MIPS
        mips_obs = kate_sfr['f24tot']*10**(0.4*(self.param.params['PRIOR_ABZP']-23.9))/norm_flux/(24.e4/(1+self.zbest)/rf_tempfilt.filters[1].pivot())**2#/2
        ok_mips = (mips_obs > 0)
        
        xm, ym, ys, N = utils.running_median(24.e4/(1+self.zbest[idx & ok_mips]), np.log10(mips_obs[idx & ok_mips]), NBIN=10, use_median=True, use_nmad=True, reverse=False)
        ax.fill_between(xm, np.maximum(10**(ym+ys), 1.e-4), np.maximum(10**(ym-ys), 1.e-4), color=c, alpha=0.4)
        
        ax.scatter(24.e4/(1+self.zbest[idx & ok_mips]), mips_obs[idx & ok_mips], color=c, marker='.', alpha=0.1)
        ax.set_xlabel(r'$\lambda_\mathrm{rest}$')
        ax.set_ylabel(r'$f_\lambda\ /\ f_V$')
        ax.loglog()
        ax.set_xlim(2000,120.e5)
        ax.set_ylim(1.e-4,4)
        ax.grid()
        
        fig.tight_layout()
        return fig
        
        #fig.savefig('gs_eazypy.RF_{0}.png'.format(label))
        
def _obj_nnls(coeffs, A, fnu_i, efnu_i):
    fmodel = np.dot(coeffs, A)
    return -0.5*np.sum((fmodel-fnu_i)**2/efnu_i**2)
             
class TemplateGrid(object):
    def __init__(self, zgrid, templates, RES='FILTERS.latest', f_numbers=[156], add_igm=True, galactic_ebv=0, n_proc=4, Eb=0, interpolator=None, filters=None, verbose=2):
        import multiprocessing as mp
        import scipy.interpolate 
        import specutils.extinction
        import astropy.units as u
                
        self.templates = templates
        self.RES = RES
        self.f_numbers = f_numbers
        self.add_igm = add_igm
        self.galactic_ebv = galactic_ebv
        self.Eb = Eb
        
        self.NTEMP = len(templates)
        self.NZ = len(zgrid)
        
        self.zgrid = zgrid
        self.dz = np.diff(zgrid)
        self.idx = np.arange(self.NZ, dtype=int)
                
        if filters is None:
            all_filters = np.load(RES+'.npy')[0]
            filters = [all_filters.filters[fnum-1] for fnum in f_numbers]
        
        self.lc = np.array([f.pivot() for f in filters])
        self.filter_names = np.array([f.name for f in filters])
        self.filters = filters
        self.NFILT = len(self.filters)
        
        self.tempfilt = np.zeros((self.NZ, self.NTEMP, self.NFILT))
        
        if n_proc >= 0:
            # Parallel            
            if n_proc == 0:
                pool = mp.Pool(processes=mp.cpu_count())
            else:
                pool = mp.Pool(processes=n_proc)
                
            results = [pool.apply_async(_integrate_tempfilt, (itemp, templates[itemp], zgrid, RES, f_numbers, add_igm, galactic_ebv, Eb, filters)) for itemp in range(self.NTEMP)]

            pool.close()
            pool.join()
                
            for res in results:
                itemp, tf_i = res.get(timeout=1)
                if verbose > 1:
                    print('Process template {0}.'.format(templates[itemp].name))
                self.tempfilt[:,itemp,:] = tf_i        
        else:
            # Serial
            for itemp in range(self.NTEMP):
                itemp, tf_i = _integrate_tempfilt(itemp, templates[itemp], zgrid, RES, f_numbers, add_igm, galactic_ebv, Eb, filters)
                if verbose > 1:
                    print('Process template {0}.'.format(templates[itemp].name))
                self.tempfilt[:,itemp,:] = tf_i        
        
        # Spline interpolator        
        if interpolator is None:
            self.spline = scipy.interpolate.Akima1DInterpolator(self.zgrid, self.tempfilt, axis=0)
            #self.spline = scipy.interpolate.CubicSpline(self.zgrid, self.tempfilt)
        else:
            self.spline = interpolator(self.zgrid, self.tempfilt)
            
    def __call__(self, z):
        """
        Return interpolated filter fluxes
        """
        
        return self.spline(z)
                        
def _integrate_tempfilt(itemp, templ, zgrid, RES, f_numbers, add_igm, galactic_ebv, Eb, filters):
    """
    For multiprocessing filter integration
    """
    import specutils.extinction
    import astropy.units as u
    
    if filters is None:
        all_filters = np.load(RES+'.npy')[0]
        filters = [all_filters.filters[fnum-1] for fnum in f_numbers]
    
    NZ = len(zgrid)
    NFILT = len(filters)
        
    if add_igm:
        igm = igm_module.Inoue14()
    else:
        igm = 1.

    f99 = specutils.extinction.ExtinctionF99(galactic_ebv*3.1)
    
    # Add bump with Drude profile in template rest frame
    width = 350
    l0 = 2175
    Abump = Eb/4.05*(templ.wave*width)**2/((templ.wave**2-l0**2)**2+(templ.wave*width)**2)
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
        A_MW[red] = f99(lz[red]*u.angstrom)
        F_MW = 10**(-0.4*A_MW)
        
        for ifilt in range(NFILT):
            fnu = templ.integrate_filter(filters[ifilt], scale=igmz*F_MW*Fbump, z=zgrid[iz])
            tempfilt[iz, ifilt] = fnu
    
    return itemp, tempfilt
            
def _fit_vertical(iz, z, A, fnu_corr, efnu_corr, TEF, zp, verbose):
    
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
        ok_band = (fnu_i > -90) & (efnu_i > 0)
        
        if ok_band.sum() < 2:
            continue
        
        chi2[iobj], coeffs[iobj], fmodel, draws = _fit_obj(fnu_i, efnu_i, A, TEFz, zp, False)
            
    return iz, chi2, coeffs

def _fit_obj(fnu_i, efnu_i, A, TEFz, zp, get_err):
    from scipy.optimize import nnls

    ok_band = (fnu_i/zp > -90) & (efnu_i/zp > 0) & np.isfinite(fnu_i) & np.isfinite(efnu_i)
    if ok_band.sum() < 2:
        return np.inf, np.zeros(A.shape[0])
        
    var = efnu_i**2 + (TEFz*fnu_i)**2
    rms = np.sqrt(var)
    
    Ax = (A/rms).T[ok_band,:]
    try:
        coeffs_i, rnorm = nnls(Ax, (fnu_i/rms)[ok_band])
    except:
        coeffs_i = np.zeros(A.shape[0])
        
    fmodel = np.dot(coeffs_i, A)
    chi2_i = np.sum((fnu_i-fmodel)**2/var*ok_band)
    
    coeffs_draw = None
    if get_err > 0:
        ok_temp = coeffs_i > 0
        
        coeffs_draw = np.zeros((get_err, A.shape[0]))
        try:
            covar = np.matrix(np.dot(Ax[:,ok_temp].T, Ax[:,ok_temp])).I
            coeffs_draw[:, ok_temp] = np.random.multivariate_normal(coeffs_i[ok_temp], covar, size=get_err)
        except:
            coeffs_draw = None
            
    return chi2_i, coeffs_i, fmodel, coeffs_draw

        