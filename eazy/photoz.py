import os
import time
import numpy as np

from collections import OrderedDict

from .filters import FilterFile
from .param import EazyParam, TranslateFile
from .igm import Inoue14
from .templates import TemplateError

class PhotoZ(object):
    def __init__(self, param_file='zphot.param.m0416.uvista', translate_file='zphot.translate.m0416.uvista', zeropoint_file=None):
        
        self.param_file = param_file
        self.translate_file = translate_file
        self.zeropoint_file = zeropoint_file
        
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
        self.param = EazyParam(param_file, read_templates=False, read_filters=False)
        self.translate = TranslateFile(translate_file)
                
        if 'MW_EBV' not in self.param.params:
            self.param.params['MW_EBV'] = 0.0354 # MACS0416
            #self.param.params['MW_EBV'] = 0.0072 # GOODS-S
            #self.param['SCALE_2175_BUMP'] = 0.4 # Test
        
        ### Read templates
        self.templates = self.param.read_templates(templates_file=self.param['TEMPLATES_FILE'])
        self.NTEMP = len(self.templates)
        
        ### Set redshift fit grid
        #self.param['Z_STEP'] = 0.003
        self.get_zgrid()
        
        ### Read catalog and filters
        self.read_catalog()
        
        if zeropoint_file is not None:
            self.read_zeropoint(zeropoint_file)
        else:
            self.zp = self.f_numbers*0+1.
            
        self.fit_chi2 = np.zeros((self.NOBJ, self.NZ))
        self.fit_coeffs = np.zeros((self.NOBJ, self.NZ, self.NTEMP))
        
        ### Interpolate templates
        #self.tempfilt = TemplateGrid(self.zgrid, self.templates, self.filters, add_igm=True, galactic_ebv=0.0354)

        t0 = time.time()
        self.tempfilt = TemplateGrid(self.zgrid, self.templates, self.param['FILTERS_RES'], self.f_numbers, add_igm=True, galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'], n_proc=0)
        t1 = time.time()
        print('Process templates: {0:.3f} s'.format(t1-t0))
        
        ### Template Error
        self.get_template_error()
        
        #### testing
        if False:
            
            idx = np.arange(self.NOBJ)[self.cat['z_spec'] > 0]
            i=37
            fig = eazy.plotExampleSED(idx[i], MAIN_OUTPUT_FILE='m0416.uvista.full')
         
            obj_ix = 2480
            obj_ix = idx[i]
                    
    def read_catalog(self):
        from astropy.table import Table
                
        if 'fits' in self.param['CATALOG_FILE'].lower():
            self.cat = Table.read(self.param['CATALOG_FILE'], format='fits')
        else:
            self.cat = Table.read(self.param['CATALOG_FILE'], format='ascii.commented_header')
        self.NOBJ = len(self.cat)
        
        all_filters = FilterFile(self.param['FILTERS_RES'])
        np.save(self.param['FILTERS_RES']+'.npy', [all_filters])

        self.filters = []
        self.flux_columns = []
        self.err_columns = []
        self.f_numbers = []
        
        for k in self.translate.trans:
            fcol = self.translate.trans[k]
            if fcol.startswith('F') & ('FTOT' not in fcol):
                f_number = int(fcol[1:])
                for ke in self.translate.trans:
                    if self.translate.trans[ke] == 'E%d' %(f_number):
                        break
                                 
                if (k in self.cat.colnames) & (ke in self.cat.colnames):
                    self.filters.append(all_filters.filters[f_number-1])
                    self.flux_columns.append(k)
                    self.err_columns.append(ke)
                    self.f_numbers.append(f_number)
                    print '%s %s (%3d): %s' %(k, ke, f_number, self.filters[-1].name.split()[0])
                        
        self.f_numbers = np.array(self.f_numbers)
        
        self.lc = np.array([f.pivot() for f in self.filters])
        self.ext_corr = np.array([10**(0.4*f.extinction_correction(self.param.params['MW_EBV'])) for f in self.filters])
        self.zp = self.ext_corr*0+1
        
        self.NFILT = len(self.filters)
        self.fnu = np.zeros((self.NOBJ, self.NFILT))
        self.efnu = np.zeros((self.NOBJ, self.NFILT))
        
        for i in range(self.NFILT):
            self.fnu[:,i] = self.cat[self.flux_columns[i]]
            self.efnu[:,i] = self.cat[self.err_columns[i]]*self.translate.error[self.err_columns[i]]
        
        self.efnu_orig = self.efnu*1.
        
        # Translate the file itself    
        for k in self.translate.trans:
            if k in self.cat.colnames:
                self.cat.rename_column(k, self.translate.trans[k])
        
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
        self.TEF = TemplateError(self.param['TEMP_ERR_FILE'], lc=self.lc, scale=self.param['TEMP_ERR_A2'])
            
    def get_zgrid(self):
        self.zgrid = log_zgrid(zr=[self.param['Z_MIN'], self.param['Z_MAX']], 
                               dz = self.param['Z_STEP'])
        self.NZ = len(self.zgrid)
    
    def iterate_zp_templates(self, idx=None, update_templates=True, update_zeropoints=True, iter=0, n_proc=4, save_templates=False, error_residuals=False):
        
        self.fit_parallel(idx=idx, n_proc=n_proc)
        
        self.best_fit()
        fig = self.residuals(update_zeropoints=update_zeropoints,
                       ref_filter=int(self.param['PRIOR_FILTER']),
                       update_templates=update_templates)
        
        fig.savefig('iter_%03d.png' %(iter))
        
        if error_residuals:
            self.error_residuals()
                                
        # dz = (self.zbest_grid-self.cat['z_spec'])/(1+self.cat['z_spec'])
        # dzb = (self.zbest-self.cat['z_spec'])/(1+self.cat['z_spec'])
        # dz2 = (zout['z_peak']-self.cat['z_spec'])/(1+self.cat['z_spec'])
        # clip = (self.zbest_grid > self.zgrid[0]) & (self.cat['z_spec'] > 0)
        # print('{0:.4f} {1:.4f} {2:.4f}'.format(threedhst.utils.nmad(dz[clip]), threedhst.utils.nmad(dzb[clip]), threedhst.utils.nmad(dz2[clip])))
        if save_templates:
            self.save_templates()
            
    def save_templates(self):
        path = os.path.dirname(self.param['TEMPLATES_FILE'])
        for templ in self.templates:
            templ_file = os.path.join('{0}/tweak_{1}'.format(path,templ.name))
            
            print('Save tweaked template {0}'.format(templ_file))
                                      
            np.savetxt(templ_file, np.array([templ.wave, templ.flux]).T,
                       fmt='%.6e')
                           
    def fit_parallel(self, idx=None, n_proc=4, verbose=True):

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
        
        results = [pool.apply_async(_fit_vertical, (iz, self.zgrid[iz],  self.tempfilt(self.zgrid[iz]), fnu_corr, efnu_corr, self.TEF)) for iz in range(self.NZ)]

        pool.close()
        pool.join()
                
        for res in results:
            iz, chi2, coeffs = res.get(timeout=1)
            self.fit_chi2[idx,iz] = chi2
            self.fit_coeffs[idx,iz,:] = coeffs
        
        t1 = time.time()
        if verbose:
            print('Fit {1:.1f} s (n_proc={0}, NOBJ={2})'.format(n_proc, t1-t0, len(idx)))
        
            
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
            coeffs_i, rnorm = nnls((A/rms).T[ok_band,:], (fnu_i/rms)[ok_band])
                
            fobs = np.dot(coeffs_i, A)
            chi2[iz] = np.sum((fnu_i-fobs)**2/var*ok_band)
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
    
    def best_fit(self, zbest=None):
        self.fobs = self.fnu*0.
        izbest = np.argmin(self.fit_chi2, axis=1)

        self.zbest_grid = self.zgrid[izbest]
        if zbest is None:
            self.zbest, self.chi_best = self.best_redshift()
        else:
            self.zbest = zbest
                
        fnu_corr = self.fnu*self.ext_corr*self.zp
        efnu_corr = self.efnu*self.ext_corr*self.zp
        
        self.coeffs_best = np.zeros((self.NOBJ, self.NTEMP))
        
        idx = np.arange(self.NOBJ)[self.zbest > self.zgrid[0]]
        for iobj in idx:
            #A = self.tempfilt(self.zgrid[izbest[iobj]])
            #self.fobs[iobj,:] = np.dot(self.fit_coeffs[iobj, izbest[iobj],:], A) 
            zi = self.zbest[iobj]
            A = self.tempfilt(zi)
            TEFz = self.TEF(zi)
            
            fnu_i = fnu_corr[iobj, :]
            efnu_i = efnu_corr[iobj,:]
            chi2, self.coeffs_best[iobj,:], self.fobs[iobj,:], draws = _fit_obj(fnu_i, efnu_i, A, TEFz, False)
            
    def best_redshift(self):
        """Fit parabola to chi2 to get best minimum
        
        TBD: include prior
        """
        from scipy import polyfit, polyval
        izbest = np.argmin(self.fit_chi2, axis=1)
        
        zbest = self.zgrid[izbest]
        chi_best = self.fit_chi2.min(axis=1)
        
        for iobj in range(self.NOBJ):
            iz = izbest[iobj]
            if (iz == 0) | (iz == self.NZ-1):
                continue
            
            c = polyfit(self.zgrid[iz-1:iz+2], 
                        self.fit_chi2[iobj, iz-1:iz+2], 2)
            
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
        r = np.abs(self.fobs - self.fnu*self.ext_corr*self.zp)
        
        # Update where residual larger than uncertainty
        upd = (self.efnu > 0) & (self.fnu > self.param['NOT_OBS_THRESHOLD'])
        upd &= (r > self.efnu) & (self.fobs > 0)
        
        self.efnu[upd] = r[upd] #np.sqrt(var_new[upd])
    
    
    def check_uncertainties(self):
        
        TEF_scale = 1.
        
        full_err = self.efnu*0.
        teff_err = self.efnu*0
        for i in range(self.NOBJ):
            teff_err[i,:] = self.TEF(self.zbest[i])*TEF_scale
        
        full_err = np.sqrt(self.efnu**2+(self.fnu*teff_err)**2)
            
        resid = (self.fobs - self.fnu*self.ext_corr*self.zp)/full_err
        
        okz = (self.zbest > 0.1)
        scale_errors = self.lc*0.
        
        for ifilt in range(self.NFILT):
            iok = okz & (self.efnu[:,ifilt] > 0) & (self.fnu[:,ifilt] > self.param['NOT_OBS_THRESHOLD'])
            print '{0} {1:d} {2:.2f}'.format(self.flux_columns[ifilt], self.f_numbers[ifilt], threedhst.utils.nmad(resid[iok,ifilt]))
            scale_errors[ifilt] = threedhst.utils.nmad(resid[iok,ifilt])
            
            #plt.hist(resid[iok,ifilt], bins=100, range=[-3,3], alpha=0.5)
        
    def residuals(self, update_zeropoints=True, update_templates=True, ref_filter=205):
        import os
        import matplotlib as mpl
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        
        import threedhst
        from astroML.sum_of_norms import sum_of_norms, norm
               
        izbest = np.argmin(self.fit_chi2, axis=1)
        zbest = self.zgrid[izbest]
        
        idx = np.arange(self.NOBJ)[izbest > 0]
        
        resid = (self.fobs - self.fnu*self.ext_corr*self.zp)/self.fobs+1
        eresid = (self.fobs - self.efnu*self.ext_corr*self.zp)/self.fobs+1

        sn = self.fnu/self.efnu
                
        fig = plt.figure(figsize=[16,4])
        gs = gridspec.GridSpec(1, 4)
        ax = fig.add_subplot(gs[:,:3])
        
        cmap = cm.rainbow
        cnorm = mpl.colors.Normalize(vmin=0, vmax=self.NFILT-1)
        
        so = np.argsort(self.lc)
        
        lcz = np.dot(1/(1+self.zgrid[izbest][:, np.newaxis]), self.lc[np.newaxis,:])
        clip = (sn > 3) & (self.efnu > 0) & (self.fnu > self.param['NOT_OBS_THRESHOLD']) & (resid > 0)
        xmf, ymf, ysf, Nf = threedhst.utils.runmed(lcz[clip], resid[clip], NBIN=20*(self.NFILT/2), use_median=True, use_nmad=True)
        
        Ng = 40
        w_best, rms, locs, widths = sum_of_norms(xmf, ymf, Ng, spacing='log', full_output=True)
        norms = (w_best * norm(xmf[:, None], locs, widths)).sum(1)
        ax.plot(xmf, norms, color='k', linewidth=2, alpha=0.5, zorder=10)
        
        self.tnorm = (w_best, locs, widths, xmf.min(), xmf.max())
        
        self.zp_delta = self.zp*0
        
        for i in range(self.NFILT):
            ifilt = so[i]
            clip = (izbest > 0) & (sn[:,ifilt] > 3) & (self.efnu[:,ifilt] > 0) & (self.fnu[:,ifilt] > self.param['NOT_OBS_THRESHOLD']) & (resid[:,ifilt] > 0)
            color = cmap(cnorm(i))

            xi = self.lc[ifilt]/(1+self.zgrid[izbest][clip])
            xm, ym, ys, N = threedhst.utils.runmed(xi, resid[clip, ifilt], NBIN=20, use_median=True, use_nmad=True)
            
            # Normalize to overall mediadn
            xgi = (w_best * norm(xi[:, None], locs, widths)).sum(1)
            delta = np.median(resid[clip, ifilt]-xgi+1)
            self.zp_delta[ifilt]  = delta
            
            fname = os.path.basename(self.filters[ifilt].name.split()[0])
            if fname.count('.') > 1:
                fname = '.'.join(fname.split('.')[:-1])
            
            ax.plot(xm, ym/delta, color=color, alpha=0.8, label='%-30s %.3f' %(fname, delta), linewidth=2)
            ax.fill_between(xm, ym/delta-ys/np.sqrt(N), ym/delta+ys/np.sqrt(N), color=color, alpha=0.1) 
                    
        ax.semilogx()
        ax.set_ylim(0.8,1.2)
        ax.set_xlim(800,6.e4)
        l = ax.legend(fontsize=8, ncol=5)
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
        
        ax.text(0.05, 0.925, r'N={0}, $\sigma$={1:.4f}'.format(clip.sum(), threedhst.utils.nmad(dz[clip])), ha='left', va='top', fontsize=10, transform=ax.transAxes)
        fig.tight_layout(pad=0.1)
        
        # update zeropoints in self.zp
        if update_zeropoints:
            ref_ix = self.f_numbers == ref_filter
            self.zp *= self.zp_delta/self.zp_delta[ref_ix]
        
        # tweak templates
        if update_templates:
            print('Reprocess tweaked templates')
            w_best, locs, widths, xmin, xmax = self.tnorm
            for itemp in range(self.NTEMP):
                templ = self.templates[itemp]
                templ_tweak = (w_best * norm(templ.wave[:, None], locs, widths)).sum(1)
                templ_tweak[(templ.wave < xmin) | (templ.wave > xmax)] = 1
                templ.flux /= templ_tweak
                templ.flux_fnu /= templ_tweak
            
            # Recompute filter fluxes from tweaked templates    
            self.tempfilt = TemplateGrid(self.zgrid, self.templates, self.param['FILTERS_RES'], self.f_numbers, add_igm=True, galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'], n_proc=0)
        
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
            igmz[lyman] = Inoue14().full_IGM(z, templz[lyman])
        else:
            igmz = 1.
        
        templf = np.dot(coeffs_i, tempflux)*igmz
        return templz, templf
        
    def show_fit(self, id, show_fnu=False, xlim=[0.3, 9], get_spec=False):
        import matplotlib.pyplot as plt
        
        if False:
            ids = self.cat['id'][(self.cat['z_spec'] > 1.9) & (self.zbest > 0.8)]

            ids = self.cat['id'][(self.cat['z_spec'] > 0.2) & (self.zbest > 0.2)]

            ids = self.cat['id'][(self.zbest > 1.7)]

            j = -1
            
            j+=1; id = ids[j]; self.show_fit(id, show_fnu=show_fnu)
            
        ix = self.cat['id'] == id
        z = self.zbest[ix][0]
                
        ## SED
        A = np.squeeze(self.tempfilt(z))
        fnu_i = np.squeeze(self.fnu[ix, :])*self.ext_corr*self.zp
        efnu_i = np.squeeze(self.efnu[ix,:])*self.ext_corr*self.zp
        
        chi2_i, coeffs_i, fobs, draws = _fit_obj(fnu_i, efnu_i, A, self.TEF(self.zbest[ix]), False)
        #templz, templf = self.full_sed(self.zbest[ix][0], coeffs_i)
        if True:
            templ = self.templates[0]
            tempflux = np.zeros((self.NTEMP, templ.wave.shape[0]))
            for i in range(self.NTEMP):
                tempflux[i, :] = self.templates[i].flux_fnu

            templz = templ.wave*(1+z)

            if self.tempfilt.add_igm:
                igmz = templ.wave*0.+1
                lyman = templ.wave < 1300
                igmz[lyman] = Inoue14().full_IGM(z, templz[lyman])
            else:
                igmz = 1.

            templf = np.dot(coeffs_i, tempflux)*igmz
            
        fnu_factor = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
        
        if show_fnu:
            if show_fnu == 2:
                flam_spec = 1.e29/(templz/1.e4)
                flam_sed = 1.e29/self.ext_corr/(self.lc/1.e4)
                ylabel = (r'$f_\nu / \lambda$ [$\mu$Jy / $\mu$m]')
            else:
                flam_spec = 1.e29
                flam_sed = 1.e29
                ylabel = (r'$f_\nu$ [$\mu$Jy]')                
        else:
            flam_spec = 3.e18/templz**2/1.e-19
            flam_sed = 3.e18/self.lc**2/self.ext_corr/1.e-19
            ylabel = (r'$f_\lambda\times10^{-19}$ cgs')
        
        if get_spec:
            data = OrderedDict(lc=self.lc, model=fobs*fnu_factor*flam_sed,
                               fobs=fnu_i*fnu_factor*flam_sed, 
                               efobs=efnu_i*fnu_factor*flam_sed,
                               templz=templz,
                               templf=templf*fnu_factor*flam_spec,
                               unit=show_fnu*1)
                               
            return data
        
        
        fig = plt.figure(figsize=[8,4])
        ax = fig.add_subplot(121)
        
        ax.set_ylabel(ylabel)
            
        ax.scatter(self.lc/1.e4, fobs*fnu_factor*flam_sed, color='r')
        ax.errorbar(self.lc/1.e4, fnu_i*fnu_factor*flam_sed, efnu_i*fnu_factor*flam_sed, color='k', marker='s', linestyle='None')
        ax.plot(templz/1.e4, templf*fnu_factor*flam_spec, alpha=0.5, zorder=-1)
        ax.set_xlim(xlim)
        xt = np.array([0.5, 1, 2, 4])*1.e4
        
        ymax = (fobs*fnu_factor*flam_sed).max()
        ax.set_ylim(-0.1*ymax, 1.2*ymax)
        ax.semilogx()

        ax.set_xticks(xt/1.e4)
        ax.set_xticklabels(xt/1.e4)

        ax.set_xlabel(r'$\lambda_\mathrm{obs}$')
        
        ## P(z)
        
        ax = fig.add_subplot(122)
        chi2 = np.squeeze(self.fit_chi2[ix,:])
        pz = np.exp(-(chi2-chi2.min())/2.)
        pz /= np.trapz(pz, self.zgrid)
        ax.plot(self.zgrid, pz, color='orange')
        ax.fill_between(self.zgrid, pz, pz*0, color='yellow', alpha=0.5)
        if self.cat['z_spec'][ix] > 0:
            ax.vlines(self.cat['z_spec'][ix][0], 0,pz.max()*1.05, color='r')
        
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
        rf_tempfilt = TemplateGrid(np.array([0,0.1]), self.templates, self.param['FILTERS_RES'], np.array(f_numbers), add_igm=False, galactic_ebv=0, Eb=self.param['SCALE_2175_BUMP'], n_proc=-1)
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
            print ix
            fnu_i = fnu_corr[ix,:]*1
            efnu_i = efnu_corr[ix,:]*1
            z = self.zbest[ix]
            A = self.tempfilt(z)   
        
            for i in range(NREST):
                ## Grow uncertainties away from RF band
                lc_i = rf_tempfilt.lc[i]
                grow = np.exp(-(lc_i-self.lc/(1+z))**2/2/(pad_width*lc_i)**2)
                TEFz = (2/(1+grow/grow.max())-1)*0.5
            
                chi2_i, coeffs_i, fobs_i, draws = _fit_obj(fnu_i, efnu_i, A, TEFz, 100)
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
        #     fobs = np.dot(coeffs_i, A)
        #     chi2_i = np.sum((fnu_i-fobs)**2/var*ok_band)
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
        #     fobs_err = np.sqrt(np.dot(coeffs_err**2, A**2))
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
            
def _obj_nnls(coeffs, A, fnu_i, efnu_i):
    fobs = np.dot(coeffs, A)
    return -0.5*np.sum((fobs-fnu_i)**2/efnu_i**2)
             
class TemplateGrid(object):
    def __init__(self, zgrid, templates, RES, f_numbers, add_igm=True, galactic_ebv=0, n_proc=4, Eb=0):
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
        self.NFILT = len(f_numbers)
        
        self.zgrid = zgrid
        self.dz = np.diff(zgrid)
        self.idx = np.arange(self.NZ, dtype=int)
        
        self.tempfilt = np.zeros((self.NZ, self.NTEMP, self.NFILT))
        
        all_filters = np.load(RES+'.npy')[0]
        filters = [all_filters.filters[fnum-1] for fnum in f_numbers]
        self.lc = np.array([f.pivot() for f in filters])
        self.filter_names = np.array([f.name for f in filters])
        self.filters = filters
        
        if n_proc >= 0:
            # Parallel            
            if n_proc == 0:
                pool = mp.Pool(processes=mp.cpu_count())
            else:
                pool = mp.Pool(processes=n_proc)
                
            results = [pool.apply_async(_integrate_tempfilt, (itemp, templates[itemp], zgrid, RES, f_numbers, add_igm, galactic_ebv, Eb)) for itemp in range(self.NTEMP)]

            pool.close()
            pool.join()
                
            for res in results:
                itemp, tf_i = res.get(timeout=1)
                print('Process template {0}.'.format(templates[itemp].name))
                self.tempfilt[:,itemp,:] = tf_i        
        else:
            # Serial
            for itemp in range(self.NTEMP):
                itemp, tf_i = _integrate_tempfilt(itemp, templates[itemp], zgrid, RES, f_numbers, add_igm, galactic_ebv, Eb)
                print('Process template {0}.'.format(templates[itemp].name))
                self.tempfilt[:,itemp,:] = tf_i        
        
        # Spline interpolator        
        self.spline = scipy.interpolate.CubicSpline(self.zgrid, self.tempfilt)
        
    def __call__(self, z):
        """
        Return interpolated filter fluxes
        """
        return self.spline(z)
                        
def _integrate_tempfilt(itemp, templ, zgrid, RES, f_numbers, add_igm, galactic_ebv, Eb):
    """
    For multiprocessing filter integration
    """
    import specutils.extinction
    import astropy.units as u
    
    all_filters = np.load(RES+'.npy')[0]
    filters = [all_filters.filters[fnum-1] for fnum in f_numbers]
    
    NZ = len(zgrid)
    NFILT = len(filters)
        
    if add_igm:
        igm = Inoue14()
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
            
def _fit_vertical(iz, z, A, fnu_corr, efnu_corr, TEF):
    
    NOBJ, NFILT = fnu_corr.shape#[0]
    NTEMP = A.shape[0]
    chi2 = np.zeros(NOBJ)
    coeffs = np.zeros((NOBJ, NTEMP))
    TEFz = TEF(z)
    
    for iobj in range(NOBJ):
        
        fnu_i = fnu_corr[iobj, :]
        efnu_i = efnu_corr[iobj,:]
        ok_band = (fnu_i > -90) & (efnu_i > 0)
        
        if ok_band.sum() < 3:
            continue
        
        chi2[iobj], coeffs[iobj], fobs, draws = _fit_obj(fnu_i, efnu_i, A, TEFz, False)
            
    return iz, chi2, coeffs
    
def _fit_obj(fnu_i, efnu_i, A, TEFz, get_err):
    from scipy.optimize import nnls

    ok_band = (fnu_i > -90) & (efnu_i > 0)
    if ok_band.sum() < 3:
        return np.inf, np.zeros(A.shape[0])
        
    var = efnu_i**2 + (TEFz*fnu_i)**2
    rms = np.sqrt(var)
    
    Ax = (A/rms).T[ok_band,:]
    coeffs_i, rnorm = nnls(Ax, (fnu_i/rms)[ok_band])
    fobs = np.dot(coeffs_i, A)
    chi2_i = np.sum((fnu_i-fobs)**2/var*ok_band)
    
    coeffs_draw = None
    if get_err > 0:
        ok_temp = coeffs_i > 0
        
        coeffs_draw = np.zeros((get_err, A.shape[0]))
        try:
            covar = np.matrix(np.dot(Ax[:,ok_temp].T, Ax[:,ok_temp])).I
            coeffs_draw[:, ok_temp] = np.random.multivariate_normal(coeffs_i[ok_temp], covar, size=get_err)
        except:
            coeffs_draw = None
            
    return chi2_i, coeffs_i, fobs, coeffs_draw
                   
def log_zgrid(zr=[0.7,3.4], dz=0.01):
    """Make a logarithmically spaced redshift grid
    
    Parameters
    ----------
    zr : [float, float]
        Minimum and maximum of the desired grid
    
    dz : float
        Step size, dz/(1+z)
    
    Returns
    -------
    zgrid : array-like
        Redshift grid
    
    """
    zgrid = np.exp(np.arange(np.log(1+zr[0]), np.log(1+zr[1]), dz))-1
    return zgrid

        