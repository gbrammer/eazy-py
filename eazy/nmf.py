def build_matrices(NTEMP=5, field='cosmos', hmax=24, zmax=2.5):
    """
    Build matrices for Blanton & Roweis optimization
    """ 
    import unicorn
    import numpy as np
    #import br07
    import glob
    from threedhst import catIO

    import eazy
    
    # UltraVista
    self = eazy.photoz.PhotoZ(param_file='zphot.param.new', translate_file='zphot.translate.new', zeropoint_file='zphot.zeropoint.new')
    zbest = np.load('zbest.npy')[0]
    self.best_fit(zbest=zbest)
    
    zsp_only=False
    
    zp_ix = self.f_numbers == self.param['PRIOR_FILTER']
    mag = self.param['PRIOR_ABZP'] - np.squeeze(2.5*np.log10(self.fnu[:, zp_ix]))
    
    idx = (mag < 23) | ((self.cat['z_spec'] > 0) & np.isfinite(mag))
    #idx = (mag < 22) | ((self.cat['z_spec'] > 0) & np.isfinite(mag))
    if zsp_only:
        idx &= (self.cat['z_spec'] > 0)
        
    if 'USE' in self.cat.colnames:
        # UltraVISTA
        use = (self.cat['USE'] == 1) & (self.cat['star'] == 0)
        use = (self.cat['contamination'] == 0) & (self.cat['nan_contam'] == 0) & (self.cat['star'] == 0)
    else:
        use = (self.cat['use'] == 1) & (self.cat['star_flag'] != 1)
        
    idx &= use
    
    idx = np.arange(self.NOBJ)[idx]
    
    iter = 200
    error_residuals, update_templates, update_zeropoints = True, False, True
    
    for i in range(3):
        self.iterate_zp_templates(idx=idx, update_templates=update_templates, iter=iter+i, error_residuals=error_residuals, update_zeropoints=update_zeropoints, save_templates=True, n_proc=4)
        self.write_zeropoint_file('zphot.zeropoint.FSPS_full')
        np.save('zbest.npy', [self.zbest])
        
    ### Fit all objects    
    idx_full = np.arange(self.NOBJ)
    error_residuals, update_templates, update_zeropoints = True, False, False
    iter = 99
    self.iterate_zp_templates(idx=idx_full, update_templates=update_templates, iter=iter, error_residuals=error_residuals, update_zeropoints=update_zeropoints, save_templates=False, n_proc=4)
    
    ### Bin on U-V vs sSFR
    plt.scatter(np.log10(ssfr[idx]), uv[idx], c=vj[idx], vmin=-0.5, vmax=2, alpha=0.2, edgecolor='0.5')
    
    xh, yh = np.clip(np.log10(ssfr[idx]), -12.95, -8.05), np.clip(uv[idx], 0.01, 2.49)
    h2 = np.histogram2d(xh, yh, range=[(-13,-8), (0,2.5)], bins=[8,8])

    xh, yh = np.clip(vj[idx], 0.01, 2.99), np.clip(uv[idx], 0.01, 3.49)
    h2 = np.histogram2d(xh, yh, range=[(0,3), (0,3.5)], bins=[16,16])
    
    ## Binned rest-frame SEDs
    #full_templates = self.param.read_templates(templates_file='templates/fsps_full/tweak_spectra.param')
    full_templates = self.param.read_templates(templates_file='templates/fsps_full/spectra.param')
    full_tempfilt = eazy.photoz.TemplateGrid(self.zgrid, full_templates, self.param['FILTERS_RES'], self.f_numbers, add_igm=True, galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'], n_proc=0)
    full_rf_tempfilt = eazy.photoz.TemplateGrid(np.array([0,0.1]), full_templates, self.param['FILTERS_RES'], np.array([153,155,161]), add_igm=False, galactic_ebv=0, Eb=self.param['SCALE_2175_BUMP'], n_proc=-1)
    full_rf_tempfilt.tempfilt = np.squeeze(full_rf_tempfilt.tempfilt[0,:,:])
        
    ix = np.argmax(h2[0])
    j, i = np.unravel_index(ix, h2[0].shape)
    
    Q = ssfr[idx] > -10
    testQ = Q
    
    sel = (xh >= h2[1][j]) & (xh <= h2[1][j+1]) & (yh >= h2[2][i]) & (yh <= h2[2][i+1]) & testQ
    
    fnu_corr = self.fnu[idx[sel],:]*self.zp*self.ext_corr
    efnu_corr = self.efnu[idx[sel],:]*self.zp*self.ext_corr

    #irest = 1
    fnu_corr = (fnu_corr.T/f_rest[idx[sel],irest,2]).T
    efnu_corr = (efnu_corr.T/f_rest[idx[sel],irest,2]).T
    lc_rest = (self.lc[:,np.newaxis]/(1+self.zbest[idx[sel]])).T
    
    templ = np.zeros((sel.sum(), self.NFILT, full_tempfilt.NTEMP))
    
    import specutils.extinction
    import astropy.units as u
    f99 = specutils.extinction.ExtinctionF99(a_v = self.tempfilt.galactic_ebv * 3.1)
    fred = 10**(-0.4*f99(full_rf_tempfilt.lc[irest]*(1+self.zbest[idx][sel])*u.AA))
    
    for ii in range(sel.sum()):
        zi = self.zbest[idx][sel][ii]
        templ[ii,:,:] = (full_tempfilt(zi).T/(full_rf_tempfilt.tempfilt[:,irest]*fred[ii]))
    
    be = efnu_corr.flatten()
    A = templ.reshape((-1,full_tempfilt.NTEMP))
    b = fnu_corr.flatten()/be
    
    ok = (self.fnu[idx[sel],:].flatten() > -99) & (self.efnu[idx[sel],:].flatten() > 0) & (lc_rest.flatten() > 1300)

    sh = self.fnu[idx[sel],:].shape
    oksh = ok.reshape(sh)
    
    yy = (fnu_corr/self.ext_corr)/(lc_rest/rf_tempfilt.lc[irest])**2
    xm, ym, ys, N = threedhst.utils.runmed(lc_rest[oksh], yy[oksh], use_median=True, use_nmad=True, NBIN=100)
    ym_i = np.interp(lc_rest, xm, ym)
    ys_i = np.interp(lc_rest, xm, ys)
    oksh &= np.abs(ym_i - yy) < 3*ys_i
    ok = oksh.flatten()
    
    coeffs, resid = scipy.optimize.nnls((A[ok,:].T/be[ok]).T,b[ok])
    # coeffs, resid, rank, s = np.linalg.lstsq((A[ok,:].T/be[ok]).T,b[ok])
    # 
    # amatrix = unicorn.utils_c.prepare_nmf_amatrix(be[ok]**2, A[ok,:].T)
    # coeffs_nmf = unicorn.utils_c.run_nmf(fnu_corr.flatten()[ok], be[ok]**2, A[ok,:].T, amatrix, verbose=True, toler=1.e-5)
    
    best = np.dot(A, coeffs)

    yym = (best.reshape(sh)/self.ext_corr)/(lc_rest/rf_tempfilt.lc[irest])**2
    xmm, ymm, ysm, Nm = threedhst.utils.runmed(lc_rest[oksh], yym[oksh], use_median=True, use_nmad=True, NBIN=100)
    
    if False:
        plt.scatter(lc_rest[oksh], (fnu_corr/self.ext_corr)[oksh]/(lc_rest[oksh]/rf_tempfilt.lc[irest])**2, alpha=0.05*100/sel.sum(), color='k', marker='.')
        plt.scatter(lc_rest[oksh], (best.reshape(sh)/self.ext_corr)[oksh]/(lc_rest[oksh]/rf_tempfilt.lc[irest])**2, alpha=0.05*100/sel.sum(), color='r', marker='.', zorder=2)
   
    plt.errorbar(xm, ym, ys, color='k')
    plt.plot(xmm, ymm, color='r', marker='.', alpha=0.4)
    plt.xlim(800,1.e5); plt.ylim(0.01,10); log()
    
    tf = np.array([full_templates[ii].flux / (full_rf_tempfilt.tempfilt[ii, irest]*3.e18/full_rf_tempfilt.lc[irest]**2) for ii in range(full_tempfilt.NTEMP)])
    tt = np.dot(coeffs, tf)
    plt.plot(full_templates[0].wave, tt, color='r')
    
    plt.scatter(lc_rest[oksh], fnu_corr[oksh]/best[ok], alpha=0.05*100/sel.sum(), color='k', marker='.')
    plt.xlim(800,1.e5); plt.ylim(0.5,1.5); plt.semilogx()
    
    