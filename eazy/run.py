def go():
    
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    
    from astropy.io import fits
    import eazy

    zsp_only = False

    # GOODS-S
    self = eazy.photoz.PhotoZ(param_file='zphot.param.goodss.uvista', translate_file='zphot.translate.goodss.uvista', zeropoint_file='zphot.zeropoint.goodss.uvista.FSPS')
    
    self = eazy.photoz.PhotoZ(param_file='zphot.param.new', translate_file='zphot.translate.new', zeropoint_file='zphot.zeropoint.new')
    
    # UVUDF
    self = eazy.photoz.PhotoZ(param_file='zphot.param', translate_file='zphot.translate', zeropoint_file='zphot.zeropoint')
    use = (self.cat['STAR'] == 0) 
    
    # R-selected
    self = eazy.photoz.PhotoZ(param_file='zphot.param.new', translate_file='zphot.translate.new', zeropoint_file='zphot.zeropoint.FSPS')
    
    # 
    self.zp = self.zp*0+1
    
    if False:
        out = fits.open('eazy_output.fits')
        self.zp = out['ZP'].data
        self.zbest = out['ZBEST'].data
        self.chibest = out['CHIBEST'].data
        self.fit_chi2 = out['FIT_CHI2'].data
        self.fobs = out['FOBS'].data
        
        for i in range(self.NFILT):
            self.cat['fit_{0}'.format(self.flux_columns[i])] = self.fobs[:,i]/self.ext_corr[i]
            
    # mag = self.param['PRIOR_ABZP'] - 2.5*np.log10(self.cat['Ks'])
    # 
    # mag = self.param['PRIOR_ABZP'] - 2.5*np.log10(self.cat['rp_tot'])

    zp_ix = self.f_numbers == self.param['PRIOR_FILTER']
    mag = self.param['PRIOR_ABZP'] - np.squeeze(2.5*np.log10(self.fnu[:, zp_ix]))
    
    idx = (mag < 23) | ((self.cat['z_spec'] > 0) & np.isfinite(mag))
    #idx = (mag < 22) | ((self.cat['z_spec'] > 0) & np.isfinite(mag))

    if zsp_only:
        idx &= (self.cat['z_spec'] > 0)
    
    use = (self.cat['use'] == 1) & (self.cat['star_flag'] != 1)
    
    if False:
        # UltraVISTA
        use = (self.cat['USE'] == 1) & (self.cat['star'] == 0)
        use = (self.cat['contamination'] == 0) & (self.cat['nan_contam'] == 0) & (self.cat['star'] == 0)
    
    idx &= use
    
    idx = np.arange(self.NOBJ)[idx]
    
    # test
    iter = 100
    error_residuals, update_templates, update_zeropoints = False, False, False
    
    self.iterate_zp_templates(idx=idx, update_templates=update_templates, iter=iter, error_residuals=error_residuals, update_zeropoints=update_zeropoints, save_templates=False, n_proc=4)
    
    rf_tempfilt, f_rest = self.rest_frame_fluxes(f_numbers=[153,155,161], pad_width=0.1, percentiles=[2.5,16,50,84,97.5])
    
    uv = -2.5*np.log10(f_rest[:,0,2]/f_rest[:,1,2])
    vj = -2.5*np.log10(f_rest[:,1,2]/f_rest[:,2,2])
    
    import astropy.cosmology
    from astropy.cosmology import Planck15
    cosmo = Planck15
    cosmo = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
    
    import astropy.units as u
    
    dL = cosmo.luminosity_distance(self.zgrid).to(u.cm)
    dL_i = np.interp(self.zbest, self.zgrid, dL)

    #mass_ratio = (np.interp(self.zbest, self.zgrid, dL)/np.interp(fout['z'], self.zgrid, dL))**2
    
    fnu_factor = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
    Lnu = (f_rest[:,:,2].T*fnu_factor*4*np.pi*dL_i**2/(1+self.zbest)).T
    nuLnu = Lnu*(3.e8/(rf_tempfilt.lc*1.e-10))
    Ltot = nuLnu / 3.839e33
    Lv = Ltot[:,1]
    Lj = Ltot[:,2]
    
    lines = open('templates/uvista_nmf/spectra_kc13_12_tweak.param').readlines()
    MLv_template = np.cast[float]([line.split()[3] for line in lines if ('MLv' in line) & (line.startswith('# '))])
    SFRv_template = np.cast[float]([line.split()[5] for line in lines if ('MLv' in line) & (line.startswith('# '))])

    irest = 1 # V
    #irest = 2 # J
    
    ### Full individual template set
    if 'fsps_full' in self.param['TEMPLATES_FILE']:
        nu = 3.e18/rf_tempfilt.lc[irest]
        Lvt = rf_tempfilt.tempfilt[:,irest]*nu # nu Lnu luminosity

        # These are masses
        lines = open(self.param['TEMPLATES_FILE']).readlines()
        MLv_template = np.cast[float]([line.split()[4] for line in lines if line.startswith('# ')])
        SFRv_template = np.cast[float]([line.split()[5] for line in lines if line.startswith('# ')])

        MLv_template /= Lvt
        SFRv_template /= Lvt

    csum = self.coeffs_best*rf_tempfilt.tempfilt[:,irest]
    csum = (csum.T/f_rest[:,irest,2]).T
    #csum = (csum.T/self.coeffs_best.sum(axis=1)).T
    MLv = np.sum(csum*MLv_template, axis=1)#/np.sum(csum, axis=1)
    stellar_mass = Ltot[:,irest]*MLv
    sfr = Ltot[:,irest]*np.sum(csum*SFRv_template, axis=1)#/np.sum(csum, axis=1)
    
    tf = rf_tempfilt.tempfilt
    uv_temp = -2.5*np.log10(tf[:,0]/tf[:,1])
    vj_temp = -2.5*np.log10(tf[:,1]/tf[:,2])
    
    plt.scatter(vj[idx], uv[idx], alpha=0.1,c=np.log10(MLv)[idx], vmin=-1, vmax=1.5, edgecolor='None')
    plt.scatter(vj_temp, uv_temp, alpha=0.8,c=np.log10(MLv_template), vmin=-1, vmax=1.5, marker='s', edgecolor='w', s=30)
    plt.xlim(-0.5,3); plt.ylim(-0.5,3)

    # sSFR
    ssfr = sfr / stellar_mass
    plt.scatter(vj[idx], uv[idx], alpha=0.1,c=np.log10(ssfr)[idx], vmin=-13, vmax=-8, edgecolor='None', cmap='jet_r')
    #plt.scatter(vj_temp, uv_temp, alpha=0.8,c=np.log10(SFRv_template/MLv_template), vmin=-13, vmax=-10, marker='s', edgecolor='w', s=30)
    plt.xlim(-0.5,3); plt.ylim(-0.5,3)
    
    j = 0
    
    mass_fraction = ((csum*MLv_template).T/(csum*MLv_template).sum(axis=1)).T
    # L fraction
    light_fraction = ((csum).T/(csum).sum(axis=1)).T
    
    weight = mass_fraction
    
    j+=1; id = self.cat['id'][idx[j]]; fig = self.show_fit(id, show_fnu=show_fnu, xlim=[0.1,10])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(vj[idx], uv[idx], alpha=0.1,c=np.log10(MLv)[idx], vmin=-1, vmax=1.5, edgecolor='None')
    ax.scatter(vj_temp, uv_temp, c=np.log10(MLv_template), vmin=-1, vmax=1.5, marker='s', edgecolor='k', s=200*weight[idx[j],:]**0.3)
    ax.scatter(vj[idx][j], uv[idx][j], alpha=0.8,c=np.log10(MLv)[idx][j], vmin=-1, vmax=1.5, edgecolor='k', s=70)
    ax.set_xlim(-0.5,5); ax.set_ylim(-0.5,5)
    
    fout = catIO.Table('/Users/brammer/3DHST/Spectra/Release/v4.1.5/FullRelease/goodss_3dhst_v4.1.5_catalogs/goodss_3dhst.v4.1.5.zbest.fout')
    
    # run
    iter = -1
    
    for iter in range(7):
        self.iterate_zp_templates(idx=idx, update_templates=(iter > 1), iter=iter, error_residuals=(iter > 0), save_templates=(iter > 8), n_proc=4)
        self.zp[-3] = 1.0
        self.write_zeropoint_file('zphot.zeropoint.FSPS')
    
    # Show fits
    if False:
        ids = self.cat['id'][idx]
        j = -1

        j+=1; id = ids[j]; fig = self.show_fit(id, show_fnu=show_fnu, xlim=[0.1,10])
     
    ## Full Run
    idx = np.arange(self.NOBJ)[use]
    
    self.fit_parallel(idx=idx, n_proc=4)
    self.best_fit()
    self.error_residuals()
    
    self.fit_parallel(idx=idx, n_proc=4)
    self.best_fit()
    rf_tempfilt, f_rest = self.rest_frame_fluxes(f_numbers=[153,155,161], pad_width=0.5, percentiles=[2.5,16,50,84,97.5])
    
    hdu = fits.HDUList()
    hdu.append(fits.ImageHDU(data=self.zp, name='zp'))
    hdu.append(fits.ImageHDU(data=self.zbest, name='zbest'))
    hdu.append(fits.ImageHDU(data=self.chi_best, name='chibest'))
    hdu.append(fits.ImageHDU(data=self.fit_chi2, name='fit_chi2'))
    hdu.append(fits.ImageHDU(data=self.fobs, name='fobs'))
    hdu.writeto('eazy_output.fits', clobber=True)
    
    np.save('eazy_templates.npy', [self.templates])
    
    ## NUV mag
    idx = np.arange(self.NOBJ)[use & (mag < 23) & (self.zbest < 1)]
    #plt.scatter(nuv_obs[idx]*self.zp[-1]*self.ext_corr[-1], nuv_mod[idx], alpha=0.1)
    #plt.xlim(0.01,100); plt.ylim(0.01, 100); plt.plot([0.01,100], [0.01,100], color='r'); plt.loglog()
    
    plt.hist(25-2.5*np.log10(nuv_obs[idx]*self.zp[-1]), bins=100, range=[18,28], alpha=0.5, log=True)
    plt.hist(25-2.5*np.log10(nuv_mod[idx]/self.ext_corr[-1]), bins=100, range=[18,28], alpha=0.5, log=True)
    
    nuv_mag = 25-2.5*np.log10(nuv_mod/self.ext_corr[-1]*totcorr)
    idx = use & (nuv_mag < 22) & (self.zbest > 0.8)
    
    