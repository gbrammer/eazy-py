def build_matrices():
    """
    Build matrices for Blanton & Roweis optimization
    """ 
    #import unicorn
    import numpy as np
    #import br07
    import glob
    #from threedhst import catIO
    import matplotlib.pyplot as plt
    
    from astropy.table import Table
    import eazy
    
    from eazy.utils import running_median, nmad
    
    # UltraVista
    self = eazy.photoz.PhotoZ(param_file='zphot.param.new', translate_file='zphot.translate.new', zeropoint_file='zphot.zeropoint.FSPS_full')
    zbest = np.load('zbest_Nov3.npy')[0]
    self.best_fit(zbest=zbest)
    
    # FAST
    fout = Table.read('UVISTA_DR3_master_v1.1.fout', format='ascii.commented_header')
    
    ## RF fluxes
    if False:
        rf_tempfilt, f_rest = self.rest_frame_fluxes(f_numbers=[153,155,161], pad_width=0.1, percentiles=[2.5,16,50,84,97.5])
        np.save('uvista_rest_frame.npy', [rf_tempfilt, f_rest, idx])
    else:
        rf_tempfilt, f_rest, idx0 = np.load('uvista_rest_frame.npy')
    
    uv_data = -2.5*np.log10(f_rest[:, 0, 2] / f_rest[:, 1, 2])
    vj_data = -2.5*np.log10(f_rest[:, 1, 2] / f_rest[:, 2, 2])
    
    u_0 = rf_tempfilt.tempfilt[:,0] / rf_tempfilt.tempfilt[:,2]
    v_0 = rf_tempfilt.tempfilt[:,1] / rf_tempfilt.tempfilt[:,2]
    j_0 = rf_tempfilt.tempfilt[:,2] / rf_tempfilt.tempfilt[:,2]
        
    uv_temp = -2.5*np.log10(u_0 / v_0)
    vj_temp = -2.5*np.log10(v_0 / j_0)
    
    ## Template data
    lines = open('templates/fsps_full/tweak_spectra.param').readlines()
    templ_names = [templ.name.split('.dat')[0] for templ in self.templates]
    data = []
    for line in lines:
        if line.startswith('# '):
            data.append(line.split()[1:])
    columns = ['id', 'label', 'age', 'mass', 'sfr', 'Av', 'Ha_flux', 'Ha_cont']
    templ_data = Table(data=np.array(data), names=columns, dtype=[int, str, float, float, float, float, float, float])
    
    zsp_only=True
    
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
    
    idx_zsp = np.arange(self.NOBJ)[idx]
        
    ############# Refine photo-zs
    iter = 200
    iter = 300 # use Nov4 12 templates
    error_residuals, update_templates, update_zeropoints = True, False, True
    for i in range(6):
        self.iterate_zp_templates(idx=idx_zsp, update_templates=(i > 2), iter=iter+i, error_residuals=True, update_zeropoints=True, save_templates=(i > 2), n_proc=6)
        self.write_zeropoint_file('zphot.zeropoint.FSPS_full_12')
        #np.save('zbest.npy', [self.zbest])
    
    #############
    ## Fit full sample without updating zeropoints, etc.
    zp_ix = self.f_numbers == self.param['PRIOR_FILTER']
    mag = self.param['PRIOR_ABZP'] - np.squeeze(2.5*np.log10(self.fnu[:, zp_ix]))
    
    idx = (mag < 23) | ((self.cat['z_spec'] > 0) & np.isfinite(mag))
    if 'USE' in self.cat.colnames:
        # UltraVISTA
        use = (self.cat['USE'] == 1) & (self.cat['star'] == 0)
        use = (self.cat['contamination'] == 0) & (self.cat['nan_contam'] == 0) & (self.cat['star'] == 0)
    else:
        use = (self.cat['use'] == 1) & (self.cat['star_flag'] != 1)
        
    idx &= use
    
    ## Minimum number of OK data
    nfilt = ((self.fnu > 0) & (self.efnu > 0) & (self.fnu/self.efnu > 3)).sum(axis=1)
    idx &= nfilt > 6
    
    # reasonable colors
    idx &= np.isfinite(uv_data) & np.isfinite(vj_data) & (uv_data < 3) & (uv_data > -0.4) & (vj_data > -0.4) & (vj_data < 2.7) #quiescent
    
    #idx &= (fout['lssfr'] < -11) & (fout['Av'] < 0.5) & (vj_data > 0.5) & np.isfinite(uv_data) & np.isfinite(vj_data) & (uv_data < 2.4) #quiescent
    
    idx_full = np.arange(self.NOBJ)[idx]
    idx = idx_full
    
    ## Refit last photo-zs
    if False:
        self.iterate_zp_templates(idx=idx, update_templates=False, iter=206, error_residuals=True, update_zeropoints=False, save_templates=False, n_proc=6)
        self.iterate_zp_templates(idx=idx, update_templates=False, iter=207, error_residuals=True, update_zeropoints=False, save_templates=False, n_proc=6)
        np.save('zbest_Nov3.npy', [self.zbest])
    
    ##############
    ## Build NMF matrix
    izbest = np.cast[int](np.round(np.interp(self.zbest, self.tempfilt.zgrid, np.arange(self.NZ))))
    
    NOBJ = len(idx)
    data = np.zeros((NOBJ, self.NFILT, self.NZ))
    data_ivar = np.zeros((NOBJ, self.NFILT, self.NZ))
    
    for i in range(len(idx)):
        print(i)
        ix = idx[i]
        iz = izbest[ix]
        
        flu = self.fnu[ix,:]*self.zp*self.ext_corr
        ivar = 1/((self.efnu[ix,:]*self.zp*self.ext_corr)**2 + (0.02*flu)**2)
        
        ok_i = (self.fnu[ix,:] > 0) & (self.efnu[ix,:] > 0)
        data[i, ok_i, iz] = flu[ok_i]
        data_ivar[i, ok_i, iz] = ivar[ok_i]
            
    ### Show UVJ
    if False:
        
        if True:
            plt.scatter(vj_data[idx], uv_data[idx], c=fout['lssfr'][idx], vmin=-12, vmax=-8, marker='s', alpha=0.03, cmap='jet_r', edgecolor='None')
            #plt.scatter(vj_data[idx], uv_data[idx], c=fout['lmass'][idx], vmin=9, vmax=11, marker='s', alpha=0.03, cmap='jet_r', edgecolor='None')

        else:
            plt.scatter(vj_data[idx], uv_data[idx], color='k', marker='.', alpha=0.03)
        
        templ_ssfr = np.log10(templ_data['sfr']/templ_data['mass'])
        plt.scatter(vj_temp, uv_temp, c=templ_ssfr, vmin=-12, vmax=-8, marker='s', alpha=0.8, edgecolor='None', cmap='jet_r')
        plt.xlim(-0.3, 3.5); plt.ylim(0,3.5)
        
    ### Generate NMF components
    # Normalize template array by J band
    M = np.zeros((self.NTEMP, self.NFILT, self.NZ))
    for i in range(self.NTEMP):
        M[i,:,:] = self.tempfilt.tempfilt[:,i,:].T / rf_tempfilt.tempfilt[i,2]

    ok_temp = templ_data['Av'] > -1
    n_components = 12
    
    ## Fulls set, 12 templates, coeffs_blue
    n_components = 8
    ok_temp = ~((templ_ssfr < -10) & (templ_data['Av'] <= 1))
    
    # Quiescent templates, 4 templates, coeffs_red
    if True:
        ok_temp = (templ_ssfr < -10) & (templ_data['Av'] <= 1)
        n_components = 4

    M = M[ok_temp, :, :]    
    ds = data.shape
    ms = M.shape
        
    delta = ms[0] // n_components+1
    
    coeffs = np.ones((n_components, ms[0]))*1./delta/(n_components*5)
    for i in range(n_components):
        coeffs[i, i*delta:(i+1)*delta] = 1./delta
    
    so = np.argsort(vj_temp[ok_temp])
    coeffs[:,so] = coeffs*1
    
    coeffsx = np.ones((n_components, self.NTEMP))*1./delta/(n_components*5)
    coeffsx[:, ok_temp] += coeffs
    
    ## Split on quiescent templates for initialization
    coeffs = np.vstack((coeffs_red, coeffs_blue))
    
    templates = np.ones((ds[0], n_components))
    
    ### Loop so you can break in and still have the coeffs updated.  
    ### total of 4000 iters
    plt.scatter(vj_data[idx], uv_data[idx], color='k', marker='.', alpha=0.03)
    #plt.scatter(vj_temp, uv_temp, color='r', marker='.', alpha=0.8)
    plt.xlim(-0.3, 3.5); plt.ylim(0,3.5)
    
    plt.scatter(-2.5*np.log10(np.dot(coeffs, v_0[ok_temp])/np.dot(coeffs, j_0[ok_temp])), -2.5*np.log10(np.dot(coeffs, u_0[ok_temp])/np.dot(coeffs, v_0[ok_temp])), color='orange', alpha=0.5, s=40, marker='s')
    
    from templates import nmf_sparse
    
    for i in range(1, 100):
        print('Restart #{0}'.format(i))
        coeffs, templates = nmf_sparse(data.reshape((ds[0],-1)), data_ivar.reshape((ds[0],-1)), M.reshape((ms[0],-1)), n_components=n_components, NITER=40, tol_limit=1.e-5, coeffs=coeffs*1, templates=templates*1.)
        
        np.save('full_nmf_coeffs.npy', [[i], idx, coeffs, templates])
        
        if True:
            plt.scatter(-2.5*np.log10(np.dot(coeffs, v_0[ok_temp])/np.dot(coeffs, j_0[ok_temp])), -2.5*np.log10(np.dot(coeffs, u_0[ok_temp])/np.dot(coeffs, v_0[ok_temp])), color='r', alpha=0.3, s=40, marker='.')
            plt.savefig('full_nmf_iter_{0:03d}.png'.format(i))
    
    ##################################
    ### Best combined templates
    i, idx0, coeffs, templates = np.load('full_nmf_coeffs.npy')
    #i, idx0, coeffs, templates = np.load('nmf_coeffs.npy')
    templ_array = np.array([templ.flux for templ in self.templates])
        
    vj0 = -2.5*np.log10(np.dot(coeffs, v_0)/np.dot(coeffs, j_0))
    uv0 = -2.5*np.log10(np.dot(coeffs, u_0)/np.dot(coeffs, v_0))
    
    plt.scatter(vj_data[idx], uv_data[idx], color='k', marker='.', alpha=0.03)
    plt.xlim(-0.4, 2.9); plt.ylim(-0.4,2.9)
    #plt.scatter(vj0, uv0, color='r', marker='s')
    
    # Put normalization back in 
    coeffs_norm = coeffs/rf_tempfilt.tempfilt[:,2]
    coeffs_norm = (coeffs_norm.T / coeffs_norm.sum(axis=1)).T
    
    # V-band
    nmf_norm = np.dot(coeffs_norm, rf_tempfilt.tempfilt[:,1])
    coeffs_norm = (coeffs_norm.T / nmf_norm).T
    
    nmf_templates = np.dot(coeffs_norm, templ_array).T
    
    ## Effective Av
    alpha = 10**(-0.4*templ_data['Av'])
    fv = np.dot(coeffs_norm, rf_tempfilt.tempfilt[:,1])
    fv_corr = np.dot(coeffs_norm, rf_tempfilt.tempfilt[:,1]/alpha)
    Av = -2.5*np.log10(fv/fv_corr)
    
    ## Other parameters
    mass = np.dot(coeffs_norm, templ_data['mass'])
    sfr = np.dot(coeffs_norm, templ_data['sfr'])
    ssfr = sfr/mass
    
    Ha_flux = np.dot(coeffs_norm, templ_data['Ha_flux'])
    Ha_cont = np.dot(coeffs_norm, templ_data['Ha_cont'])
    
    Ha_EW = Ha_flux / Ha_cont
    
    NTEMP = coeffs.shape[0]
    
    nu_v = 3.e8/5000.e-10
    Lv = fv*nu_v
    
    fp = open('templates/uvista_nmf/spectra_12_Nov4.param','w')
    fp.write('## i age mass sfr Av Ha_flux Ha_cont fnu_v\n')
    for i in range(NTEMP):
        temp_file = 'nmf_fsps_Nov4_{0:02d}.dat'.format(i+1)
        label = '# {id:2d} -1  {mass:5.4e}  {sfr:.3e}  {Av:.3f}  {Ha_flux:.3e}  {Ha_cont:.3e}  {fnu:.3e}'.format(id=i+1, mass=mass[i], sfr=sfr[i], Av=Av[i], Ha_flux=Ha_flux[i], Ha_cont=Ha_cont[i], fnu=fv[i])
        print(label)
        fp.write(label+'\n')
        fp.write('{0:2d}  templates/uvista_nmf/{1}  1.0 0 1.0\n'.format(i+1, temp_file))
        
        fpt = open('./templates/uvista_nmf/{0}'.format(temp_file), 'w')
        fpt.write('#  i age mass sfr Av Ha_flux Ha_cont fnu_v\n')
        fpt.write(label+'\n')
        fpt.close()
        
        fpt = open('./templates/uvista_nmf/{0}'.format(temp_file), 'ab')
        np.savetxt(fpt, np.array([self.templates[0].wave, nmf_templates[:,i]]).T, fmt='%.6e')
        fpt.close()
        
    fp.close()
    
    ## Show the templates in UVJ: sSFR, MLv, Av, Ha EW
    sh = 0.8
    fig = plt.figure(figsize=[11,8])
    
    ax = fig.add_subplot(221)
    ax.scatter(vj_data[idx], uv_data[idx], color='k', marker='.', alpha=0.01)
    ax.set_xlim(-0.4, 2.9); ax.set_ylim(-0.4,2.9)
    sc = ax.scatter(vj0, uv0, marker='s', c=np.log10(ssfr), s=100, vmin=-12, vmax=-8, cmap='jet_r')
    cb = plt.colorbar(sc, ax=ax, shrink=sh)
    cb.set_label('log sSFR')
    for i in range(NTEMP):
        ax.text(vj0[i], uv0[i]+0.1, '{0:d}'.format(i+1), ha='center', va='bottom', size=8, backgroundcolor='w')
        
    ax.grid()
    
    ax = fig.add_subplot(222)
    ax.scatter(vj_data[idx], uv_data[idx], color='k', marker='.', alpha=0.01)
    ax.set_xlim(-0.4, 2.9); ax.set_ylim(-0.4,2.9)
    sc = ax.scatter(vj0, uv0, marker='s', c=np.log10(mass/Lv), s=100, vmin=-1, vmax=1.5, cmap='jet')
    cb = plt.colorbar(sc, ax=ax, shrink=sh)
    cb.set_label('log M/Lv')
    ax.grid()
    
    ax = fig.add_subplot(223)
    ax.scatter(vj_data[idx], uv_data[idx], color='k', marker='.', alpha=0.01)
    ax.set_xlim(-0.4, 2.9); ax.set_ylim(-0.4,2.9)
    sc = ax.scatter(vj0, uv0, marker='s', c=Av, s=100, vmin=0, vmax=3, cmap='jet')
    cb = plt.colorbar(sc, ax=ax, shrink=sh)
    cb.set_label('Av')
    ax.grid()
    
    ax = fig.add_subplot(224)
    ax.scatter(vj_data[idx], uv_data[idx], color='k', marker='.', alpha=0.01)
    ax.set_xlim(-0.4, 2.9); ax.set_ylim(-0.4,2.9)
    sc = ax.scatter(vj0, uv0, marker='s', c=np.log10(Ha_flux/Ha_cont), s=100, vmin=1, vmax=3, cmap='jet_r')
    cb = plt.colorbar(sc, ax=ax, shrink=sh)
    cb.set_label(r'log H$\alpha$ EW')
    ax.grid()
    
    fig.tight_layout(pad=0.4)
    fig.savefig('spectra_12_Nov4.png')
    
    fig = plt.figure(figsize=[8,8])
    for i in range(NTEMP):
        ax = fig.add_subplot(3,4,1+i)
        ax.plot(self.templates[0].wave[::10], nmf_templates[::10,:]/(3.e18/5500.**2), color='k', alpha=0.1)
        ax.plot(self.templates[0].wave, nmf_templates[:,i]/(3.e18/5500.**2), color='k', alpha=0.8, linewidth=2)
        ax.semilogx()
        ax.set_xlim(1000,5.e4)
        ax.set_ylim(0,10)
        if i < 8:
            ax.set_xticklabels([])
        
        if (i % 4) > 0: 
            ax.set_yticklabels([])
        
        ax.text(0.9, 0.9, '#{0}'.format(i+1), transform=ax.transAxes, ha='right', va='top', size=12)
        
    fig.tight_layout(pad=0.1)
    fig.savefig('spectra_12_Nov4_SED.png')
    
def check_parameters():
    """
    Redshifts & SP parameters from the new NMF fits
    """
    self = eazy.photoz.PhotoZ(param_file='zphot.param.new', translate_file='zphot.translate.new', zeropoint_file='zphot.zeropoint.FSPS_full_12')
    
    zp_ix = self.f_numbers == self.param['PRIOR_FILTER']
    mag = self.param['PRIOR_ABZP'] - np.squeeze(2.5*np.log10(self.fnu[:, zp_ix]))
    
    idx = (mag < 23) | ((self.cat['z_spec'] > 0) & np.isfinite(mag))
    if 'USE' in self.cat.colnames:
        # UltraVISTA
        use = (self.cat['USE'] == 1) & (self.cat['star'] == 0)
        use = (self.cat['contamination'] == 0) & (self.cat['nan_contam'] == 0) & (self.cat['star'] == 0)
    else:
        use = (self.cat['use'] == 1) & (self.cat['star_flag'] != 1)
        
    idx &= use
    
    ## Minimum number of OK data
    nfilt = ((self.fnu > 0) & (self.efnu > 0) & (self.fnu/self.efnu > 3)).sum(axis=1)
    idx &= nfilt > 6
    
    #idx &= (fout['lssfr'] < -11) & (fout['Av'] < 0.5) & (vj_data > 0.5) & np.isfinite(uv_data) & np.isfinite(vj_data) & (uv_data < 2.4) #quiescent
    
    idx_full = np.arange(self.NOBJ)[idx]
    idx = idx_full
    
    ## Refit last photo-zs
    if True:
        self.iterate_zp_templates(idx=idx, update_templates=False, iter=401, error_residuals=True, update_zeropoints=False, save_templates=False, n_proc=6)
        self.iterate_zp_templates(idx=idx, update_templates=False, iter=402, error_residuals=True, update_zeropoints=False, save_templates=False, n_proc=6)
        np.save('zbest_Nov4_12.npy', [self.zbest])
    
    rf_tempfilt, f_rest = self.rest_frame_fluxes(f_numbers=[153,155,161], pad_width=0.1, percentiles=[2.5,16,50,84,97.5])
    
############################################
# def old_build_matrices(NTEMP=5, field='cosmos', hmax=24, zmax=2.5):
#     """
#     Build matrices for Blanton & Roweis optimization
#     """ 
#     #import unicorn
#     import numpy as np
#     #import br07
#     import glob
#     #from threedhst import catIO
#     from astropy.table import Table
#     import eazy
#     
#     from eazy.utils import running_median, nmad
#     
#     # UltraVista
#     self = eazy.photoz.PhotoZ(param_file='zphot.param.new', translate_file='zphot.translate.new', zeropoint_file='zphot.zeropoint.new')
#     zbest = np.load('zbest.npy')[0]
#     self.best_fit(zbest=zbest)
#     
#     zsp_only=False
#     
#     zp_ix = self.f_numbers == self.param['PRIOR_FILTER']
#     mag = self.param['PRIOR_ABZP'] - np.squeeze(2.5*np.log10(self.fnu[:, zp_ix]))
#     
#     idx = (mag < 23) | ((self.cat['z_spec'] > 0) & np.isfinite(mag))
#     #idx = (mag < 22) | ((self.cat['z_spec'] > 0) & np.isfinite(mag))
#     if zsp_only:
#         idx &= (self.cat['z_spec'] > 0)
#         
#     if 'USE' in self.cat.colnames:
#         # UltraVISTA
#         use = (self.cat['USE'] == 1) & (self.cat['star'] == 0)
#         use = (self.cat['contamination'] == 0) & (self.cat['nan_contam'] == 0) & (self.cat['star'] == 0)
#     else:
#         use = (self.cat['use'] == 1) & (self.cat['star_flag'] != 1)
#         
#     idx &= use
#     
#     idx = np.arange(self.NOBJ)[idx]
#     
#     iter = 200
#     error_residuals, update_templates, update_zeropoints = True, False, True
#     
#     for i in range(3):
#         self.iterate_zp_templates(idx=idx, update_templates=update_templates, iter=iter+i, error_residuals=error_residuals, update_zeropoints=update_zeropoints, save_templates=True, n_proc=6)
#         self.write_zeropoint_file('zphot.zeropoint.FSPS_full')
#         np.save('zbest.npy', [self.zbest])
#         
#     ### Fit all objects    
#     idx_full = np.arange(self.NOBJ)
#     error_residuals, update_templates, update_zeropoints = True, False, False
#     iter = 99
#     self.iterate_zp_templates(idx=idx_full, update_templates=update_templates, iter=iter, error_residuals=error_residuals, update_zeropoints=update_zeropoints, save_templates=False, n_proc=6)
#     
#     ### Bin on U-V vs sSFR
#     plt.scatter(np.log10(ssfr[idx]), uv[idx], c=vj[idx], vmin=-0.5, vmax=2, alpha=0.2, edgecolor='0.5')
#     
#     xh, yh = np.clip(np.log10(ssfr[idx]), -12.95, -8.05), np.clip(uv[idx], 0.01, 2.49)
#     h2 = np.histogram2d(xh, yh, range=[(-13,-8), (0,2.5)], bins=[8,8])
# 
#     xh, yh = np.clip(vj[idx], 0.01, 2.99), np.clip(uv[idx], 0.01, 3.49)
#     h2 = np.histogram2d(xh, yh, range=[(0,3), (0,3.5)], bins=[16,16])
#     
#     ## Binned rest-frame SEDs
#     #full_templates = self.param.read_templates(templates_file='templates/fsps_full/tweak_spectra.param')
#     full_templates = self.param.read_templates(templates_file='templates/fsps_full/spectra.param')
#     full_tempfilt = eazy.photoz.TemplateGrid(self.zgrid, full_templates, self.param['FILTERS_RES'], self.f_numbers, add_igm=True, galactic_ebv=self.param.params['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'], n_proc=0)
#     full_rf_tempfilt = eazy.photoz.TemplateGrid(np.array([0,0.1]), full_templates, self.param['FILTERS_RES'], np.array([153,155,161]), add_igm=False, galactic_ebv=0, Eb=self.param['SCALE_2175_BUMP'], n_proc=-1)
#     full_rf_tempfilt.tempfilt = np.squeeze(full_rf_tempfilt.tempfilt[0,:,:])
#         
#     ix = np.argmax(h2[0])
#     j, i = np.unravel_index(ix, h2[0].shape)
#     
#     Q = ssfr[idx] > -10
#     testQ = Q
#     
#     sel = (xh >= h2[1][j]) & (xh <= h2[1][j+1]) & (yh >= h2[2][i]) & (yh <= h2[2][i+1]) & testQ
#     
#     fnu_corr = self.fnu[idx[sel],:]*self.zp*self.ext_corr
#     efnu_corr = self.efnu[idx[sel],:]*self.zp*self.ext_corr
# 
#     #irest = 1
#     fnu_corr = (fnu_corr.T/f_rest[idx[sel],irest,2]).T
#     efnu_corr = (efnu_corr.T/f_rest[idx[sel],irest,2]).T
#     lc_rest = (self.lc[:,np.newaxis]/(1+self.zbest[idx[sel]])).T
#     
#     templ = np.zeros((sel.sum(), self.NFILT, full_tempfilt.NTEMP))
#     
#     import specutils.extinction
#     import astropy.units as u
#     f99 = specutils.extinction.ExtinctionF99(a_v = self.tempfilt.galactic_ebv * 3.1)
#     fred = 10**(-0.4*f99(full_rf_tempfilt.lc[irest]*(1+self.zbest[idx][sel])*u.AA))
#     
#     for ii in range(sel.sum()):
#         zi = self.zbest[idx][sel][ii]
#         templ[ii,:,:] = (full_tempfilt(zi).T/(full_rf_tempfilt.tempfilt[:,irest]*fred[ii]))
#     
#     be = efnu_corr.flatten()
#     A = templ.reshape((-1,full_tempfilt.NTEMP))
#     b = fnu_corr.flatten()/be
#     
#     ok = (self.fnu[idx[sel],:].flatten() > -99) & (self.efnu[idx[sel],:].flatten() > 0) & (lc_rest.flatten() > 1300)
# 
#     sh = self.fnu[idx[sel],:].shape
#     oksh = ok.reshape(sh)
#     
#     yy = (fnu_corr/self.ext_corr)/(lc_rest/rf_tempfilt.lc[irest])**2
#     xm, ym, ys, N = running_median(lc_rest[oksh], yy[oksh], use_median=True, use_nmad=True, NBIN=100)
#     ym_i = np.interp(lc_rest, xm, ym)
#     ys_i = np.interp(lc_rest, xm, ys)
#     oksh &= np.abs(ym_i - yy) < 3*ys_i
#     ok = oksh.flatten()
#     
#     coeffs, resid = scipy.optimize.nnls((A[ok,:].T/be[ok]).T,b[ok])
#     # coeffs, resid, rank, s = np.linalg.lstsq((A[ok,:].T/be[ok]).T,b[ok])
#     # 
#     # amatrix = unicorn.utils_c.prepare_nmf_amatrix(be[ok]**2, A[ok,:].T)
#     # coeffs_nmf = unicorn.utils_c.run_nmf(fnu_corr.flatten()[ok], be[ok]**2, A[ok,:].T, amatrix, verbose=True, toler=1.e-5)
#     
#     best = np.dot(A, coeffs)
# 
#     yym = (best.reshape(sh)/self.ext_corr)/(lc_rest/rf_tempfilt.lc[irest])**2
#     xmm, ymm, ysm, Nm = running_median(lc_rest[oksh], yym[oksh], use_median=True, use_nmad=True, NBIN=100)
#     
#     if False:
#         plt.scatter(lc_rest[oksh], (fnu_corr/self.ext_corr)[oksh]/(lc_rest[oksh]/rf_tempfilt.lc[irest])**2, alpha=0.05*100/sel.sum(), color='k', marker='.')
#         plt.scatter(lc_rest[oksh], (best.reshape(sh)/self.ext_corr)[oksh]/(lc_rest[oksh]/rf_tempfilt.lc[irest])**2, alpha=0.05*100/sel.sum(), color='r', marker='.', zorder=2)
#    
#     plt.errorbar(xm, ym, ys, color='k')
#     plt.plot(xmm, ymm, color='r', marker='.', alpha=0.4)
#     plt.xlim(800,1.e5); plt.ylim(0.01,10); log()
#     
#     tf = np.array([full_templates[ii].flux / (full_rf_tempfilt.tempfilt[ii, irest]*3.e18/full_rf_tempfilt.lc[irest]**2) for ii in range(full_tempfilt.NTEMP)])
#     tt = np.dot(coeffs, tf)
#     plt.plot(full_templates[0].wave, tt, color='r')
#     
#     plt.scatter(lc_rest[oksh], fnu_corr[oksh]/best[ok], alpha=0.05*100/sel.sum(), color='k', marker='.')
#     plt.xlim(800,1.e5); plt.ylim(0.5,1.5); plt.semilogx()
    
    