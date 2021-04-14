"""
Making templates with SPS tools
"""

def agn_templates():
    
    import fsps
    import eazy
    import matplotlib.pyplot as plt
    
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
    """
    Make extreme starburst template
    """
    import eazy
    import grizli.utils
    
    sps = None
    
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
    
    ex = sps.ExtendedFsps(logzsol=-1, zcontinuous=True, add_neb_emission=True, sfh=4, tau=0.2)
    ex.set_fir_template()
    
    ex.params['logzsol'] = -1
    ex.params['gas_logz'] = -1
    ex.params['gas_logu'] = -2
    ex.params['tau'] = 0.3
    tage, Av = 0.2, 0.03

    templ = ex.get_full_spectrum(tage=tage, Av=Av, scale_lyman_series=0.1, set_all_templates=False)
    
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
    
def spline_sfh():
    """
    Generate a template set with spline-basis SFHs
    """
    import os
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    from grizli.utils import bspline_templates
    import eazy.sps
    
    # Grid encompassing FSPS.log_age    
    step = 0.022
    ages = 10**np.arange(np.log10(3.e-4), 
                          np.log10(14.12)+step, step)
    
    ages = np.arange(3.e-4, 14.127, 0.001)
    
    # Spline range between tmin, tmax
    tmin, tmax, Ns = 0.02, 8, 7
    
    # For Grid
    tmin, tmax, Ns = 0.03, 9, 21
    
    spl = bspline_templates(ages, get_matrix=True, 
                            log=True, df=Ns, clip=0.,
                            minmax=(np.log(tmin), np.log(tmax)))
    
    Na = len(ages)
    
    # Log-normals
    centers = [0.03, 0.1, 0.3, 0.8, 1.4, 3., 8.]
    widths = [0.21, 0.21, 0.21, 0.08, 0.08, 0.21, 0.21]
    
    centers = [0.03, 0.10, 0.25, 0.50, 0.8, 1.6, 4.]
    widths =  [0.17, 0.17, 0.17, 0.09, 0.09, 0.17, 0.17]
    
    # For grid
    if 0:
        centers = [0.03, 0.065, 0.1, 0.2, 0.3, 0.55, 0.8, 1.1, 1.4, 2.2, 3., 5., 8.]
        widths = np.diff(np.log10(centers))/2.35
        widths = np.append([widths[0]], widths)
    
    Ng = len(centers)
    gau = np.zeros((Na, Ng))    
    for i in range(Ng):
        log_age = np.log10(ages/centers[i])
        w = widths[i]
        gau[:,i] = 1./np.sqrt(2*np.pi*w**2)*np.exp(-log_age**2/2/w**2)
    
    spl = gau    
        
    age100 = ages <= 0.1
    
    sp = eazy.sps.ExtendedFsps(logzsol=0, zcontinuous=True, add_neb_emission=True, sfh=4)
    
    sp.set_fir_template()
    #sp.set_dust(dust_obj_type='KC13')    
    #sp.params['dust_index'] = 0. #-0.3
    sp.set_dust(dust_obj_type='WG00x')
    sp.params['imf_type'] = 1.
    sp.get_spectrum(tage=0.1)
    
    N = spl.shape[1]
    vband = ["v"]
    ssfr = np.zeros(N)

    logz = np.array([-1., -0.5, 0., 0., 0., 0., 0.])
    logu = np.array([-1.5, -2.0, -2.5, -2.5, -2.5, -2.5, -2.5])
    dusts = [[0.02, 0.5],
             [0.02, 0.5, 1.0, 2.0, 3.0, 5.],
             [0.02, 1., 2.0],
             [0.02, 1., 2.0],
             [0.02], 
             [0.02],
             [0.02]]
    
    # for i in [0,1,2,3,4]:
    #     dusts[i] = list(np.arange(0.2, 3.05-0.25*i, 0.1))
    
    # Full grid
    if N > 7:
        dusts = [[0.02, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4., 5., 6., 7., 8., 10., 12]]*N
        logz = np.zeros(N)
        logu = np.zeros(N)-2.5
        logz[:2] = -1.
        logu[:2] = -1.5

    bands = ['u','v','2mass_j', 'i1500', 'i2800']
    uvj_mags = np.zeros((N, len(bands)))
    band_ages = np.zeros((N, len(bands)))
    
    _ = sp.get_spectrum()  
    fsps_ages = 10**(sp.log_age-9)
    from grizli.utils_c.interp import interp_conserve_c
    
    norm_sfh = spl.T*0.
    ysfh = np.ones((N, len(fsps_ages)))
    
    # 30 Myr
    t0 = 0.03
    end_age = np.zeros(N)
    end_clip = 1.e-3
    
    sfh_interp = norm_sfh*0.
    
    for i in range(N):
        sfh = spl.T[i,:]*1.
        sfh_t0 = np.interp(t0, ages, sfh)
        #sfh[ages < t0] = sfh_t0
        sfh_i = sfh/np.trapz(sfh, ages*1.e9) 
        
        # Evaluate tage where sfr is down by some factor from the max        
        if sfh[-1]/sfh.max() > end_clip:
            tstart = ages[-1]
        else:
            tstart = ages[sfh/sfh.max() > end_clip].max()
        
        end_age[i] = tstart
        #end_age[i] = 14.12 #ages.max()
        
        # Constant SFR below t0 (30 Myr)
        sfr0 = np.interp(t0, ages, sfh_i)
        sfh_i[ages < t0] = sfr0
        sfr_min = sfh_i.max()/1000.
        imax = np.argmax(sfh_i)
        sfh_i[(sfh_i < sfr_min) & (ages < ages[imax])] = sfr_min
        
        if 1:
            # Now evaluate SFH in "forward" direction,rather than lookback
            tforward = end_age[i]-ages[::-1]
            clip = tforward > 0
            sfh_i[~clip[::-1]] = sfr_min
            norm_sfh[i,:] = sfh_i*1.
                
            sfh = np.interp(ages, tforward, norm_sfh[i,::-1], left=0., right=sfr0) #, integrate=0)
            sfh[sfh == 0] = sfr_min
            sfh_interp[i,:] = sfh
            plt.plot(ages, sfh)
        else:
            sfh_interp[i,:] = sfh_i[::-1]
            norm_sfh[i,:] = sfh_i*1.
            plt.plot(ages, sfh_i)
        
        sp.params['sfh'] = 3
        sp.set_tabular_sfh(ages, sfh_interp[i,:]) 
        sp.params['logzsol'] = logz[i]
        sp.params['gas_logz'] = logz[i]
        sp.params['gas_logu'] = logu[i]
    
        sp.params['compute_light_ages'] = True
        band_ages[i,:] = sp.get_mags(tage=end_age[i], bands=bands)

        sp.params['compute_light_ages'] = False
        uvj_mags[i,:] = sp.get_mags(tage=end_age[i], bands=bands)
        ssfr[i] = sp.sfr100/sp.stellar_mass
    
    if False:
        #plt.figure()
        plt.plot(uvj_mags[:,1] - uvj_mags[:,2], uvj_mags[:,0]-uvj_mags[:,1], color='w', alpha=1., zorder=100)
        plt.plot(uvj_mags[:,1] - uvj_mags[:,2], uvj_mags[:,0]-uvj_mags[:,1], color='0.5', alpha=0.5, zorder=101)
        plt.scatter(uvj_mags[:,1] - uvj_mags[:,2], uvj_mags[:,0]-uvj_mags[:,1], c=np.log10(np.maximum(ssfr, 1.e-12)), vmin=-12, vmax=-8, zorder=1000, marker='s', s=100, edgecolor='w')
    
    
    ### Make all templates
    res = eazy.filters.FilterFile('FILTER.RES.latest')
    uvj_res = [res[153],res[155],res[161]]
    breakme=False
        
    templates = []
    sp.params['dust2'] = 0.
    sp.params['sfh'] = 3.
    sp.params['compute_light_ages'] = False
    kwargs = {'scale_lyman_series':0.1}
    
    plt.figure()
    
    for i in range(N):
        #sp.set_tabular_sfh(ages, sfh_interp[i,:]) 
        sp.set_tabular_sfh(ages, sfh_interp[i,:]) 
        
        sp.params['logzsol'] = logz[i]
        sp.params['gas_logz'] = logz[i]
        sp.params['gas_logu'] = logu[i]
        
        ssfr_i = sp.sfr100/sp.stellar_mass
        
        for Av in dusts[i]:
            templ = sp.get_full_spectrum(tage=end_age[i], Av=Av,
                                         get_template=True, 
                                         set_all_templates=False, **kwargs)
            
            templ.ageV = band_ages[i,1]
            jflux = templ.integrate_filter(uvj_res[2], flam=True)
            templates.append(templ)
            plt.plot(templ.wave, templ.flux/jflux, label=templ.name, alpha=0.8)
        
        if (i == 0) & breakme:
            break
    
    plt.loglog()
    
    fig = plt.figure()
    plt.scatter(vj[sel], uv[sel], c=np.log10(zout['SFR']/zout['mass'])[sel], vmin=-12, vmax=-8) 
    uvj = np.array([templ.integrate_filter_list(uvj_res) for templ in templates])
    ssfr_temp = np.array([templ.meta['sfr100']/templ.meta['stellar_mass'] for templ in templates])
    plt.scatter(-2.5*np.log10(uvj[:,1]/uvj[:,2]), -2.5*np.log10(uvj[:,0]/uvj[:,1]), c=np.log10(ssfr_temp), vmin=-12, vmax=-8, marker='s', edgecolor='w', s=100)
    
    # Try NMF decomposition
    if False:
        from sklearn.decomposition import NMF
        model = NMF(n_components=5, init='nndsvda', random_state=0, verbose=True, solver='cd', tol=1.e-6, alpha=0.01, beta_loss='frobenius', max_iter=100000)
        clip = (templ.wave > 2500) & (templ.wave < 1.6e4)
        X = [templ.flux/(uvj[i,1]*3.e18/5500**2) for i, templ in enumerate(templates)]
        dust_array = np.hstack(dusts)
        ix = np.array([1,3,4,5,6,9,10,12,13]) 
        ix = np.where((dust_array > 0.1) & (dust_array < 4))[0]
    
        W = model.fit_transform(np.array(X)[ix,clip].T)
        H = model.components_
        uvjr = H.dot((uvj.T/uvj[:,1]).T[ix,:])

        plt.scatter(-2.5*np.log10(uvj[ix,1]/uvj[ix,2]), -2.5*np.log10(uvj[ix,0]/uvj[ix,1]), c=np.log10(ssfr_temp)[ix], vmin=-12, vmax=-8, marker='s', edgecolor='r', s=100)
        plt.scatter(-2.5*np.log10(uvjr[:,1]/uvjr[:,2]), -2.5*np.log10(uvjr[:,0]/uvjr[:,1]), vmin=-12, vmax=-8, marker='v', edgecolor='w', s=100, alpha=0.8)
    
    # Write templates
    if N > 8:
        param_file = 'templates/spline_templates/spline.grid.param'
    else:
        param_file = 'templates/spline_templates/spline.param'

    fp = open(param_file,'w')
    for i, templ in enumerate(templates):
        tab = templ.to_table()
        tab.meta['ageV'] = templ.ageV
        name = 'spline_age{0:4.2f}_av{1:3.1f}'.format(tab.meta['ageV'], tab.meta['Av'])
        line = f'{i+1} templates/spline_templates/{name}.fits 1.0'
        print(line)
        fp.write(line+'\n')
        tab.write(f'templates/spline_templates/{name}.fits', overwrite=True)
    
    fp.close()
    
    # Metadata
    cols = ('file','Av','mass','Lv','sfr','LIR','energy_abs','ageV')
    rows = []
    for i, templ in enumerate(templates):
        vflux = templ.integrate_filter(uvj_res[1], flam=False)
        Lv = vflux*3.e18/uvj_res[1].pivot
        tab = templ.to_table()
        name = 'spline_age{0:4.2f}_av{1:3.1f}'.format(templ.ageV, templ.meta['Av'])
        row = [name, templ.meta['Av'], templ.meta['stellar_mass'], Lv, templ.meta['sfr100'], templ.meta['energy_absorbed'], templ.meta['energy_absorbed'], templ.ageV]
        rows.append(row)
    
    par = utils.GTable(names=cols, rows=rows)
    for line in ['Ha','O3','Hb','O2','Lya']:
        par['line_flux_'+line] = 0.
        par['line_C_'+line] = 0.
        par['line_EW_'+line] = 0.
        
    par.write(param_file+'.fits', overwrite=True)
    
    # color space spanned by random draws
    NT = len(templates)
    rnd = 10**(np.random.rand(1000*NT, NT)-3)
    for i in range(NT):
        rnd[i*1000:(i+1)*1000,i] = 1.
        
    uvjr = rnd.dot((uvj.T/uvj[:,0]).T)
    plt.scatter(-2.5*np.log10(uvjr[:,1]/uvjr[:,2]), -2.5*np.log10(uvjr[:,0]/uvjr[:,1]), c='k', marker='.', alpha=0.1)

    # Interpolation not working?
    for i in range(N):
        sp.set_tabular_sfh(ages, sfh_interp[i,:]) 

        #sp.set_tabular_sfh(fsps_ages, ysfh[i,:])
        _ = sp.get_spectrum()
    
        if 0:
            # Show in lookback time
            plt.plot(ages, norm_sfh[i,:], color='r', alpha=0.5)
            plt.plot(end_age[i]-10**(sp.log_age-9), sp.sfr, color='k', alpha=0.5) 
            plt.plot(end_age[i]-10**(sp.log_age-9), ysfh[i,:], color='g', alpha=0.5) 
            plt.vlines(end_age[i], 1.e-30, 10, color='b', alpha=0.5)
            sp.get_spectrum(tage=end_age[i], zmet=None)
            plt.scatter(0.01, sp.sfr, marker='o', color='b')
        else:
            _ = sp.get_spectrum()
            plt.plot(10**(sp.log_age-9), sp.sfr, color='k', alpha=0.5) 
            plt.plot(end_age[i]-ages, norm_sfh[i,:], color='r', alpha=0.5)
            plt.vlines(end_age[i], 1.e-30, 10, color='b', alpha=0.5)
            sp.get_spectrum(tage=end_age[i], zmet=None)
            plt.scatter(end_age[i], sp.sfr, marker='o', color='b')
        
            #_ = sp.get_spectrum()
            #plt.plot(10**(sp.log_age-9), sp.sfr/np.interp(end_age[i], fsps_ages, sp.sfr)*np.interp(end_age[i], end_age[i]-ages[::-1], norm_sfh[i,::-1]), alpha=0.5) 
        
    #####################
    # Do light-weighted ages add linearly? - YES!
    ij = [1,3]
    coeffs = np.array([0.2, 0.5])
 
    sfh = sfh_i*0.   
    for k in range(len(ij)):
        i = ij[k]
        sfh += coeffs[k]*spl.T[i,:]/np.trapz(spl.T[i,:], ages*1.e9) 
    
    sp.set_tabular_sfh(13-ages[::-1], sfh[::-1]) 
    sp.params['compute_light_ages'] = True
    combined_age = sp.get_mags(tage=13, bands=vband)
    
    vflux = 10**(-0.4*uvj_mags[ij,1])
    c_norm = (coeffs*vflux); c_norm /= c_norm.sum() 
    coeffs_age = vband_ages[ij].dot(c_norm)
    print(combined_age, coeffs_age)
    
    
    
def fit_dust_wg00():
    """
    Fit WG00 as flexible model
    """
    from dust_attenuation import averages, shapes, radiative_transfer
    from astropy.modeling.fitting import LevMarLSQFitter
    #from importlib import reload
    #reload(averages); reload(shapes)
    #reload(averages); reload(shapes)
    
    shapes.x_range_N09 = [0.01, 1000] 
    averages.x_range_C00 = [0.01, 1000]
    averages.x_range_L02 = [0.01, 0.18]

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

def test_fit_phot():

    import matplotlib.pyplot as plt
    import eazy.sps
    # dummy vars
    ids = []
    i = -1
    
    sp = eazy.sps.ExtendedFsps(logzsol=0, zcontinuous=True, add_neb_emission=True, sfh=4)
    sp.set_fir_template()
    #sp.set_dust(dust_obj_type='KC13')    
    #sp.params['dust_index'] = 0. #-0.3
    sp.set_dust(dust_obj_type='WG00x')
    sp.params['imf_type'] = 1.
    __ = sp.get_full_spectrum(tage=0.1)
    
    sp.set_lognormal_sfh(lognorm_center=-1, lognorm_logwidth=0.1, verbose=True) 
    sp.params['sfh'] = 3
    sp.is_lognorm_sfh = True
    sp.set_lognormal_sfh(lognorm_center=-1, lognorm_logwidth=0.1, verbose=True) 

    plist = ['tage','Av']
    func_kwargs = {'lorentz': False, 'emwave': [1216.0, 6564], 'scale_lyman_series': 0.0}
    lsq_kwargs = {'method': 'trf', 'max_nfev': 200, 'loss': 'huber', 'x_scale': 1.0, 'verbose': True, 'diff_step': [0.05, 0.05]}
    
    TEF = self.TEF
    
    i+=1
    id = ids[i]
    show_fnu=True
    zshow = None
    show_components = True
    maglim = (27.2,19.5)
    
    _ = self.show_fit(id, show_components=show_components, show_fnu=show_fnu, maglim=maglim, zshow=zshow, showpz=False)
    
    ax = _[0].axes[0]
    
    is_flam = (show_fnu < 1)
    
    z=_[1]['z']
    sp.params['zred'] = z
    eazy.sps.BOUNDS['zred'] = [z-0.2, z+0.2, 0.01]
    
    phot_dict = _[1]
    filters = self.filters
    
    _phot = sp.fit_phot(phot_dict, filters, plist=plist, func_kwargs=func_kwargs, verbose=True, lsq_kwargs=lsq_kwargs, show=ax, TEF=TEF)
    _phot['z'] = sp.params['zred']
    _phot['age'] = _phot['age_i1500']*1000
    for k in sp.meta:
        _phot[k] = sp.meta[k]
        
    label = 'logm={log_mass:.2f}\n lw_age={age:.1f} Myr\n Av={Av:.1f} \n logzsol={logzsol:.1f} gas_logz={gas_logz:.1f} gas_logu={gas_logu:.1f}'.format(**_phot)
    templ = _phot['templ']
    iz = templ.zindex(_phot['z'])
    
    igm = templ.igm_absorption(_phot['z'], scale_tau=1.4)
    ax.plot(templ.wave*(1+_phot['z'])/1.e4, templ.flux_fnu(iz)*_phot['scale']*igm, label=label, color='r', alpha=0.5)
    ax.legend()
    # ax.set_xlim(0.3, 3000)
    # ax.set_ylim(0.05, 5000)
    # if id == 319:
    #     ax.errorbar(1200., 460, 80, color='r', marker='o', alpha=0.5)
    # else:
    #     ax.errorbar(1200., 230, 80, color='r', marker='o', alpha=0.5)
    
    ax.set_ylim(-0.6, 10)  
    ax.set_xlim(0.45, 6)  
    _[0].tight_layout(pad=0.1)
    _[0].savefig(f'{root}_{id:05d}.fsps.png')
