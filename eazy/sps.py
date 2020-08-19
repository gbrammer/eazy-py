"""
Tools for making FSPS templates
"""
import os
from collections import OrderedDict

import numpy as np
import astropy.units as u
from astropy.cosmology import WMAP9

try:
    from dust_attenuation.baseclasses import BaseAttAvModel
except:
    BaseAttAvModel = object

from astropy.modeling import Parameter
import astropy.units as u

try:
    from fsps import StellarPopulation
except:
    # Broken, but imports
    StellarPopulation = object

from . import utils
from . import templates

DEFAULT_LABEL = 'fsps_tau{tau:3.1f}_logz{logzsol:4.2f}_tage{tage:4.2f}_av{Av:4.2f}'

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

class KC13(BaseAttAvModel):
    """
    Kriek & Conroy (2013) attenuation model, extends Noll 2009 with UV bump 
    amplitude correlated with the slope, delta.
    
    Slightly different from KC13 since the N09 model uses Leitherer (2002) 
    below 1500 Angstroms.
    
    """
    name = 'Kriek+Conroy2013'
    
    delta = Parameter(description="delta: slope of the power law",
                      default=0., min=-3., max=3.)
    
    extra_bump = 1.
    
    def _init_N09(self):
        from dust_attenuation import averages, shapes, radiative_transfer

        # Allow extrapolation
        shapes.x_range_N09 = [0.009, 2.e8] 
        averages.x_range_C00 = [0.009, 2.e8]
        averages.x_range_L02 = [0.009, 0.18]

        self.N09 = shapes.N09()
                
    def evaluate(self, x, Av, delta):
        import dust_attenuation
        
        if not hasattr(self, 'N09'):
            self._init_N09()
                    
        #Av = np.polyval(self.coeffs['Av'], tau_V)
        x0 = 0.2175
        gamma = 0.0350
        ampl = (0.85 - 1.9*delta)*self.extra_bump
        
        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.Angstrom
        else:
            xin = x
        
        if dust_attenuation.__version__ >= '0.0.dev131':                
            return self.N09.evaluate(xin, x0, gamma, ampl, delta, Av)
        else:
            return self.N09.evaluate(xin, Av, x0, gamma, ampl, delta)
            
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
        
        import dust_attenuation
        
        if not hasattr(self, 'N09'):
            self._init_N09()
            
        tau_V = self.get_tau(Av)
        
        #Av = np.polyval(self.coeffs['Av'], tau_V)
        x0 = np.polyval(self.coeffs['x0'], tau_V)
        gamma = np.polyval(self.coeffs['gamma'], tau_V)
        if self.include_bump:
            ampl = np.polyval(self.coeffs['ampl'], tau_V)*self.include_bump
        else:
            ampl = 0.
            
        slope = np.polyval(self.coeffs['slope'], tau_V)
        
        if not hasattr(x, 'unit'):
            xin = np.atleast_1d(x)*u.Angstrom
        else:
            xin = x
        
        if dust_attenuation.__version__ >= '0.0.dev131':                
            return self.N09.evaluate(xin, x0, gamma, ampl, slope, Av)
        else:
            return self.N09.evaluate(xin, Av, x0, gamma, ampl, slope)
        
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

BOUNDS = {}
BOUNDS['tage'] = [0.03, 12, 0.05]
BOUNDS['tau'] = [0.03, 2, 0.05]
BOUNDS['zred'] = [0.0, 13, 1.e-4]
BOUNDS['Av'] = [0.0, 15, 0.05]
BOUNDS['gas_logu'] = [-4, 0, 0.05]
BOUNDS['gas_logz'] = [-2, 0.3, 0.05]
BOUNDS['sigma_smooth'] = [0, 500, 0.05]

class ExtendedFsps(StellarPopulation):
    """
    Extended functionality for the `~fsps.StellarPopulation` object
    """
    
    lognorm_center = 0.
    lognorm_logwidth = 0.05
    is_lognorm_sfh = False
    lognorm_fburst = -30
    
    cosmology = WMAP9
    scale_lyman_series = 0.1
    scale_lines = OrderedDict()
    
    #_meta_bands = ['v']
    
    @property
    def fsps_ages(self):
        """
        (linear) ages of the FSPS SSP age grid, Gyr
        """
        if hasattr(self, '_fsps_ages'):
            return self._fsps_ages
        
        _ = self.get_spectrum()  
        fsps_ages = 10**(self.log_age-9)
        self._fsps_ages = fsps_ages
        return fsps_ages
        
    def set_lognormal_sfh(self, min_sigma=3, verbose=False, **kwargs):
        """
        Set lognormal tabular SFH
        """
        try:
            from grizli.utils_c.interp import interp_conserve_c as interp_func
        except:
            interp_func = utils.interp_conserve
        
        if 'lognorm_center' in kwargs:
            self.lognorm_center = kwargs['lognorm_center']
        
        if 'lognorm_logwidth' in kwargs:
            self.lognorm_logwidth = kwargs['lognorm_logwidth']
                
        if self.is_lognorm_sfh:
            self.params['sfh'] = 3
        
        if verbose:
            msg = 'lognormal SFH ({0}, {1}) [sfh3={2}]'
            print(msg.format(self.lognorm_center, self.lognorm_logwidth, 
                             self.is_lognorm_sfh))
                
        xages = np.logspace(np.log10(self.fsps_ages[0]), 
                            np.log10(self.fsps_ages[-1]), 2048)

        mu = self.lognorm_center#*np.log(10)
        # sfh = 1./t*exp(-(log(t)-mu)**2/2/sig**2)
        logn_sfh = 10**(-(np.log10(xages)-mu)**2/2/self.lognorm_logwidth**2)
        logn_sfh *= 1./xages 
        
        # Normalize
        logn_sfh *= 1.e-9/(self.lognorm_logwidth*np.sqrt(2*np.pi*np.log(10)))
        
        self.set_tabular_sfh(xages, logn_sfh) 
        self._lognorm_sfh = (xages, logn_sfh)
    
    def lognormal_integral(self, tage=0.1, **kwargs):
        """
        Integral of lognormal SFH up to t=tage
        """
        from scipy.special import erfc
        mu = self.lognorm_center*np.log(10)
        sig = self.lognorm_logwidth*np.sqrt(np.log(10))
        cdf = 0.5*erfc(-(np.log(tage)-mu)/sig/np.sqrt(2)) 
        return cdf
        
    def _set_extend_attrs(self, line_sigma=50, lya_sigma=200, **kwargs):
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
        
        for l in self.emline_names:
            self.scale_lines[l] = 1.
            
        # Precomputed arrays for WG00 reddening defined between 0.1..3 um
        self.wg00lim = (self.wavelengths > 1000) & (self.wavelengths < 3.e4)
        self.wg00red = (self.wavelengths > 1000)*1.
        
        self.exec_params = None
        self.narrow = None
        
    def narrow_emission_lines(self, tage=0.1, emwave=DEFAULT_LINES, line_sigma=100, oversample=5, clip_sigma=10, verbose=False, get_eqw=True, scale_lyman_series=None, scale_lines={}, force_recompute=False, use_sigma_smooth=True, lorentz=False, **kwargs):
        """
        Replace broad FSPS lines with specified line widths
    
        tage : age in Gyr of FSPS model
        FSPS sigma: line width in A in FSPS models
        emwave : (approx) wavelength of line to replace
        line_sigma : line width in km/s of new line
        oversample : factor by which to sample the Gaussian profiles
        clip_sigma : sigmas from line center to use for the line
        scale_lyman_series : scaling to apply to Lyman-series emission lines
        scale_lines : scaling to apply to other emission lines, by name
        
        Returns: `dict` with keys
            wave_full, flux_full, line_full = wave and flux with fine lines
            wave, flux_line, flux_clean = original model + removed lines
            ymin, ymax = range of new line useful for plotting
        
        """
        if not hasattr(self, 'emline_dlam'):
            self._set_extend_attrs(line_sigma=line_sigma, **kwargs)
        
        self.params['add_neb_emission'] = True
        
        if scale_lyman_series is None:
            scale_lyman_series = self.scale_lyman_series
        else:
            self.scale_lyman_series = scale_lyman_series
        
        if scale_lines is None:
            scale_lines = self.scale_lines
        else:
            for k in scale_lines:
                if k in self.scale_lines:
                    self.scale_lines[k] = scale_lines[k]
                else:
                    print(f'Line "{k}" not found in `self.scale_lines`')
        
        # Avoid recomputing if all parameters are the same (i.e., change Av)
        call_params = np.hstack([self.param_floats(params=None), emwave, 
                                 list(self.scale_lines.values()), 
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
        fsps_sigma = [np.sqrt((2*self.emline_dlam[i])**2 + 
             (self.params['sigma_smooth']/3.e5*self.emline_wavelengths[i])**2)
                              for i in line_ix]
        
        if line_sigma < 0:
            lines_sigma = [-line_sigma for ix in line_ix]
        elif (self.params['sigma_smooth'] > 0) & (use_sigma_smooth):
            lines_sigma = [self.params['sigma_smooth'] for ix in line_ix]
        else:
            lines_sigma = [self.emline_sigma[ix] for ix in line_ix]
            
        line_dlam = [sig/3.e5*lwave 
                     for sig, lwave in zip(lines_sigma, line_wave)]
    
        clean = line*1
        wlimits = [np.min(emwave), np.max(emwave)]
        wlimits = [2./3*wlimits[0], 4.3*wlimits[1]]
    
        wfine = utils.log_zgrid(wlimits, np.min(lines_sigma)/oversample/3.e5)
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
                    
            if lorentz:
                # astropy.modeling.functional_models.Lorentz1D.html
                # gamma is FWHM/2., integral is gamma*pi
                gam = 2.35482*line_dlam[i]/2.
                gline = gam**2/(gam**2 + (wfull-line_wave[i])**2) 
                norm = line_lum[i]/(gam*np.pi)
            else:
                # Gaussian
                gline = np.exp(-(wfull - line_wave[i])**2/2/line_dlam[i]**2)
                norm = line_lum[i]/np.sqrt(2*np.pi*line_dlam[i]**2)
                
            if self.emline_names[line_ix[i]].startswith('Ly'):
                norm *= scale_lyman_series
            
            if self.emline_names[line_ix[i]] in self.scale_lines:
                norm *= self.scale_lines[self.emline_names[line_ix[i]]]
                    
            gfull += gline*norm
            
            if get_eqw:
                clip = np.abs(wfull - line_wave[i]) < clip_sigma*line_dlam[i]
                eqw = np.trapz(gline[clip]*norm/cfull[clip], wfull[clip])
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
            
    def set_fir_template(self, arrays=None, file='templates/magdis/magdis_09.txt', verbose=True, unset=False):
        """
        Set the far-IR template for reprocessed dust emission
        """
        
        if unset:
            if verbose:
                print('Unset FIR template attributes')
            
            for attr in ['fir_template', 'fir_name', 'fir_arrays']:
                if hasattr(self, attr):
                    delattr(self, attr)
            
            return True
            
        if os.path.exists(file):
            if verbose:
                print(f'Set FIR dust template from {file}')
            _ = np.loadtxt(file, unpack=True)
            wave, flux = _[0], _[1]
            self.fir_name = file

        elif arrays is not None:
            if verbose:
                print(f'Set FIR dust template from input arrays')
            wave, flux = arrays
            self.fir_name = 'user-supplied'
        else:
            if verbose:
                print(f'Set FIR dust template from FSPS (DL07)')

            # Set with fsps
            self.params['dust1'] = 0
            self.params['dust2'] = 1.
            
            self.params['add_dust_emission'] = True
            wave, flux = self.get_spectrum(tage=1., peraa=True)
            self.params['add_dust_emission'] = False
            wave, flux_nodust = self.get_spectrum(tage=1., peraa=True)
            flux -= flux_nodust

            self.fir_name = 'fsps-dl07'
            
        fir_flux = np.interp(self.wavelengths, wave, flux, left=0, right=0)
        self.fir_template = fir_flux/np.trapz(fir_flux, self.wavelengths)
        self.fir_arrays = arrays
        return True
        
    def set_dust(self, Av=0., dust_obj_type='WG00x', wg00_kwargs=WG00_DEFAULTS):
        """
        Set `dust_obj` attribute
        
        dust_obj_type: 
        
            'WG00'  = `~dust_attenuation.radiative_transfer.WG00`
            'C00'   = `~dust_attenuation.averages.C00`
            'WG00x' = `ParameterizedWG00`
            'KC13'  = Kriek & Conroy (2013) with dust_index parameter
            
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
            elif dust_obj_type == 'C00':
                self.dust_obj = averages.C00(Av=Av)
            else:
                self.dust_obj = KC13(Av=Av)
                
            print('Init dust_obj: {0} {1}'.format(dust_obj_type, self.dust_obj.param_names))
            
        self.Av = Av
        
        if dust_obj_type == 'WG00':
            Avs = np.array([0.151, 0.298, 0.44 , 0.574, 0.825, 1.05 , 1.252, 1.428, 1.584, 1.726, 1.853, 1.961, 2.065, 2.154, 2.318, 2.454, 2.573, 2.686, 3.11 , 3.447, 3.758, 4.049, 4.317, 4.59 , 4.868, 5.148])
            taus = np.array([ 0.25,  0.5 ,  0.75,  1.  ,  1.5 ,  2.  ,  2.5 ,  3.  ,  3.5 , 4.  ,  4.5 ,  5.  ,  5.5 ,  6.  ,  7.  ,  8.  ,  9.  , 10.  , 15.  , 20.  , 25.  , 30.  , 35.  , 40.  , 45.  , 50.  ])
            tau_V = np.interp(Av, Avs, taus, left=0.25, right=50)
            self.dust_obj.tau_V = tau_V
            self.Av = self.dust_obj(5500*u.Angstrom)
        elif dust_obj_type == 'KC13':
            self.dust_obj.Av = Av
            self.dust_obj.delta = self.params['dust_index']
        else:
            self.dust_obj.Av = Av
    
    def get_full_spectrum(self, tage=1.0, Av=0., get_template=True, set_all_templates=False, **kwargs):
        """
        Get full spectrum with reprocessed emission lines and dust emission
        
        dust_fraction: Fraction of the SED that sees the specified Av
        
        """
        
        # Set the dust model
        if Av is None:
            Av = self.Av
            
        if 'dust_obj_type' in kwargs:
            self.set_dust(Av=Av, dust_obj_type=kwargs['dust_obj_type'])
        elif hasattr(self, 'dust_obj'):
            self.set_dust(Av=Av, dust_obj_type=self.dust_obj_type)
        else:
            self.set_dust(Av=Av, dust_obj_type='WG00x')
        
        # Lognormal SFH?
        if ('lognorm_center' in kwargs) | ('lognorm_logwidth' in kwargs):
            self.set_lognormal_sfh(**kwargs)
        
        if 'lognorm_fburst' in kwargs:
            self.lognorm_fburst = kwargs['lognorm_fburst']
        
        # Burst fraction for lognormal SFH
        if self.is_lognorm_sfh:
            if not hasattr(self, '_lognorm_sfh'):
                self.set_lognormal_sfh()
        
            t1 = self.lognormal_integral(tage)
            dt = (tage-self._lognorm_sfh[0])
            t100 = (dt <= 0.1) & (dt >= 0)
            
            sfhy = self._lognorm_sfh[1]*1.
            sfhy += t1*10**self.lognorm_fburst/100e6*t100
            self.set_tabular_sfh(self._lognorm_sfh[0], sfhy) 
                             
        # Set FSPS parameters
        for k in kwargs:
            if k in self.params.all_params:
                self.params[k] = kwargs[k]
        
        # Run the emission line function
        if tage is None:
            tage = self.params['tage']
            
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
            
            # To lines
            red_lines = (self.emline_wavelengths > 1000)*1.
            wlim = (self.emline_wavelengths > 1000) 
            wlim &= (self.emline_wavelengths < 3.e4)
            Alam  = self.dust_obj(self.emline_wavelengths[wlim]*u.Angstrom)
            red_lines[wlim] = 10**(-0.4*Alam)
            
        else:
            red = 10**(-0.4*self.dust_obj(wave*u.Angstrom))
            Alam = self.dust_obj(self.emline_wavelengths*u.Angstrom)
            red_lines = 10**(-0.4*Alam)
        
        # Apply dust to line luminosities
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
    
    def lognorm_avg_sfr(self, tage=None, dt=0.1):
        """
        Analytic average SFR for lognorm SFH
        """
        if tage is None:
            tage = self.params['tage']
            
        t1 = self.lognormal_integral(tage)
        t0 = self.lognormal_integral(np.maximum(tage-dt, 0))
        sfr_avg = (t1*(1+10**self.lognorm_fburst)-t0)/(dt*1.e9)
        return sfr_avg
        
    @property 
    def sfr100(self):
        """
        SFR averaged over maximum(tage, 100 Myr) from `sfr_avg`
        """
        if self.params['sfh'] == 0:
            sfr_avg = 0.
        elif self.params['sfh'] == 3:
            # Try to integrate SFH arrays if attribute set
            if self.is_lognorm_sfh:
                sfr_avg = self.lognorm_avg_sfr(tage=None, dt=0.1)
                
            elif hasattr(self, '_sfh_tab'):
                age_lb = self.params['tage'] - self._sfh_tab[0]
                age100 = (age_lb < 0.1) & (age_lb >= 0)
                if age100.sum() < 2:
                    sfr_avg = 0.
                else:
                    sfr_avg = np.trapz(self._sfh_tab[1][age100][::-1],
                                       age_lb[age100][::-1])/0.1
                
            else:
                sfr_avg = 0.
        else:
            sfr_avg = self.sfr_avg(dt=np.minimum(self.params['tage'], 0.1))
            
        return sfr_avg
    
    @property 
    def sfr10(self):
        """
        SFR averaged over last MAXIMUM(tage, 10 Myr) from `sfr_avg`
        """
        if self.params['sfh'] == 0:
            sfr_avg = 0.
        elif self.params['sfh'] == 3:
            # Try to integrate SFH arrays if attribute set
            if self.is_lognorm_sfh:
                sfr_avg = self.lognorm_avg_sfr(tage=None, dt=0.01)
                
            elif hasattr(self, '_sfh_tab'):
                age_lb = self.params['tage'] - self._sfh_tab[0]
                age10 = (age_lb < 0.01) & (age_lb >= 0)
                if age10.sum() < 2:
                    sfr_avg = 0.
                else:
                    sfr_avg = np.trapz(self._sfh_tab[1][age10][::-1],
                                       age_lb[age10][::-1])/0.01
                
            else:
                sfr_avg = 0.
        else:
            sfr_avg = self.sfr_avg(dt=np.minimum(self.params['tage'], 0.01))
            
        return sfr_avg
        
    @property 
    def meta(self):
        """
        Full metadata, including line properties
        """
        import fsps
        meta = self.param_dict
        
        if self._zcontinuous:
            meta['metallicity'] = 10**self.params['logzsol']*0.019
        else:
            meta['metallicity'] = self.zlegend[self.params['zmet']]
            
        for k in ['log_age','stellar_mass', 'formed_mass', 'log_lbol', 
                  'sfr', 'sfr100', 'dust_obj_type','Av','energy_absorbed', 
                  'fir_name', '_zcontinuous', 'scale_lyman_series',
                  'lognorm_center', 'lognorm_logwidth', 'is_lognorm_sfh', 
                  'lognorm_fburst']:
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
        
        # Band information
        if hasattr(self, '_meta_bands'):
            light_ages = self.light_age_band(self._meta_bands, flat=False)
            band_flux = self.get_mags(tage=self.params['tage'], zmet=None,
                                      bands=self._meta_bands, units='flam')
            
            band_waves = [fsps.get_filter(b).lambda_eff*u.Angstrom 
                          for b in self._meta_bands]
            band_lum = [f*w for f, w in zip(band_flux, band_waves)]
            
            for i, b in enumerate(self._meta_bands):
                meta['lwage_'+b] = light_ages[i]
                meta['lum_'+b] = band_lum[i].value
        try:
            meta['libraries'] = ';'.join([s.decode() for s in self.libraries])  
        except:
            try:
                meta['libraries'] = ';'.join([s in self.libraries])  
            except:
                meta['libraries'] = '[error]'
                
        return meta
        
    @property 
    def param_dict(self):
        """
        `dict` version of `self.params`
        """
        d = OrderedDict()
        for p in self.params.all_params:
            d[p] = self.params[p]
        
        return d
    
    def light_age_band(self, bands=['v'], flat=True):
        """
        Get light-weighted age of current model
        """
        self.params['compute_light_ages'] = True
        band_ages = self.get_mags(tage=self.params['tage'], zmet=None, 
                                  bands=bands)
        self.params['compute_light_ages'] = False
        if flat & (band_ages.shape == (1,)):
            return band_ages[0]
        else:
            return band_ages
        
    def pset(self, params):
        """
        Return a subset dictionary of `self.meta`
        """
        d = OrderedDict()
        for p in params:
            if p in self.meta:
                d[p] = self.meta[p]
            else:
                d[p] = np.nan
                
        return d
        
    def param_floats(self, params=None):
        """
        Return a list of parameter values.  If `params` is None, then use 
        full list in `self.params.all_params`. 
        """
        
        if params is None:
            params = self.params.all_params
            
        d = []
        for p in params:
            d.append(self.params[p]*1)
        
        return np.array(d)
    
    def parameter_bounds(self, params, limit_age=False):
        """
        Parameter bounds for `scipy.optimize.least_squares`
        
        """
        blo = []
        bhi = []
        steps = []
        for p in params:
            if p in BOUNDS:
                blo.append(BOUNDS[p][0])
                bhi.append(BOUNDS[p][1])
                steps.append(BOUNDS[p][2])
            else:
                blo.append(-np.inf)
                bhi.append(np.inf)
                steps.append(0.05)
                
        return (blo, bhi), steps
            
    def fit_spec(wave_obs, flux_obs, err_obs, mask=None, plist=['tage', 'Av', 'gas_logu', 'sigma_smooth'], func_kwargs={'lorentz':False}, verbose=True, bspl_kwargs=None, lsq_kwargs={'method':'trf', 'max_nfev':200, 'loss':'huber', 'x_scale':1.0, 'verbose':True}, show=False):
        """
        Fit models to observed spectrum
        """
        from scipy.optimize import least_squares
        import grizli.utils
        
        sys_err = 0.015
        
        if wave_obs is None:
            # mpdaf muse spectrum
            spec = _[0].spectra['MUSE_TOT_SKYSUB']
            wave_obs = spec.wave.coord()
            flux_obs = spec.data.filled(fill_value=np.nan)
            err_obs = np.sqrt(spec.var.filled(fill_value=np.nan))
            err_obs = np.sqrt(err_obs**2+(sys_err*flux_obs)**2) 
            
            mask = np.isfinite(flux_obs+err_obs) & (err_obs > 0) 
            omask = mask
            
            #mask = omask & (wave_obs/(1+0.0342) > 6520) & (wave_obs/(1+0.0342) < 6780)
            mask = omask & (wave_obs/(1+0.0342) > 4800) & (wave_obs/(1+0.0342) < 5050)
            
        theta0 = np.array([self.meta[p] for p in plist])
        
        if bspl_kwargs is not None:
            bspl = grizli.utils.bspline_templates(wave_obs, get_matrix=True, **bspl_kwargs)
        else:
            bspl = None
            
        kwargs = func_kwargs.copy()
        for i, p in enumerate(plist):
            kwargs[p] = theta0[i]
        
        # Model test
        margs = (self, plist, wave_obs, flux_obs, err_obs, mask, bspl, kwargs, 'model')        
        flux_model, Anorm, chi2_init = objfun_fitspec(theta0, *margs)
        
        if show:                
            mask &= np.isfinite(flux_model+flux_obs+err_obs) & (err_obs > 0) 

            plt.close('all')
            
            fig = plt.figure(figsize=(12, 6))
            plt.errorbar(wave_obs[mask], flux_obs[mask], err_obs[mask], color='k', alpha=0.5, linestyle='None', marker='.')
            plt.plot(wave_obs, flux_model, color='pink', linewidth=2, alpha=0.8)
            
        bounds, steps = self.parameter_bounds(plist)
        #lsq_kwargs['diff_step'] = np.array(steps)/2.
        #lsq_kwargs['diff_step'] = 0.05
        lsq_kwargs['diff_step'] = steps
        lmargs = (self, plist, wave_obs, flux_obs, err_obs, mask, bspl, kwargs, 'least_squares verbose')        
        _res = least_squares(objfun_fitspec, theta0, bounds=bounds, args=lmargs, **lsq_kwargs)
        
        fit_model, Anorm, chi2_fit = objfun_fitspec(_res.x, *margs)
        
        if True:
            plt.plot(wave_obs, fit_model, color='r', linewidth=2, alpha=0.8, zorder=100)
            plt.xlim(wave_obs[mask].min(), wave_obs[mask].max())
        
    @staticmethod
    def objfun_fitspec(theta, self, plist, wave_obs, flux_obs, err_obs, mask, bspl, kwargs, ret_type):
        """
        Objective function for fitting spectra
        """
        try:
            from grizli.utils_c.interp import interp_conserve_c as interp_func
        except:
            interp_func = utils.interp_conserve
            
        for i, p in enumerate(plist):
            kwargs[p] = theta[i]
                    
        templ = self.get_full_spectrum(**kwargs)
        flux_model = templ.resample(wave_obs, z=self.params['zred'],
                                   in_place=False,
                                   return_array=True, interp_func=interp_func)
        
        if mask is None:
            mask = np.isfinite(flux_model+flux_obs+err_obs) & (err_obs > 0)
        
        if bspl is not None:
            _A = (bspl.T*flux_model)
            _yx = (flux_obs / err_obs)[mask]
            _c = np.linalg.lstsq((_A/err_obs).T[mask,:], _yx, rcond=-1)
            Anorm = np.mean(bspl.dot(_c[0]))
            flux_model = _A.T.dot(_c[0])
            
        else:
            lsq_num = (flux_obs*flux_model/err_obs**2)[mask].sum()
            lsq_den = (flux_model**2/err_obs**2)[mask].sum()
            Anorm = lsq_num/lsq_den
            flux_model *= Anorm
        
        chi = ((flux_model - flux_obs)/err_obs)[mask]
        chi2 = (chi**2).sum()
        
        if 'verbose' in ret_type:
            print('{0} {1:.4f}'.format(theta, chi2))
            
        if 'model' in ret_type:
            return flux_model, Anorm, chi2
        elif 'least_squares' in ret_type:
            return chi
        elif 'logpdf' in ret_type:
            return -chi2/2.
        else:
            return chi2
    
    def line_to_obsframe(self, zred=None, cosmology=None, verbose=False, unit=u.erg/u.second/u.cm**2):
        """
        Scale factor to convert line luminosities to observed frame
        """
        from astropy.constants import L_sun
        
        if zred == None:
            zred = self.params['zred']
            if verbose:
                msg = 'continuum_to_obsframe: Use params[zred] = {0:.3f}'
                print(msg.format(zred))
                    
        if cosmology is None:
            cosmology = self.cosmology
        else:
            self.cosmology = cosmology
        
        if zred <= 0:
            dL = 1*u.cm
        else:
            dL = cosmology.luminosity_distance(zred).to(u.cm)

        to_cgs = (1*L_sun/(4*np.pi*dL**2)).to(unit)
        return to_cgs.value
        
    def continuum_to_obsframe(self, zred=None, cosmology=None, unit=u.microJansky, verbose=False):
        """
        Compute a normalization factor to scale input FSPS model flux density 
        units of (L_sun / Hz) or (L_sun / \AA) to observed-frame `unit`.
        """
        from astropy.constants import L_sun

        if zred == None:
            zred = self.params['zred']
            if verbose:
                msg = 'continuum_to_obsframe: Use params[zred] = {0:.3f}'
                print(msg.format(zred))
                    
        if cosmology is None:
            cosmology = self.cosmology
        else:
            self.cosmology = cosmology
        
        if zred <= 0:
            dL = 1*u.cm
        else:
            dL = cosmology.luminosity_distance(zred).to(u.cm)
        
        # FSPS L_sun / Hz to observed-frame
        try:
            # Unit is like f-lambda
            _x = (1*unit).to(u.erg/u.second/u.cm**2/u.Angstrom) 
            is_flam = True
            obs_unit = (1*L_sun/u.Angstrom/(4*np.pi*dL**2)).to(unit)/(1+zred)
        except:
            # Unit is like f-nu
            is_flam = False
            obs_unit = (1*L_sun/u.Hz/(4*np.pi*dL**2)).to(unit)*(1+zred)
        
        return obs_unit.value
        
    def fit_phot(self, phot_dict, filters=None, flux_unit=u.microJansky, plist=['tage', 'Av', 'gas_logu', 'sigma_smooth'], func_kwargs={'lorentz':False}, verbose=True, lsq_kwargs={'method':'trf', 'max_nfev':200, 'loss':'huber', 'x_scale':1.0, 'verbose':True}, show=False, TEF=None, photoz_obj=None):
        """
        Fit models to observed spectrum
        """
        from scipy.optimize import least_squares
        import grizli.utils
        
        sys_err = 0.02
        
        if False:
            import eazy.sps
            
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
            ax.plot(templ.wave*(1+_phot['z'])/1.e4, templ.flux_fnu*_phot['scale'], label=label, color='r', alpha=0.5)
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
            
            
        flux = phot_dict['fobs']
        err = phot_dict['efobs']
        if 'flux_unit' in phot_dict:
            flux_unit = phot_dict['flux_unit']
            
        x0 = np.array([self.meta[p] for p in plist])
        
        # Are input fluxes f-lambda or f-nu?
        try:
            _x = (1*flux_unit).to(u.erg/u.second/u.cm**2/u.Angstrom) 
            is_flam = True
        except:
            is_flam = False
         
        # Initialize keywords   
        kwargs = func_kwargs.copy()
        for i_p, p in enumerate(plist):
            kwargs[p] = x0[i_p]
        
        # Initial model
        margs = (self, plist, flux, err, is_flam, filters, TEF, kwargs, 'model')        
        flux_model, Anorm, chi2_init, templ = self.objfun_fitphot(x0, *margs)
        
        if show:                
            
            if hasattr(show, 'plot'):
                ax = show
            else:
                plt.close('all')
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                
            mask = err > 0
            pivot = np.array([f.pivot for f in filters])
            so = np.argsort(pivot)
            
            ax.errorbar(pivot[mask]/1.e4, flux[mask], err[mask],
                        color='k', alpha=0.5, linestyle='None', marker='.')
            ax.scatter(pivot[so]/1.e4, flux_model[so], color='pink', 
                       alpha=0.8, zorder=100)
        
        # Parameter bounds    
        bounds, steps = self.parameter_bounds(plist)
        lsq_kwargs['diff_step'] = steps
        
        # Run the optimization
        lmargs = (self, plist, flux, err, is_flam, filters, TEF, kwargs, 'least_squares verbose')        
        _res = least_squares(self.objfun_fitphot, x0, bounds=bounds,
                             args=lmargs, **lsq_kwargs)
        
        _out = self.objfun_fitphot(_res.x, *margs)
        
        _fit = {}
        _fit['fmodel'] = _out[0]
        _fit['scale'] = _out[1]
        _fit['chi2'] = _out[2]
        _fit['templ'] = _out[3]
        _fit['plist'] = plist
        _fit['theta'] = _res.x
        _fit['res'] = _res
        
        # Stellar mass
        #fit_model, Anorm, chi2_fit, templ = _phot
        
        # Parameter scaling to observed frame.  
        # e.g., stellar mass = self.stellar_mass * scale / to_obsframe
        z = self.params['zred']
        _obsfr = self.continuum_to_obsframe(zred=z, unit=flux_unit)
        _fit['to_obsframe'] = _obsfr
        
        scl = _fit['scale']/_fit['to_obsframe']
        _fit['log_mass'] = np.log10(self.stellar_mass*scl)
        _fit['sfr'] = self.sfr*scl
        _fit['sfr10'] = self.sfr10*scl
        _fit['sfr100'] = self.sfr100*scl
        age_bands = ['i1500','v']
        ages = self.light_age_band(bands=age_bands)
        for i, b in enumerate(age_bands):
            _fit['age_'+b] = ages[i]
            
        if show:
            ax.scatter(pivot[so]/1.e4, _fit['fmodel'][so], color='r', 
                      alpha=0.8, zorder=101)
            
            if is_flam:
                ax.plot(templ.wave*(1+z)/1.e4, templ.flux*Anorm, 
                    color='r', alpha=0.3, zorder=10000)
            else:
                ax.plot(templ.wave*(1+z)/1.e4, templ.flux_fnu*Anorm, 
                    color='r', alpha=0.3, zorder=10000)
        
        return _fit
        
    @staticmethod
    def objfun_fitphot(theta, self, plist, flux_fnu, err_fnu, is_flam, filters, TEF, kwargs, ret_type):
        """
        Objective function for fitting spectra
        """
        try:
            from grizli.utils_c.interp import interp_conserve_c as interp_func
        except:
            interp_func = utils.interp_conserve
            
        for i, p in enumerate(plist):
            kwargs[p] = theta[i]
                    
        templ = self.get_full_spectrum(**kwargs)
        model_fnu = templ.integrate_filter_list(filters,
                                     z=self.params['zred'], flam=is_flam, 
                                     include_igm=True)
        
        mask = (err_fnu > 0)
        full_var = err_fnu**2
        
        if TEF is not None:
            tefz = TEF(self.params['zred'])
            full_var += (flux_fnu*tefz)**2
            
        lsq_num = (flux_fnu*model_fnu/full_var)[mask].sum()
        lsq_den = (model_fnu**2/full_var)[mask].sum()
        Anorm = lsq_num/lsq_den
        model_fnu *= Anorm
        
        chi = ((model_fnu - flux_fnu)/np.sqrt(full_var))[mask]
        chi2 = (chi**2).sum()
        
        if 'verbose' in ret_type:
            print('{0} {1:.4f}'.format(theta, (chi**2).sum()))
            
        if 'model' in ret_type:
            return model_fnu, Anorm, chi2, templ
        elif 'least_squares' in ret_type:
            return chi
        elif 'logpdf' in ret_type:
            return -chi2/2
        else:
            return chi2
            
        