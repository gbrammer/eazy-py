import pytest
import os

import numpy as np
np.random.seed(0)
np.seterr(all='ignore')

from .. import utils
from .. import templates
from .. import filters
from .. import photoz
from .. import filters

from . import test_filters
from . import test_templates

ez = None

# Single redshift for testing
z_spec = 1.0

# Additional catalog objects with random noise
NRND = 16

def make_fake_catalog(SN=20):
    """
    Make a fake photometric catalog
    """
    
    data_path = test_filters.test_data_path()
    os.chdir(data_path)
    
    #### Generate data
    res = filters.FilterFile('filters/FILTER.RES.latest')
    templ = test_templates.test_read_template_ascii()
    
    ### WFC3 SED + K + IRAC
    f_numbers = [209, 211, 214, 217, 202, 203, 205, 269, 18, 19]
    f_list = [res[f] for f in f_numbers]
        
    ### Photometry from a single template
    fnu = templ.integrate_filter(f_list, z=z_spec)

    ### Norm to F160W
    i_f160 = -4
    flux_f160 = 1. # microJy
    fnu *= flux_f160/fnu[i_f160]
    
    ### Add noise
    #SN = 20
    efnu_f160w = (fnu/SN)[i_f160]
    lc = np.array([f.pivot for f in f_list])
    efnu = efnu_f160w*(lc/lc[i_f160])**2
    
    ### Make table
    tab = photoz.Table()
    tab['id'] = np.arange(NRND+1, dtype=int)+1
    tab['z_spec'] = z_spec
    tab['ra'] = 150.1
    tab['dec'] = 2.5
    
    ### Simpler filter names for catalog
    f_names = []
    for f in f_list:
        f_name = f.name.split(' ')[0].split('/')[-1].split('.dat')[0]
        f_name = f_name.replace('irac_tr','ch')
        f_name = f_name.replace('hawki_k','k').split('_')[0]
        f_names.append(f_name)
    
    ### Translate file
    translate_file = 'zphot.translate.test'
    
    with open(translate_file,'w') as fp:
        for i, f in enumerate(f_names):
            tab[f'f_{f}'] = fnu[i] + np.append(0, np.random.normal(size=NRND)*efnu[i])
            tab[f'e_{f}'] = efnu[i]

            fp.write(f'f_{f} F{f_numbers[i]}\n')
            fp.write(f'e_{f} E{f_numbers[i]}\n')

    ### ASCII catalog
    cat_file = 'eazy_test.cat'
    tab.write(cat_file, overwrite=True, format='ascii.commented_header')
    tab.write(cat_file+'.fits', overwrite=True, format='fits')
    return tab, cat_file, translate_file


def test_full_photoz():
    """
    End-to-end test
    """
    global ez
        
    tab, cat_file, translate_file = make_fake_catalog(SN=20)
    
    data_path = test_filters.test_data_path()
    os.chdir(data_path)
        
    ### Parameters
    params = {}
    params['CATALOG_FILE'] = cat_file
    params['MAIN_OUTPUT_FILE'] = 'eazy_test'

    # Galactic extinction
    params['MW_EBV'] = 0.0

    params['Z_STEP'] = 0.01
    params['Z_MIN'] = z_spec - 0.5*(1+z_spec)
    params['Z_MAX'] = z_spec + 0.5*(1+z_spec)

    params['SYS_ERR'] = 0.02

    params['PRIOR_ABZP'] = 23.9 # uJy 
    params['PRIOR_FILTER'] = 205 # f160W
    params['PRIOR_FILE'] = 'templates/prior_F160W_TAO.dat'

    params['FILTERS_RES'] = 'filters/FILTER.RES.latest'
    params['TEMPLATES_FILE'] = 'templates/fsps_full/fsps_QSF_12_v3.param'
    params['VERBOSITY'] = 1
    params['FIX_ZSPEC'] = False
    
    ### Initialize object
    ez = photoz.PhotoZ(param_file=None, translate_file=translate_file, 
                         zeropoint_file=None, params=params, load_prior=True, 
                         load_products=False)
    
    # FITS catalog
    params['CATALOG_FILE'] = cat_file+'.fits'
    ez = photoz.PhotoZ(param_file=None, translate_file=translate_file, 
                         zeropoint_file=None, params=params, load_prior=True, 
                         load_products=False)


def test_photoz_methods():
    """
    Test methods on `~eazy.photoz.PhotoZ` object.
    """
    global ez
    
    ### Catalog subset
    ez.fit_parallel(idx=np.where(ez.cat['id'] < 2)[0], fitter='nnls')
    
    ### Full catalog, fitting methods
    ez.fit_parallel(fitter='lstsq')
    ez.fit_parallel(fitter='bounded')
    ez.fit_parallel(fitter='nnls')
        
    ###### Methods
    
    # Specified zbest
    ez.fit_at_zbest(zbest=np.full(NRND+1, z_spec), 
              prior=False, beta_prior=False, 
              get_err=False, clip_wavelength=1100, fitter='nnls', 
              selection=None, n_proc=0, par_skip=10000)
    
    # default zbest
    ez.fit_at_zbest(zbest=None, prior=False, beta_prior=False, 
            get_err=False, clip_wavelength=1100, fitter='nnls', 
            selection=None, n_proc=0, par_skip=10000)
    
    # priors
    for prior in [True, False]:
        for beta_prior in [True, False]:
            ez.fit_at_zbest(zbest=None, prior=prior, beta_prior=beta_prior, 
                      get_err=False, clip_wavelength=1100, fitter='nnls', 
                      selection=None, n_proc=0, par_skip=10000)
    
    # Peak-finder
    peaks, numpeaks = ez.find_peaks()
    assert(np.allclose(numpeaks, 1))
    assert(np.allclose(ez.zgrid[peaks[0][0]], z_spec, atol=0.01*(1+z_spec)))
    
    return ez


def test_sps_parameters():
    """
    Derived parameters
    """
    global ez
    
    ### Run all photo-zs
    ez.fit_parallel(fitter='nnls')
        
    ### SPS parameters
    # Full RF-colors with filter weighting
    zout, hdu = ez.standard_output(zbest=None, rf_pad_width=0.5, rf_max_err=2, 
                                   prior=True, beta_prior=True, simple=False)

    # "Simple" best-fit template RF colors
    zout, hdu = ez.standard_output(zbest=None, rf_pad_width=0.5, rf_max_err=2, 
                                   prior=True, beta_prior=True, simple=True)
    
    assert(np.allclose(zout['z_phot'][0], z_spec, atol=0.1*(1+z_spec)))

    coeffs_norm = ez.coeffs_best[0,:]/ez.coeffs_best[0,:].max()     
    assert(np.argmax(coeffs_norm) == 0)
    assert(np.sum(coeffs_norm) < 1.1)
    
    ### All zout data
    # zdict = {}
    # for k in zout.colnames:
    #     zdict[k] = zout[k][0]
    
    zdict = {'nusefilt': 10,
    'z_ml': 0.99616235,
    'z_ml_chi2': 0.013447836,
    'z_ml_risk': 0.0105553605,
    'lc_min': 3353.6304006459895,
    'lc_max': 45020.33785230743,
    'z_phot': 0.99616235,
    'z_phot_chi2': 0.013447836,
    'z_phot_risk': 0.0105553605,
    'z_min_risk': 0.9937155,
    'min_risk': 0.010250151,
    'z_raw_chi2': 0.9937155,
    'raw_chi2': 0.035614725,
    'z025': 0.92501247,
    'z160': 0.9604295,
    'z500': 0.99208033,
    'z840': 1.0187114,
    'z975': 1.0420052,
    'restU': 0.41460526,
    'restU_err': 0.01217702,
    'restB': 0.8223915,
    'restB_err': 0.027577162,
    'restV': 0.92202765,
    'restV_err': 0.017819434,
    'restJ': 1.024555,
    'restJ_err': 0.05461645,
    'dL': 6575.8372348364455,
    'mass': 1338132577.7487125,
    'sfr': 0.026515421690098212,
    'Lv': 3418389791.239653,
    'LIR': 438179193.31513166,
    'MLv': 0.39145113912344354,
    'Av': 0.06295947926487588,
    'rest270': 0.11133574,
    'rest270_err': 0.007641867,
    'rest274': 0.23238972,
    'rest274_err': 0.008679345,
    'rest120': 0.12516989,
    'rest120_err': 0.005393833,
    'rest121': 0.1816069,
    'rest121_err': 0.00364957,
    'rest156': 0.3724664,
    'rest156_err': 0.014633045,
    'rest157': 0.86651146,
    'rest157_err': 0.018754214,
    'rest158': 0.94490474,
    'rest158_err': 0.027536243,
    'rest159': 0.997915,
    'rest159_err': 0.023829281,
    'rest160': 1.0238949,
    'rest160_err': 0.0475851,
    'rest161': 1.024555,
    'rest161_err': 0.05461645,
    'rest162': 1.010895,
    'rest162_err': 0.06887752,
    'rest163': 0.7563232,
    'rest163_err': 0.06583378,
    'DISTMOD': 43.33921454218198}
    
    for k in zdict:
        if '_err' in k:
            assert(np.allclose(zout[k][0], zdict[k], rtol=0.5))
        else:
            assert(np.allclose(zout[k][0], zdict[k], rtol=0.1))

    # confirm that zout['z_phot'] == zout['z_ml']
    assert( np.all(zout['z_ml'] == zout['z_phot']) )
        
    ### user-specified zbest
    zuser = np.full(NRND+1, z_spec)
    z2, _ = ez.standard_output(zbest=zuser, rf_pad_width=0.5, rf_max_err=2, 
                                     prior=True, beta_prior=True, simple=True)
    
    # confirm that z2 has 'z_ml' and 'z_phot' columns and they're different 
    assert( np.all(z2['z_ml'] != z2['z_phot']) )

    # confirm that z2['z_ml'] == zout['z_phot']
    assert( np.all(z2['z_ml'] == zout['z_phot']) )
    
    # zphot is now the user-specified redshift
    assert(np.allclose(z2['z_phot'], zuser, rtol=1.e-2))
    
    # SPS parameters are different, as calculated for zuser
    assert( np.all(z2['mass'] != zout['mass']) )
    assert( np.all(z2['sfr'] != zout['sfr']) )


def test_fit_stars():
    """
    Fit phoenix star library for Star/Galaxy separation
    """
    global ez
    ez.fit_phoenix_stars()
    assert(np.allclose(ez.star_chi2[0,0], 3191.3662))


def test_photoz_figures():
    """
    Figures generated with PhotoZ object
    """
    import matplotlib.pyplot as plt
    
    global ez
    
    ### SED figure
    fig, data = ez.show_fit(id=0, id_is_idx=True, show_fnu=False)
    fig.savefig('eazy_test.sed.png', dpi=72)
    
    assert(isinstance(fig, plt.Figure))
    assert(isinstance(data, dict))
    
    fig = ez.show_fit(id=1, show_fnu=False)
    fig = ez.show_fit(id=1, show_fnu=True)
    fig = ez.show_fit(id=1, show_fnu=2)

    fig = ez.show_fit(id=1, show_fnu=False, zshow=z_spec)

    fig = ez.show_fit(id=1, show_components=True)
    
    fig = ez.zphot_zspec()
    fig = ez.zphot_zspec(zmin=0, zmax=2)
    fig.savefig('eazy_test.zphot_zspec.png', dpi=72)
    
    plt.close('all')
    