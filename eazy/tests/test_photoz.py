import pytest
import os

import numpy as np

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
    efnu = fnu/SN
    
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
    np.random.seed(0)

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
    params['Z_MIN'] = 0.5
    params['Z_MAX'] = 2.1

    params['SYS_ERR'] = 0.02

    params['PRIOR_ABZP'] = 23.9 # uJy 
    params['PRIOR_FILTER'] = 205 # f160W
    params['PRIOR_FILE'] = 'templates/prior_F160W_TAO.dat'

    params['FILTERS_RES'] = 'filters/FILTER.RES.latest'
    params['TEMPLATES_FILE'] = 'templates/fsps_full/fsps_QSF_12_v3.param'
    params['VERBOSITY'] = 1
    params['FIX_ZSPEC'] = False
    
    ### Initialize object
    self = photoz.PhotoZ(param_file=None, translate_file=translate_file, 
                         zeropoint_file=None, params=params, load_prior=True, 
                         load_products=False)
    
    # FITS catalog
    params['CATALOG_FILE'] = cat_file+'.fits'
    ez = photoz.PhotoZ(param_file=None, translate_file=translate_file, 
                         zeropoint_file=None, params=params, load_prior=True, 
                         load_products=False)


def test_photoz_methods():
    global ez
    
    ### Catalog subset
    ez.fit_parallel(idx=np.where(ez.cat['id'] < 2)[0], fitter='nnls')
    
    ### Full catalog, fitting methods
    for fitter in ['lstsq','bounded','nnls']:
        ez.fit_parallel(fitter=fitter)
        
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
    
    return ez


def test_sps_parameters():
    """
    Derived parameters
    """
    global ez
    
    ### Run all photo-zs
    ez.fit_parallel(fitter='nnls')
        
    ### SPS parameters
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
    
    zdict = {
    'nusefilt': 10,
    'lc_min': 3353.6304006459895,
    'lc_max': 45020.33785230743,
    'z_phot': 0.99673086,
    'z_phot_chi2': 0.0034560256,
    'z_phot_risk': 0.035717614,
    'z_min_risk': 0.9649467,
    'min_risk': 0.030646123,
    'z_raw_chi2': 1.0046412,
    'raw_chi2': 0.026701305,
    'z025': 0.8389708,
    'z160': 0.92368615,
    'z500': 0.9808423,
    'z840': 1.0210177,
    'z975': 1.0537113,
    'restU': 0.41497922,
    'restU_err': 0.017939776,
    'restB': 0.8218043,
    'restB_err': 0.03606099,
    'restV': 0.9209713,
    'restV_err': 0.035820752,
    'restJ': 1.0253503,
    'restJ_err': 0.023321927,
    'dL': 6580.476033271505,
    'mass': 1338116524.1430814,
    'sfr': 0.016536130604908845,
    'Lv': 3424873079.787285,
    'LIR': 390855097.57019836,
    'MLv': 0.39070543432406274,
    'Av': 0.06089298262931584,
    'rest270': 0.11237647,
    'rest270_err': 0.011739388,
    'rest274': 0.23368931,
    'rest274_err': 0.012826629,
    'rest120': 0.12594292,
    'rest120_err': 0.008697912,
    'rest121': 0.18348014,
    'rest121_err': 0.005728759,
    'rest156': 0.37412244,
    'rest156_err': 0.0184125,
    'rest157': 0.8652829,
    'rest157_err': 0.031125754,
    'rest158': 0.9475169,
    'rest158_err': 0.018318474,
    'rest159': 0.9954984,
    'rest159_err': 0.030312747,
    'rest160': 1.0260057,
    'rest160_err': 0.021030843,
    'rest161': 1.0253503,
    'rest161_err': 0.023321927,
    'rest162': 1.012053,
    'rest162_err': 0.025552988,
    'rest163': 0.7521126,
    'rest163_err': 0.02862215,
    'DISTMOD': 43.340373559176065}
    
    for k in zdict:
        assert(np.allclose(zout[k][0], zdict[k], rtol=0.1))

    ### user-specified zbest
    z2, _ = ez.standard_output(zbest=np.full(NRND+1, z_spec),
                                   rf_pad_width=0.5, rf_max_err=2, 
                                     prior=True, beta_prior=True, simple=True)
    
    # confirm that z2 has 'z_ml' and 'z_phot' columns and they're different 
    assert( np.any(z2['z_ml'] != z2['z_phot']) )

    # confirm that z2['z_ml'] == zout['z_phot']
    assert( np.all(z2['z_ml'] == zout['z_phot']) )
    
    # zphot is the user-specified redshift
    assert(np.allclose(z2['z_phot'], z_spec, rtol=1.e-2))

    # confirm that zout['z_phot'] == zout['z_ml']
    assert( np.all(zout['z_ml'] == zout['z_phot']) )

    ### ToDo: Check that sps parameters are different...
    
    ### ToDo: add tests for run_find_peaks, simple=False
    
def test_fit_stars():
    """
    Fit phoenix star library for Star/Galaxy separation
    """
    global ez
    ez.fit_phoenix_stars()
    assert(np.allclose(ez.star_chi2[0,0], 1930.887))


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

    fig = ez.show_fit(id=1, show_components=True)
    
    fig = ez.zphot_zspec()
    fig = ez.zphot_zspec(zmin=0, zmax=2)
    fig.savefig('eazy_test.zphot_zspec.png', dpi=72)
    
    plt.close('all')
    