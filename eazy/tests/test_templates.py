import pytest
import os

import numpy as np

from .. import utils
from .. import templates
from .. import filters

def test_data_path():
    """
    Data path
    """
    path = os.path.join(os.path.dirname(__file__), '../data/')
    assert(os.path.exists(path))
    return path


def test_templates_path():
    """
    Does ``templates`` path exist?
    """
    path = test_data_path()
    assert(os.path.exists(os.path.join(path, 'templates')))
    return os.path.join(path, 'templates')


def test_read_template_ascii():
    """
    Test interpolation function
    """
    path = test_templates_path()
    ascii_file = os.path.join(path, 'fsps_full/fsps_QSF_12_v3_001.dat')
    
    templ = templates.Template(file=ascii_file)
    assert(templ.name == 'fsps_QSF_12_v3_001.dat')
    assert(np.allclose(templ.flux.shape, [1,5994]))
    return templ


def test_read_template_fits():
    """
    Read template FITS file
    """
    path = test_templates_path()
    fits_file = os.path.join(path, 
                              'spline_templates_v2/spline_age0.01_av0.0.fits')
    
    templ = templates.Template(file=fits_file)
    assert(np.allclose(templ.flux.shape, [templ.NZ, 12603]))
    assert(templ.name == 'spline_age0.01_av0.0.fits')
    return templ


def test_gaussian_templates():
    """
    Test templates.gaussian_templates
    """
    wave = np.arange(5000., 6000.)
    centers = np.arange(5100.,5901.,100)

    width = 10
    widths = centers*0+width
    
    NW = len(wave)
    NG = len(centers)
    norm = np.sqrt(2*np.pi*width**2)
    
    n0 = templates.gaussian_templates(wave, centers=centers, widths=widths, 
                                       norm=False)
    
    assert(np.allclose(n0.shape, (NW, NG)))  
    assert(np.allclose(n0.max(), 1., rtol=1.e-4))    
    assert(np.allclose(n0.sum(), norm*NG, rtol=1.e-4))
    
    # Normalized
    n1 = templates.gaussian_templates(wave, centers=centers, widths=widths, 
                                       norm=True)
    
    assert(np.allclose(n1.shape, (NW, NG)))  
    assert(np.allclose(n1.max(), 1./norm, rtol=1.e-4))    
    assert(np.allclose(n1.sum(), NG, rtol=1.e-4))
                                     
    return True


def test_bspline_templates():
    """
    templates.bspline_templates
    """
    wave = np.arange(5000., 6000.)
    NW = len(wave)
    
    df=6
    
    for df in [6, 8, 12]:
        for log in [True, False]:
            spl = templates.bspline_templates(wave, degree=3, df=df, 
                                              get_matrix=True, log=log, 
                                              clip=0.0001, minmax=None)
    
            assert(np.allclose(spl.shape, (NW, df)))  
            assert(np.allclose(spl.sum(axis=1), 1., rtol=1.e-4))
    
    spt = templates.bspline_templates(wave, degree=3, df=df, 
                                     get_matrix=False, log=log, 
                                     clip=0.0001, minmax=None)                
    
    assert(len(spt) == df)
    keys = list(spt.keys())
    for i, k in enumerate(keys):
        templ = spt[k]
        assert(np.allclose(templ.wave, wave))  
        assert(np.allclose(spl[:,i], np.squeeze(templ.flux)))


def test_redshift_dependent():
    """
    Redshift-dependent templates
    """
    wave = np.arange(5000., 6000.)

    # No dependence
    flux = np.ones(len(wave))
    templ = templates.Template(arrays=(wave, flux), redshifts=[0])
    assert(templ.zindex(-0.1, redshift_type='nearest') == 0)
    assert(templ.zindex(0.3, redshift_type='nearest') == 0)

    
    assert(templ.zindex(-0.1, redshift_type='floor') == 0)
    assert(templ.zindex(0.3, redshift_type='floor') == 0)
    
    assert(np.allclose(templ.zindex(-0.1, redshift_type='interp'), (0, 1.0)))
    assert(np.allclose(templ.zindex(0.1, redshift_type='interp'), (0, 1.0)))
    
    # Redshift-dependent
    flux = np.ones((2, len(wave)))
    flux[1,:] = 2
    
    templ = templates.Template(arrays=(wave, flux), redshifts=[0,1])
    
    assert(templ.zindex(-0.1, redshift_type='nearest') == 0)
    assert(templ.zindex(0.3, redshift_type='nearest') == 0)
    assert(templ.zindex(0.6, redshift_type='nearest') == 1)
    assert(templ.zindex(2.6, redshift_type='nearest') == 1)
    
    assert(templ.zindex(-0.1, redshift_type='floor') == 0)
    assert(templ.zindex(0.3, redshift_type='floor') == 0)
    assert(templ.zindex(0.6, redshift_type='floor') == 0)
    assert(templ.zindex(2.6, redshift_type='floor') == 1)
    
    assert(np.allclose(templ.zindex(-0.1, redshift_type='interp'), (0, 1.0)))
    assert(np.allclose(templ.zindex(0.1, redshift_type='interp'), (0, 0.9)))
    assert(np.allclose(templ.zindex(0.9, redshift_type='interp'), (0, 0.1)))
    assert(np.allclose(templ.zindex(1.1, redshift_type='interp'), (1, 1.0)))
    
    assert(np.allclose(templ.flux_flam(iz=0, redshift_type='nearest'), 1.))
    assert(np.allclose(templ.flux_flam(iz=1, redshift_type='nearest'), 2.))

    assert(np.allclose(templ.flux_flam(z=-1., redshift_type='nearest'), 1.))
    assert(np.allclose(templ.flux_flam(z=0.0, redshift_type='nearest'), 1.))
    assert(np.allclose(templ.flux_flam(z=0.3, redshift_type='nearest'), 1.))
    assert(np.allclose(templ.flux_flam(z=1.5, redshift_type='nearest'), 2.))
    
    assert(np.allclose(templ.flux_flam(z=-1., redshift_type='interp'), 1.))
    assert(np.allclose(templ.flux_flam(z=0.0, redshift_type='interp'), 1.))
    assert(np.allclose(templ.flux_flam(z=0.3, redshift_type='interp'), 1.3))
    assert(np.allclose(templ.flux_flam(z=1.5, redshift_type='interp'), 2.))
    
    
def test_integrate_filter():
    """
    Integrating templates through filter throughput
    """
    import astropy.units as u
    
    # Tophat filter
    wx = np.arange(5400, 5600., 1)
    wy = wx*0.
    wy[10:-10] = 1
    
    f1 = filters.FilterDefinition(wave=wx, throughput=wy)
    
    # Flat-fnu spectrum
    wave = np.arange(1000., 9000.)
    fnu = np.ones((2, len(wave)))*u.microJansky
    fnu[1,:] *= 2
        
    flam = fnu.to(utils.FLAM_CGS, 
            equivalencies=u.equivalencies.spectral_density(wave*u.Angstrom))
    
    templ = templates.Template(arrays=(wave, flam), redshifts=[0,1])
    
    fnu_int = templ.integrate_filter(f1, z=0)
    assert(np.allclose(fnu_int*utils.FNU_CGS, 1*u.microJansky))
    
    fnu_int = templ.integrate_filter(f1, z=0, scale=2.)
    assert(np.allclose(fnu_int*utils.FNU_CGS, 2*u.microJansky))
    
    fnu_int = templ.integrate_filter(f1, z=0.3, redshift_type='nearest')
    assert(np.allclose(fnu_int*utils.FNU_CGS, 1*u.microJansky))

    fnu_int = templ.integrate_filter(f1, z=0.3, redshift_type='interp')                         
    assert(np.allclose(fnu_int*utils.FNU_CGS, 1.3*u.microJansky))
    
    # Return f-lambda
    for z in [0, 0.2]:
        flam_interp = templ.integrate_filter(f1, z=z, flam=True, 
                                         redshift_type='nearest')

        wz = f1.pivot*(1+z)*u.Angstrom
        flam_unit = (1*u.microJansky).to(utils.FLAM_CGS,
                           equivalencies=u.equivalencies.spectral_density(wz))

        assert(np.allclose(flam_interp*utils.FLAM_CGS, flam_unit))
    
    