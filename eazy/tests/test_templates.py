import pytest
import os

import numpy as np

from .. import utils
from .. import templates

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

