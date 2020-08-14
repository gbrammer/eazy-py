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