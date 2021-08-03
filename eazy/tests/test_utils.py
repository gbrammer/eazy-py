import numpy as np

from .. import utils

def test_milky_way():
    """
    Test that milky way extinction is avaliable
    """
    import astropy.units as u
    
    f99 = utils.GalacticExtinction(EBV=1./3.1, Rv=3.1)
    
    # Out of range, test that at least run
    _ = f99(10)
    _ = f99(5.e4)
    _ = f99(10.e4)
    
    # Data types
    np.testing.assert_allclose(f99(5500), 1., rtol=0.05, atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)

    np.testing.assert_allclose(f99(5500.), 1., rtol=0.05, atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)
    
    np.testing.assert_allclose(f99(5500.*u.Angstrom), 1., rtol=0.05, 
                               atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)
    
    np.testing.assert_allclose(f99(0.55*u.micron), 1., rtol=0.05, 
                               atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)

    np.testing.assert_allclose(f99(100*u.micron), 0., rtol=0.05, 
                               atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)
                                   
    # Arrays
    np.testing.assert_allclose(f99([5500., 5500.]), 1., rtol=0.05, 
                               atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)
    
    arr = np.ones(10)*5500.                            
    np.testing.assert_allclose(f99(arr), 1., rtol=0.05, 
                               atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)

    np.testing.assert_allclose(f99(arr*u.Angstrom), 1., rtol=0.05, 
                               atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)

    np.testing.assert_allclose(f99(arr*u.Angstrom), 1., rtol=0.05, 
                               atol=0.05,
                               equal_nan=False, err_msg='', verbose=True)
                               

def test_interp_conserve():
    """
    Test interpolation function
    """
    # High-frequence sine function, should integrate to zero
    ph = 0.1
    xp = np.arange(-np.pi,3*np.pi,0.001)
    fp = np.sin(xp/ph)
    fp[(xp <= 0) | (xp > 2*np.pi)] = 0

    x = np.arange(-np.pi/2,2.5*np.pi,1)

    y1 = utils.interp_conserve(x, xp, fp)
    integral = np.trapz(y1, x)
    
    np.testing.assert_allclose(integral, 0., rtol=1e-04, atol=1.e-4,
                               equal_nan=False, err_msg='', verbose=True)


def test_log_zgrid():
    """
    Test log_zgrid function
    """
    ref = np.array([0.1, 0.21568801, 0.34354303, 0.48484469, 
                    0.64100717, 0.8135934])
    
    vals = utils.log_zgrid(zr=[0.1, 1], dz=0.1)
    np.testing.assert_allclose(vals, ref, rtol=1e-04, atol=1.e-4,
                               equal_nan=False, err_msg='', verbose=True)


def test_invert():
    """
    Test matrix invert helper
    """        
    a = np.array([[1., 2.], [3., 4.]])
    ainv = utils.safe_invert(a)
    assert(np.allclose(np.dot(a, ainv), np.eye(2)))


def test_query_string():
    """
    """
    ra, dec = 53.14474, -27.78552
    
    qstr = utils.query_html(ra, dec, with_coords=True, replace_comma=False, 
                            queries=['CDS'])
    
    expected = '(53.144740, -27.785520) <a href="http://vizier.u-strasbg.fr/viz-bin/VizieR?-c=53.14474+%2D27.78552&-c.rs=1.0">CDS</a>'
    assert(qstr == expected)
                    
    for co in [True, False]:
        for re in [True, False]:
            qstr = utils.query_html(ra, dec, with_coords=co, replace_comma=re, 
                            queries=['CDS','ESO','MAST','ALMA', 'LEG','HSC'])
                            
    