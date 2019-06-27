import unittest
import numpy as np

from .. import utils

class Dummy(unittest.TestCase):  
    def test_interp_conserve(self):
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
        
    def test_log_zgrid(self):
        """
        Test log_zgrid function
        """
        ref = np.array([0.1, 0.21568801, 0.34354303, 0.48484469, 
                        0.64100717, 0.8135934])
        
        vals = utils.log_zgrid(zr=[0.1, 1], dz=0.1)
        np.testing.assert_allclose(vals, ref, rtol=1e-04, atol=1.e-4,
                                   equal_nan=False, err_msg='', verbose=True)
        