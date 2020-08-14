import pytest
import os

import numpy as np

from .. import igm

def test_igm():
    """
    Test IGM module (Inoue14)
    """
    
    igm_obj = igm.Inoue14()
    
    z = 3.0
    rest_wave = np.array([4800.])
    igm_val = np.array([0.6823])
    
    assert(np.allclose(igm_obj.full_IGM(z, rest_wave), igm_val, rtol=1.e-2))
    
    # With scaling               
    scale_tau = 2.    
    igm_obj.scale_tau = scale_tau
    
    igm_scaled = np.exp(2*np.log(igm_val))
    assert(np.allclose(igm_obj.full_IGM(z, rest_wave), igm_scaled, 
                       rtol=1.e-2))
    
    igm_obj = igm.Inoue14(scale_tau=scale_tau)
    assert(np.allclose(igm_obj.full_IGM(z, rest_wave), igm_scaled, 
                       rtol=1.e-2))
    
    