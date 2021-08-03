import numpy as np

from .. import igm

def test_igm():
    """
    Test IGM module (Inoue14)
    """
    
    igm_obj = igm.Inoue14()
    
    # Test against result from a particular version
    # (db97f839cf8afe4a22c31c5d6195fd707ba4de32)
    z = 3.0
    rest_wave = np.arange(850, 1251, 50)
    igm_val = np.array([0.33537573, 0.54634578, 0.74207249, 0.74194787, 
                        0.79182545, 0.75792504, 0.72135181, 0.68233589, 1.0])
    
    assert(np.allclose(igm_obj.full_IGM(z, rest_wave*(1+z)), igm_val, rtol=1.e-2))
    
    # With scaling               
    scale_tau = 2.    
    igm_obj.scale_tau = scale_tau
    
    igm_scaled = np.exp(2*np.log(igm_val))
    assert(np.allclose(igm_obj.full_IGM(z, rest_wave*(1+z)), igm_scaled, 
                       rtol=1.e-2))
    
    igm_obj = igm.Inoue14(scale_tau=scale_tau)
    assert(np.allclose(igm_obj.full_IGM(z, rest_wave*(1+z)), igm_scaled, 
                       rtol=1.e-2))
    
    