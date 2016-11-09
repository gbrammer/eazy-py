def running_median(xi, yi, NBIN=10, use_median=True, use_nmad=True, reverse=False):
    """
    Running median/biweight/nmad
    """
    import numpy as np
    import astropy.stats
    
    NPER = xi.size // NBIN
    xm = np.arange(NBIN)*1.
    xs = xm*0
    ym = xm*0
    ys = xm*0
    N = np.arange(NBIN)
    
    so = np.argsort(xi)
    if reverse:
        so = so[::-1]
        
    idx = np.arange(NPER, dtype=int)
    for i in range(NBIN):
        N[i] = xi[so][idx+NPER*i].size
        
        if use_median:
            xm[i] = np.median(xi[so][idx+NPER*i])
            ym[i] =  np.median(yi[so][idx+NPER*i])
        else:
            xm[i] = astropy.stats.biweight_location(xi[so][idx+NPER*i])
            ym[i] = astropy.stats.biweight_location(yi[so][idx+NPER*i])
            
        if use_nmad:
            mad = astropy.stats.median_absolute_deviation
            ys[i] = 1.48*mad(yi[so][idx+NPER*i])
        else:
            ys[i] = astropy.stats.biweight_midvariance(yi[so][idx+NPER*i])
            
    return xm, ym, ys, N

def nmad(arr):
    import astropy.stats
    return 1.48*astropy.stats.median_absolute_deviation(arr)
    