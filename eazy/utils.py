import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import astropy.stats
import astropy.units as u

CLIGHT = 299792458.0 # m/s

TRUE_VALUES = [True, 1, '1', 'True', 'TRUE', 'true', 'y', 'yes', 'Y', 'Yes']
FALSE_VALUES = [False, 0, '0', 'False', 'FALSE', 'false', 'n', 'no', 'N', 'No']

FNU_CGS = u.erg/u.second/u.cm**2/u.Hz
FLAM_CGS = u.erg/u.second/u.cm**2/u.Angstrom

def bool_param(value, false_values=FALSE_VALUES, true_values=TRUE_VALUES, which='false', check_both=True):
    """
    Flexible booleans
    
    If ``which == 'false'``, test that ``value not in false_values``.
    
    If ``which == 'true'``, test ``value in true_values``.
    
    If ``check_both`` and ``value`` isn't in either list, return the value 
    itself.
    
    """
    if which == 'false':
        test = value not in false_values
    elif which == 'true':
        test = value in true_values
    else:
        raise ValueError("Option ``which`` must be 'true' or 'false'")
    
    if check_both:
        if value not in true_values + false_values:
            test = value
            
    return test


def path_to_eazy_data():
    """
    Return internal path to ``eazy/data``.
    """
    return os.path.join(os.path.dirname(__file__), 'data')


def set_warnings(numpy_level='ignore', astropy_level='ignore'):
    """
    Set global numpy and astropy warnings
    
    Parameters
    ----------
    numpy_level : 'ignore', 'warn', 'raise', 'call', 'print', 'log'
        Numpy error level (see `numpy.seterr`).
        
    astropy_level : 'error', 'ignore', 'always', 'default', 'module', 'once'
        Astropy error level (see `warnings.simplefilter`).
    
    """
    from astropy.utils.exceptions import AstropyWarning
    
    np.seterr(all=numpy_level)
    warnings.simplefilter(astropy_level, category=AstropyWarning)


def running_median(xi, yi, NBIN=10, reverse=False, bins=None, x_func=np.median, y_func=np.median, std_func=astropy.stats.mad_std, x_kwargs={}, y_kwargs={}, std_kwargs={}, use_biweight=False, integrate=False, **kwargs):
    """
    Binned median/biweight/nmad statistics
    
    Parameters
    ----------
    xi : array-like
        Data of independent variable
    
    yi : array-like
        Data of dependent variable
    
    NBIN : int
        Number of bins along `xi`
    
    reverse : bool
        Calculate bins starting at largest values of `xi`
    
    bins : array-like
        Fixed bins, rather than calculating with `NBIN`
    
    x_func : function
        Function to compute moments of `xi`
    
    y_func, std_func : function
        Functions to compute moments of `yi`.  Assumed to be the central 
        value and dispersion, but don't have to be
    
    x_kwargs, y_kwargs, std_kwargs : dict
        Keyword arguments to pass to moment functions
    
    use_biweight : bool
        Use robust biweight estimators:
        
            - `x_func` : `astropy.stats.biweight_location`
            - `y_func` : `astropy.stats.biweight_location`
            - `std_func` : `astropy.stats.biweight_midvariance`
        
    integrate : bool
        Numerically integrate `yi` with the trapezoidal rule within the bins
    
    Returns
    -------
    xm, ym, ys : array-like
        Binned moments of `xi` and `yi`
    
    yn : array-like
        Number of entries per bin
        
    """
    
    NPER = xi.size // NBIN
    if bins is None:
        so = np.argsort(xi)
        if reverse:
            so = so[::-1]
        
        bx = np.linspace(0,len(xi),NBIN+1)
        bins = np.interp(bx, np.arange(len(xi)), xi[so])
        if reverse:
            bins = bins[::-1]
    
    NBIN = len(bins)-1
           
    xm = np.ones(NBIN)
    xs = np.zeros_like(xm)
    ym = np.zeros_like(xm)
    ys = np.zeros_like(xm)
    N = np.zeros(NBIN, dtype=int)
    
    if use_biweight:
        x_func = astropy.stats.biweight_location
        y_func = astropy.stats.biweight_location
        std_func = astropy.stats.biweight_midvariance
        
    #idx = np.arange(NPER, dtype=int)
    for i in range(NBIN):
        in_bin = (xi > bins[i]) & (xi <= bins[i+1])
        N[i] = in_bin.sum() #N[i] = xi[so][idx+NPER*i].size
        
        if integrate:
            xso = np.argsort(xi[in_bin])
            ma = xi[in_bin].max()
            mi = xi[in_bin].min()
            xm[i] = (ma+mi)/2.
            dx = (ma-mi)
            ym[i] = np.trapz(yi[in_bin][xso], xi[in_bin][xso])/dx
        else:
            xm[i] = x_func(xi[in_bin], **x_kwargs)
            ym[i] = y_func(yi[in_bin], **y_kwargs)
        
        ys[i] = std_func(yi[in_bin], **std_kwargs)
                    
    return xm, ym, ys, N


def nmad(data):
    """
    Normalized median absolute deviation statistic
    
    Just a wrapper around `astropy.stats.mad_std`.
    """
    import astropy.stats
    #return 1.48*astropy.stats.median_absolute_deviation(arr)
    return astropy.stats.mad_std(data)


def log_zgrid(zr=[0.7,3.4], dz=0.01):
    """Make a logarithmically spaced redshift grid
    
    Parameters
    ----------
    zr : [float, float]
        Minimum and maximum of the desired grid
    
    dz : float
        Step size, dz/(1+z)
    
    Returns
    -------
    zgrid : array-like
        Redshift grid
    
    """
    zgrid = np.exp(np.arange(np.log(1+zr[0]), np.log(1+zr[1]), dz))-1
    return zgrid


def trapz_dx(x):
    """
    Return trapezoid rule coefficients, useful for numerical integration 
    using a dot product
    
    Parameters
    ----------
    x : array-like
        Independent variable
    
    Returns
    -------
    dx : array_like
        Coefficients for trapezoidal rule integration.
    """
    dx = np.zeros_like(x)
    diff = np.diff(x)/2.
    dx[:-1] += diff
    dx[1:] += diff
    return dx


def clipLog(im, lexp=1000, cmap=[-1.4914, 0.6273], scale=[-0.1,10]):
    """
    Return normalized array like DS9 log
    """
    import numpy as np
    
    contrast, bias = cmap
    clip = (np.clip(im, scale[0], scale[1])-scale[0])/(scale[1]-scale[0])
    clip_log = np.clip((np.log10(lexp*clip+1)/np.log10(lexp)-bias)*contrast+0.5, 0, 1)
    
    return clip_log


def get_mw_dust(ra, dec, **kwargs):
    """
    Wrapper around functions to try to query for the MW E(B-V)
    """
    try:
        ebv = get_dustmaps_dust(ra, dec, web=True)
        return ebv
    except:
        pass
        
    try:
        ebv = get_dustmaps_dust(ra, dec, web=False)
        return ebv
    except:
        pass
    
    try:
        ebv = get_irsa_dust(ra, dec, **kwargs)
        return ebv
    except:
        pass
    
    return 0.00
    
def get_dustmaps_dust(ra, dec, web=True, **kwargs):
    "Use https://github.com/gregreen/dustmaps"
    
    from dustmaps.sfd import SFDQuery, SFDWebQuery
    from astropy.coordinates import SkyCoord
    
    coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
    
    if web:
        sfd = SFDWebQuery()
    else:
        sfd = SFDQuery()
        
    ebv = sfd(coords)
    return ebv

def get_irsa_dust(ra=53.1227, dec=-27.805089, type='SandF'):
    """
    Get Galactic dust reddening from NED/IRSA at a given position
    http://irsa.ipac.caltech.edu/applications/DUST/docs/dustProgramInterface.html
    
    Parameters
    ----------
    ra, dec : float
        RA/Dec in decimal degrees.
        
    type : 'SFD', 'SandF'
        Dust model, with        
            SandF = Schlafly & Finkbeiner 2011 (ApJ 737, 103) 
              SFD = Schlegel et al. 1998 (ApJ 500, 525)
    
    Returns
    -------
    ebv : float
        Color excess E(B-V), in magnitudes
    
    """
    import os
    import tempfile   
    import urllib.request
    from astropy.table import Table
    from lxml import objectify
    
    query = 'http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={0:.4f}+{1:.4f}+equ+j2000'.format(ra, dec)
    
    req = urllib.request.Request(query)
    response = urllib.request.urlopen(req)
    resp_text = response.read().decode('utf-8')
    
    root = objectify.fromstring(resp_text)
    stats = root.result.statistics

    if type == 'SFD':
        return float(str(stats.refPixelValueSFD).split()[0])
    else:
        return float(str(stats.refPixelValueSandF).split()[0])


def fill_between_steps(x, y, z, ax=None, *args, **kwargs):
    """
    Make `matplotlib.pyplot.fill_between` work like linestyle='steps-mid'.
    """
    so = np.argsort(x)
    mid = x[so][:-1] + np.diff(x[so])/2.
    xfull = np.append(np.append(x, mid), mid+np.diff(x[so])/1.e6)
    yfull = np.append(np.append(y, y[:-1]), y[1:])
    zfull = np.append(np.append(z, z[:-1]), z[1:])
    
    so = np.argsort(xfull)
    if ax is None:
        ax = plt.gca()
    
    ax.fill_between(xfull[so], yfull[so], zfull[so], *args, **kwargs)


def safe_invert(arr):
    """
    Version-safe matrix inversion using `numpy.linalg.inv` or `numpy.matrix.I`
    """
    try:
        from numpy.linalg import inv
        _inv = inv(arr)
    except:
        _inv = np.matrix(arr).I.A
    
    return _inv


class GalacticExtinction(object):
    def __init__(self, EBV=0, Rv=3.1, force=None, radec=None, ebv_type='SandF'):
        """
        Wrapper to use either `specutils.extinction` or the `extinction` 
        modules, which have different calling formats.  The results from 
        both of these modules should be equivalent.
                
        Parameters
        ----------
        EBV : float
            Galactic reddening, e.g., from `https://irsa.ipac.caltech.edu/applications/DUST/`.
            
        Rv : float
            Selective extinction ratio, `Rv=Av/(E(B-V))`.
        
        radec : None or (float, float)
            If provided, try to determine EBV based on these coordinates 
            with `get_irsa_dust(type=[ebv_type])` or `dustmaps`. 
            
        force : None, 'extinction', 'specutils.extinction'
            Force use one or the other modules.  If `None`, then first try
            to import `specutils.extinction` and if that fails use
            `extinction`.
        """
        import importlib
        
        # Import handler
        if force == 'specutils.extinction':
            import specutils.extinction
            self.module = 'specutils.extinction'
        elif force == 'extinction':
            from extinction import Fitzpatrick99
            self.module = 'extinction'
        elif force == 'dust_extinction':
            from dust_extinction.parameter_averages import F99
            self.module = 'dust_extinction'
        else:
            modules = [['dust_extinction.parameter_averages', 'F99'], 
                       ['extinction','Fitzpatrick99'],
                       ['specutils.extinction','ExtinctionF99']]
            
            self.module = None
            for (mod, cla) in modules:
                try:
                    _F99 = getattr(importlib.import_module(mod), cla)
                    self.module = mod
                    break
                except:
                    continue
            
            if self.module is None:
                raise ImportError("Couldn't import extinction module from "
                                  "dust_extinction, extinction or specutils") 
                                       
            # try:
            #     from specutils.extinction import ExtinctionF99
            #     self.module = 'specutils.extinction'
            # except:
            #     from extinction import Fitzpatrick99
            #     self.module = 'extinction'
        
        if radec is not None:
            self.EBV = get_mw_dust(ra=radec[0], dec=radec[1], type=ebv_type)
        else:    
            self.EBV = EBV
        
        self.Rv = Rv
        
        if self.module == 'dust_extinction.parameter_averages':
            self.f99 = _F99(Rv=self.Rv)
            
        elif self.module == 'specutils.extinction':
            self.f99 = _F99(self.Av)
            #self.Alambda = f99(self.wave*u.angstrom)
        else:
            self.f99 = _F99(self.Rv)
            #self.Alambda = f99(self.wave*u.angstrom, Av)
    
    @property
    def Av(self):
        return self.EBV*self.Rv
    
    @property    
    def info(self):
        msg = ('F99 extinction with `{0}`: Rv={1:.1f}, '
              'E(B-V)={2:.3f} (Av={3:.2f})')
        return msg.format(self.module, self.Rv, self.EBV, self.Av)
        
    def __call__(self, wave):
        """
        Compute Fitzpatrick99 extinction.
        
        Parameters
        ----------
        wave : float or `numpy.ndarray`
            Observed-frame wavelengths.  If no `unit` attribute available, 
            assume units are `astropy.units.Angstrom`.
        
        Returns
        -------
        Alambda : like ``wave``
            F99 extinction (mags) as a function of wavelength.  Output will
            be set to zero below 909 Angstroms and above 6 microns as the
            extinction modules themselves don't compute outside that range.
            
        """
        import astropy.units as u
        if not hasattr(wave, 'unit'):
            unit = u.Angstrom
        else:
            unit = 1
                       
        inwave = np.squeeze(wave).flatten()
        if self.module == 'dust_extinction.parameter_averages':
            clip = (inwave*unit > 1/10.*u.micron) 
            clip &= (inwave*unit < 1/0.3*u.micron)
        else:
            clip = (inwave*unit > 909*u.angstrom) & (inwave*unit < 6*u.micron)
        Alambda = np.zeros(inwave.shape)
                
        if clip.sum() == 0:
            return Alambda
        else:
            if self.module == 'dust_extinction.parameter_averages':
                flam = self.f99.extinguish(inwave[clip]*unit, Av=self.Av)
                Alambda[clip] = -2.5*np.log10(flam)
                
            elif self.module == 'specutils.extinction':
                Alambda[clip] = self.f99(inwave[clip]*unit)
            else:
                Alambda[clip] = self.f99(inwave[clip]*unit, self.Av)
        
        return Alambda

def abs_mag_to_luminosity(absmag, pivot=None, output_unit=u.L_sun):
    """
    Convert absolute AB mag to luminosity units
    
    Parameters
    ----------
    absmag : array-like
        Absolute AB magnitude.
    
    pivot : float
        Filter pivot wavelength associated with the magnitude.  If no units, 
        then assume `astropy.units.Angstrom`.
    
    output_unit : `astropy.units.core.Unit`
        Desired output unit.  Must specify a ``pivot`` wavelength for output
        power units, e.g., `astropy.unit.L_sun`.
    
    """
    if pivot is None:
        nu = 1.
    else:
        if hasattr(pivot, 'unit'):
            wunit = 1
        else:
            wunit = u.Angstrom
            
        nu = ((CLIGHT*u.m/u.second)/(pivot*wunit)).to(u.Hz)
        
    fjy = 3631*u.jansky * 10**(-0.4*absmag)
    d10 = (10*u.pc).to(u.cm)
    f10 = fjy * 4 * np.pi * d10**2 * nu
    return f10.to(output_unit)
    
def zphot_zspec(zphot, zspec, zlimits=None, zmin=0, zmax=4, axes=None, figsize=[6,7], minor=0.5, skip=2, selection=None, catastrophic_limit=0.15, title=None, min_zphot=0.02, alpha=0.2, extra_xlabel='', extra_ylabel='', xlabel=r'$z_\mathrm{spec}$', ylabel=r'$z_\mathrm{phot}$', label_pos=(0.05, 0.95), label_kwargs=dict(ha='left', va='top', fontsize=10), label_prefix='', format_axes=True, color='k', point_label=None, **kwargs):
    """
    Make zphot_zspec plot scaled by log(1+z) and show uncertainties

    Parameters
    ----------
    zphot : array-like
        Redshift on dependent axis

    zspec : array-like
        Redshift on independent axis

    zlimits : (N, 2) array
        Redshifts to use for photo-z errorbars, e.g. from 
        `~eazy.photoz.Photoz.pz_percentiles`, where `N` is the number of 
        objects as in `zphot` and `zspec`

    zmin, zmax : float
        Plot limits

    axes : `matplotlib` axes, None
        If specified, overplot in existing axes rather than generating a new
        plot.  For example, run the function once to generate the figure and
        then plot different points onto the existing axes:
        
        >>> fig = eazy.utils.zphot_spec(zphot, zspec, selection=sample1)
        >>> _ = eazy.utils.zphot_spec(zphot, zspec, selection=sample2, 
        >>>                           axes=fig.axes, color='b')

    figsize : list
        Figure canvas dimensions

    minor : float
        Axis tick interval

    skip : int
        Put axis labels every `skip` ticks

    selection : array-like
        Subsample selection (boolean or indices) applied as `zphot[selection]`

    catastrophic_limit : float
        Limit to define "catastrophic" failures, which is used for computing
        precision / outlier statistics printed on the plot

    title : str
        Title to add to the plot axes

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure object

    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    clip = (zphot > min_zphot) & (zspec > zmin) & (zspec <= zmax)

    if selection is not None:
        clip &= selection

    dz = (zphot-zspec)/(1+zspec)

    #izbest = np.argmin(self.fit_chi2, axis=1)

    clip_cat = (np.abs(dz) < catastrophic_limit)
    frac_cat = 1-(clip & clip_cat).sum() / clip.sum()
    NOUT = (clip & ~clip_cat).sum()
    
    gs = GridSpec(2,1, height_ratios=[6,1])
    NEW_AXES = axes is None
    if NEW_AXES:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(gs[0,0])
    else:
        ax = axes[0]
        fig = None
        
    if title is not None:
        ax.set_title(title)

    if zlimits is not None:
        yerr = np.log10(1+np.abs(zlimits.T - zphot))
        ax.errorbar(np.log10(1+zspec[clip & ~clip_cat]), 
                    np.log10(1+zphot[clip & ~clip_cat]), 
                    yerr=yerr[:,clip & ~clip_cat], marker='.', alpha=alpha, 
                    color='r', linestyle='None')
        
        ax.errorbar(np.log10(1+zspec[clip & clip_cat]), 
                    np.log10(1+zphot[clip & clip_cat]), 
                    yerr=yerr[:,clip & clip_cat], marker='.', alpha=alpha, 
                    color=color, linestyle='None', label=point_label)
    else:
        ax.scatter(np.log10(1+zspec[clip & ~clip_cat]),
                   np.log10(1+zphot[clip & ~clip_cat]), 
                   marker='.', alpha=alpha, color='r')

        ax.scatter(np.log10(1+zspec[clip & clip_cat]), 
                   np.log10(1+zphot[clip & clip_cat]), 
                   marker='.', alpha=alpha, color=color, label=point_label)
        
    if NEW_AXES | format_axes:
        xt = np.arange(zmin, zmax+0.1, minor)
        xl = np.log10(1+xt)
        ax.plot(xl, xl, color='r', alpha=0.5)
        ax.set_xlim(xl[0], xl[-1])
        ax.set_ylim(xl[0],xl[-1])
        xtl = list(xt)

        if skip > 0:
            for i in range(1, len(xt), skip):
                xtl[i] = ''

        ax.set_xticks(xl)
        if axes is None:
            ax.set_xticklabels([])
        else:
            if len(axes) == 1:
                ax.set_xticks(xl)
                ax.set_xticklabels(xtl);
                ax.set_xlabel(xlabel + extra_xlabel)
            
        ax.set_yticks(xl); ax.set_yticklabels(xtl);
        ax.set_ylabel(ylabel + extra_ylabel)

    sample_nmad = nmad(dz[clip])
    sample_cat_nmad = nmad(dz[clip & clip_cat])

    if label_pos is not None:
        msg = r'{label_prefix} N={N} ({NOUT}, {err_frac:4.1f}%), $\sigma$={sample_nmad:.4f} ({sample_cat_nmad:.4f})'
        msg = msg.format(label_prefix=label_prefix, 
                         N=clip.sum(), err_frac=frac_cat*100, 
                         sample_nmad=sample_nmad, 
                         sample_cat_nmad=sample_cat_nmad, NOUT=NOUT)
                         
        ax.text(label_pos[0], label_pos[1], msg, transform=ax.transAxes)
    

    if axes is None:
        ax = fig.add_subplot(gs[1,0])
    else:
        if len(axes) == 2:
            ax = axes[1]
        else:
            return True
            
    if zlimits is not None:
        yerr = np.abs(zlimits.T-zphot)
        ax.errorbar(np.log10(1+zspec[clip & ~clip_cat]), dz[clip & ~clip_cat], 
                    yerr=yerr[:,clip & ~clip_cat], 
                    marker='.', alpha=alpha, color='r', linestyle='None')
                    
        ax.errorbar(np.log10(1+zspec[clip & clip_cat]), dz[clip & clip_cat], 
                    yerr=yerr[:,clip & clip_cat], 
                    marker='.', alpha=alpha, color='k', linestyle='None')
    else:
        ax.scatter(np.log10(1+zspec[clip & ~clip_cat]), dz[clip & ~clip_cat], 
                    marker='.', alpha=alpha, color='r')
        ax.scatter(np.log10(1+zspec[clip & clip_cat]), dz[clip & clip_cat],
                    marker='.', alpha=alpha, color='k')
        
    if fig is not None:
        ax.set_xticks(xl); ax.set_xticklabels(xtl);
        ax.set_xlim(xl[0], xl[-1])
        ax.set_ylim(-6*sample_nmad, 6*sample_nmad)
        ax.set_yticks([-3*sample_nmad, 0, 3*sample_nmad])
        ax.set_yticklabels([r'$-3\sigma$',r'$0$',r'$+3\sigma$'])
        ax.set_xlabel(xlabel + extra_xlabel)
        ax.set_ylabel(r'$\Delta z / 1+z$')
        for a in fig.axes:
            a.grid()

        fig.tight_layout(pad=0.1)
        return fig
    else:
        return True


def query_html(ra, dec, with_coords=True, replace_comma=True, queries=['CDS','ESO','MAST','ALMA', 'LEG','HSC'], **kwargs):
    """
    Return HTML string of queries around a position
    
    Parameters
    ----------
    ra, dec : float
        Coordinates in decimal degrees
    
    with_coords : bool
        Include '(ra, dec)' in output string
    
    replace_comma : bool
        Replace ',' with URL-safe '%2C'
    
    queries : list
        - CDS: Vizier/CDS catalogs
        - ESO: ESO archive
        - MAST: STScI/MAST HST archive
        - ALMA: ALMA archive
        - LEG/LEGACY: LegacySurvey map interface
        - HSC: HSC map interface
    
    Returns
    -------
    html : str
        HTML-formatted string with query links
        
    """
    if with_coords:
        html = [f"({ra:.6f}, {dec:.6f})"]
    else:
        html = []
    
    # Function/name mapping
    funcs = [cds_query, eso_query, mast_query, alma_query, show_legacysurvey, hscmap_query]
    names = ['CDS','ESO','MAST','ALMA', 'LEG','HSC']
    query_map = {}
    for name, func in zip(names, funcs):
        query_map[name] = func
    
    query_map['LEGACY'] = query_map['LEG']
    
    for name in queries:
        if name in query_map:
            func = query_map[name]
        else:
            continue
            
        url = func(ra, dec, **kwargs)
        html.append(f'<a href="{url}">{name}</a>')
    
    html = ' '.join(html)
    
    if replace_comma:
        html = html.replace(',','%2C')
        
    return html


def cds_query(ra, dec, radius=1., unit='s', **kwargs):
    """
    Open browswer with CDS catalog query around central position
    
    """
    #rd = self.get('pan fk5').strip()
    rd = f'{ra} {dec}'
    rdst = rd.replace('+', '%2B').replace('-', '%2D').replace(' ', '+')
    url = (f'http://vizier.u-strasbg.fr/viz-bin/VizieR?'
           f'-c={rdst}&-c.r{unit}={radius}')
           
    #os.system(f'open {url}')
    return url


def eso_query(ra, dec, radius=1., unit='m', dp_types=['CUBE','IMAGE'], extra='', **kwargs):
    """
    Open browser with ESO archive query around central position.
    
    Note: ESO query is data footprint **contains*** point
    
    """
    #ra, dec = self.get('pan fk5').strip().split()
    
    # native is deg
    if unit == 'd':
        r = f'{radius:.2f}'
    elif unit == 'm':
        r = f'{radius/60:.3f}'
    elif unit == 's':
        r = f'{radius/3600:.5f}'
        
    dp_type = ','.join(dp_types)
    
    url = (f'https://archive.eso.org/scienceportal/home?'
            f'pos={ra},{dec}&r={r}&dp_type={dp_type}{extra}')
                    
    #os.system(f'open {url}')
    return url


def mast_query(ra, dec, instruments=['WFC3','ACS','WFPC2'], mast_radius=1., mast_unit='m', max=1000, **kwargs):
    """
    Open browser with MAST archive query around central position
    
    Note: MAST query is **distance to** point
    
    """
    #ra, dec = self.get('pan fk5').strip().split()
    if len(instruments) > 0:
        instr='&sci_instrume='+','.join(instruments)
    else:
        instr = ''
    
    # native is arcmin
    if mast_unit == 'd':
        r = f'{mast_radius*60:.2f}'
    elif mast_unit == 'm':
        r = f'{mast_radius:.3f}'
    elif mast_unit == 's':
        r = f'{mast_radius/60:.5f}'
        
    url = (f'https://archive.stsci.edu/hst/search.php?RA={ra}&DEC={dec}'
           f'&radius={r}'
           f'&sci_aec=S{instr}&max_records={max}&outputformat=HTML_Table'
            '&action=Search')
            
    #os.system(f'open {url}')
    return url


def alma_query(ra, dec, mirror="almascience.eso.org", radius=1, unit='m', extra='', **kwargs):
    """
    Open browser with ALMA archive query around central position
    """

    # native is arcmin
    if unit == 'd':
        r = f'{radius*60:.2f}'
    elif unit == 'm':
        r = f'{radius:.3f}'
    elif unit == 's':
        r = f'{radius/60:.5f}'
        
    url = (f"https://{mirror}/aq/?result_view=observation"
           f"&raDec={ra}%20{dec},{r}{extra}")
    #os.system(f'open "{url}"')
    return url


def hscmap_query(ra, dec, open=True, **kwargs):
    """
    Function to open HSC explorer in browser centered on target coordinates
    """
    
    import os
    rrad = ra/180*np.pi
    drad = dec/180*np.pi
    url = (f"https://hsc-release.mtk.nao.ac.jp/hscMap-pdr2/app/#/?_=%7B%22view%22%3A%7B%22a%22%3A{rrad},%22d%22%3A{drad}"
           ",%22fovy%22%3A0.00009647627785850188,%22roll%22%3A0%7D,%22sspParams%22%3A%7B%22type%22%3A%22"
           "SDSS_TRUE_COLOR%22,%22filter%22%3A%5B%22HSC-Y%22,%22HSC-Z%22,%22HSC-I%22%5D,%22simpleRgb"
           "%22%3A%7B%22beta%22%3A22026.465794806718,%22a%22%3A1,%22bias%22%3A0.05,%22b0%22%3A0%7D,%22"
           "sdssTrueColor%22%3A%7B%22beta%22%3A40106.59228119989,%22a%22%3A2.594451857120983,%22bias%22%3A0.05,"
           "%22b0%22%3A0%7D%7D,%22externalTiles%22%3A%5B%5D,%22activeReruns%22%3A%5B%22pdr2_wide%22,%22pdr2_dud"
           "%22%5D%7D")
        
    return url


def show_legacysurvey(ra, dec, layer='dr8', zoom=17, **kwargs):
    """
    Open browser with legacysurvey.org panner around central position
    """
    #ra, dec = self.get('pan fk5').strip().split()
    url = (f'http://legacysurvey.org/viewer?ra={ra}&dec={dec}'
           f'&layer={layer}&zoom={zoom}')
            
    #os.system(f'open {url}')
    return url


def interp_conserve(x, xp, fp, left=0., right=0.):
    """
    Interpolation analogous to `numpy.interp` but conserving "flux".
    
    Parameters
    ----------
    x : `numpy.ndarray`
        Desired interpolation locations

    xp, fp : `numpy.ndarray`
        The `x` and `y` coordinates of the function to be interpolated.  The
        `x` array can be irregularly spaced but should be increase
        monotonically.
    
    left, right : float
        Values to use for extrapolation below the minimum and maximum limits
        of `x`.
        
    Returns
    -------
    y : like `x`
        Interpolated values.
    
    Interpolation performed by trapezoidal integration between the midpoints
    of the output `x` array with `numpy.trapz`.
    
    .. note:: For a faster `cython` implementation of this function, see 
              `grizli.utils_c.interp_conserve_c`.
              
    """
    midpoint = (x[1:]-x[:-1])/2.+x[:-1]
    midpoint = np.append(midpoint, np.array([x[0],x[-1]]))
    midpoint = midpoint[np.argsort(midpoint)]
    int_midpoint = np.interp(midpoint, xp, fp, left=left, right=right)
    int_midpoint[midpoint > xp.max()] = right
    int_midpoint[midpoint < xp.min()] = left
    
    fullx = np.append(xp, midpoint)
    fully = np.append(fp, int_midpoint)
    
    so = np.argsort(fullx)
    fullx, fully = fullx[so], fully[so]
    
    outy = x*0.
    dx = midpoint[1:]-midpoint[:-1]
    for i in range(len(x)):
        bin = (fullx >= midpoint[i]) & (fullx <= midpoint[i+1])
        outy[i] = np.trapz(fully[bin], fullx[bin])/dx[i]
        
    return outy


class emceeChain():     
    def __init__(self, chain=None, file=None, param_names=[],
                       burn_fraction=0.5, sampler=None):
        
        self.param_names = []
        
        if chain is not None:
            self.chain = chain
        
        if file is not None:
            if 'fits' in file.lower():
                self.load_fits(file=file)            
            else:
                self.load_chain(file=file)
                     
        self.process_chain(param_names = param_names,
                           burn_fraction=burn_fraction)
        #
        if sampler is not None:
            from numpy import unravel_index
            max_ix = unravel_index(sampler.lnprobability.argmax(), sampler.lnprobability.shape)
            self.map = self.chain[max_ix[0], max_ix[1],:]
            self.is_map = True
        else:
            self.map = self.median
            self.is_map = False
            
    def process_chain(self, param_names=[], burn_fraction=0.5):
        """
        Define parameter names and get parameter statistics
        """
        self.nwalkers, self.nstep, self.nparam = self.chain.shape
                
        if param_names == []:            
            if self.param_names == []:
                for i in range(self.nparam):
                    param_names.append('a%d' %(i+1))
               
                self.param_names = param_names
        
        else:
            if len(param_names) != self.nparam:
                print('param_names must have N=%d (or zero) entries' %(self.nparam))
                return False
                        
            self.param_names = param_names
                
        self.param_dict = {}
        for i in range(self.nparam):
            self.param_dict[self.param_names[i]] = i
        
        self.nburn = int(np.round(burn_fraction*self.nstep))
        self.stats = {}
        self.median = np.zeros(self.nparam)
                
        for param in self.param_names:
            pid = self.param_dict[param]
            self.stats[param] = self.get_stats(pid, burn=self.nburn)
            self.median[pid] = self.stats[param]['q50']
            
    def get_stats(self, pid, burn=0, raw=False):
        """
        Get percentile statistics for a parameter in the chain
        """
        if raw:
            pchain = pid*1.
        else:
            pchain = self.chain[:,burn:,pid].flatten()
        
        stats = {}
        stats['q05'] = np.percentile(pchain, 5)
        stats['q16'] = np.percentile(pchain, 16)
        stats['q50'] = np.percentile(pchain, 50)
        stats['q84'] = np.percentile(pchain, 84)
        stats['q95'] = np.percentile(pchain, 95)
        stats['mean'] = np.mean(pchain)
        stats['std'] = np.std(pchain)
        stats['width'] = (stats['q84']-stats['q16'])/2.
        return stats
        
    def show_chain(self, param='a1', chain=None, alpha=0.15, color='blue', scale=1, diff=0, ax = None, add_labels=True, hist=False, autoscale=True, *args, **kwargs):
        """
        Make a plot of the chain for a given parameter.
        
        For plotting, multiply the parameter by `scale` and subtract `diff`.
        
        """
        if chain is None:
            pid = self.param_dict[param]
            chain = self.chain[:,:,pid]
        
        if ax is not None:
            plotter = ax
            xlabel = ax.set_xlabel
            ylabel = ax.set_ylabel
            ylim = ax.set_ylim
        else:
            plotter = plt
            xlabel = plt.xlabel
            ylabel = plt.ylabel
            ylim = plt.ylim
            
        if hist:
            h = plotter.hist(chain[:,self.nburn:].flatten(), alpha=alpha, color=color, *args, **kwargs)
            if add_labels:
                ylabel('N')
                xlabel(param)
        else:
            for i in range(self.nwalkers):
                p = plotter.plot(chain[i,:]*scale-diff, alpha=alpha, color=color, *args, **kwargs)
            if add_labels:
                xlabel('Step')
                ylabel(param)
            #
            if autoscale:
                ylim(self.stats[param]['q50']*scale + np.array([-8,8])*self.stats[param]['width']*scale)
            
    def save_chain(self, file='emcee_chain.pkl', verbose=True):
        """
        Save the chain to a Pkl file
        """
        import cPickle as pickle
        
        fp = open(file,'wb')
        pickle.dump(self.nwalkers, fp)
        pickle.dump(self.nstep, fp)
        pickle.dump(self.nparam, fp)
        pickle.dump(self.param_names, fp)
        pickle.dump(self.chain, fp)
        fp.close()
        
        if verbose:
            print('Wrote %s.' %(file))
        
    def load_chain(self, file='emcee_chain.pkl'):
        """
        Read the chain from the pickle file
        """
        import cPickle as pickle
        
        fp = open(file, 'rb')
        self.nwalkers = pickle.load(fp)
        self.nstep = pickle.load(fp)
        self.nparam = pickle.load(fp)
        self.param_names = pickle.load(fp)
        self.chain = pickle.load(fp)
        fp.close()
    
    def save_fits(self, file='emcee_chain.fits', verbose=True):
        """
        Make a FITS file of an EMCEE chain
        """
        import astropy.io.fits as pyfits
        
        header = pyfits.Header()
        header.update('NWALKERS', self.nwalkers)
        header.update('NSTEP', self.nstep)
        header.update('NPARAM', self.nparam)
        
        hdu = [pyfits.PrimaryHDU(header=header)]
        
        for param in self.param_names:
            header.update('PARAM', param)
            hdu.append(pyfits.ImageHDU(data=self.__getitem__(param), header=header, name=param))
        
        hduList = pyfits.HDUList(hdu)
        hduList.writeto(file, clobber=True, output_verify='silentfix')
    
        if verbose:
            print('Wrote %s.' %(file))
    
    def load_fits(self, file='emcee_chain.fits'):
        """
        Load emcee chain fits file created by ``save_fits``.
        """
        import astropy.io.fits as pyfits
        
        im = pyfits.open(file)
        self.nwalkers = im[0].header['NWALKERS']
        self.nstep = im[0].header['NSTEP']
        self.nparam = im[0].header['NPARAM']
        self.param_names = []
        self.chain = np.ones((self.nwalkers, self.nstep, self.nparam))
        for i in range(self.nparam):
            self.param_names.append(im[i+1].header['PARAM'])
            self.chain[:,:,i] = im[i+1].data
        
        im.close()
    
    def parameter_correlations(self, size=8, shrink=5, show=None, file=None):
        if show is None:
            show = self.param_names
        
        NP = len(show)
        #fig = unicorn.plotting.plot_init(square=True, aspect=1, xs=size, left=0.05, right=0.01, bottom=0.01, top=0.01, NO_GUI=False, use_tex=False, fontsize=7)
        fig = plt.figure(figsize=(7,7))
        fig.subplots_adjust(wspace=0.0,hspace=0.0)
        
        counter = 0
        for i in range(NP):
            for j in range(NP):
                counter = counter + 1
                ax = fig.add_subplot(NP, NP, counter)
                a = ax.plot(self[show[i]][:,self.nburn::shrink].flatten(), self[show[j]][:,self.nburn::shrink].flatten(), alpha=0.03, color='black', linestyle='None', marker=',')
                a = ax.set_xlim(self.stats[show[i]]['q50']-3*self.stats[show[i]]['std'], self.stats[show[i]]['q50']+3*self.stats[show[i]]['std'])
                a = ax.set_ylim(self.stats[show[j]]['q50']-3*self.stats[show[j]]['std'], self.stats[show[j]]['q50']+3*self.stats[show[j]]['std'])
                if i == j:
                    a = ax.text(0.5, 0.92, show[i], fontsize=8, color='red', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

        if file is not None:
            fig.savefig(file)
    
    def draw_random(self, N=10):
        """
        Draw random sets of parameters from the chain
        """
        #ok_walk = self.sampler.acceptance_fraction > min_acceptance
        iwalk = np.cast[int](np.random.rand(N)*self.nwalkers)
        istep = self.nburn + np.cast[int](np.random.rand(N)*(self.nstep-self.nburn))
        draw = self.chain[iwalk, istep, :]
        return draw
        
    def __getitem__(self, param):
        pid = self.param_dict[param]
        return self.chain[:,:,pid]
        
    def contour(self, p1, p2, labels=None, levels=[0.683, 0.955], colors=None, limits=None, bins=20, ax=None, fill=False, **kwargs):
        """
        Plot sigma contours
        """
        import astroML.plotting
        
        if isinstance(p1, str):
            trace1 = self.__getitem__(p1)[:,self.nburn:].flatten()
            pname1 = p1
        else:
            trace1 = p1.flatten()
            pname1 = ''
        
        if isinstance(p2, str):
            trace2 = self.__getitem__(p2)[:,self.nburn:].flatten()
            pname2 = p2
        else:
            trace2 = p2.flatten()
            pname2 = ''
        
        if labels is None:
            labels = [pname1, pname2]
            
        if limits is None:
            limits = [(t.min(), t.max()) for t in [trace1, trace2]]
        
        bins = [np.linspace(limits[i][0], limits[i][1], bins + 1)
                for i in range(2)]
        
        
        H, xbins, ybins = np.histogram2d(trace1, trace2, bins=(bins[0], bins[1]))
        H[H == 0] = 1E-16
        Nsigma = astroML.plotting.mcmc.convert_to_stdev(np.log(H))
        
        if ax is None:
            ax = plt
        
        ax.contour(0.5 * (xbins[1:] + xbins[:-1]),
                   0.5 * (ybins[1:] + ybins[:-1]),
                   Nsigma.T, levels=levels, **kwargs)
        
        if fill:
            if ax is plt:
                col = plt.gca().collections
            else:
                col = ax.collections
            
            n_levels = len(levels)
            if colors is None:
                dc = 1./(n_levels+1)
                colors = ['%.2f' %((i+1)*dc) for i in np.arange(n_levels)]
                print(colors)

            for i in range(n_levels): 
                print(colors[i])
                col[i].set_facecolor(colors[i])
                col[i].set_edgecolor(colors[i])
                col[i].set_zorder(-10-i)
                col[i].set_alpha(0.8)
                
                
        if ax is plt:
            ax.xlabel(labels[0])
            ax.ylabel(labels[1])
            ax.xlim(limits[0])
            ax.ylim(limits[1])
        else:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
