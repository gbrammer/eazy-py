"""
Tools for saving/recoving state from HDF5 files
"""
import numpy as np

from . import photoz
from . import utils
from . import templates as templates_code

def write_hdf5(pzobj, h5file='test.hdf5', include_fit_coeffs=False, include_templates=True):
    """
    Write self-contained HDF5 file
    
    Parameters
    ----------
    pzobj : `~eazy.photoz.PhotoZ`
        Original code object with computed redshifts, coeffs, etc.
    
    h5file : str
        HDF5 filename
    
    include_fit_coeffs : bool
        Inlude full `fit_coeffs` array with ``(NOBJ, NZ, NTEMP)`` fit
        coefficients.  This can make the file very large, and it's really only 
        needed if you want to use `~eazy.photoz.PhotoZ.prior_beta`.
        
    include_templates : bool
        Include template arrays
        
    """
    import h5py
    with h5py.File(h5file,'w') as f:

        grp = f.create_group("cat")
        dset = grp.create_dataset('id', data=pzobj.OBJID)
        dset = grp.create_dataset('ra', data=pzobj.RA)
        dset = grp.create_dataset('dec', data=pzobj.DEC)
        dset = grp.create_dataset('z_spec', data=pzobj.ZSPEC)
        
        for k in pzobj.cat.meta:
            print('h5: cat meta: ', k, pzobj.cat.meta[k])
            grp.attrs[k] = pzobj.cat.meta[k]

        for name in ['flux_columns','err_columns']:
            print(f'h5: cat/{name}')
            attr = getattr(pzobj, name)
            dset = grp.create_dataset(name, data=attr) 

        for name in ['f_numbers','fnu','efnu_orig','ok_data','zp',
                     'ext_corr','ext_redden','pivot']:
            print(f'h5: cat/{name}')
            attr = getattr(pzobj, name)
            dset = grp.create_dataset(name, data=attr)
        
        grp.attrs['MW_EBV'] = pzobj.param['MW_EBV']
        
        grp = f.create_group("fit")
        for name in ['zml','zbest','chi2_fit','coeffs_best']:
            print(f'h5: fit/{name}')
            attr = getattr(pzobj, name)
            dset = grp.create_dataset(name, data=attr)
        
        dset = grp.create_dataset('tef_x', data=pzobj.TEF.te_x)
        dset = grp.create_dataset('tef_y', data=pzobj.TEF.te_y)
        
        f['fit/zml'].attrs['ZML_WITH_PRIOR'] = pzobj.ZML_WITH_PRIOR
        f['fit/zml'].attrs['ZML_WITH_BETA_PRIOR'] = pzobj.ZML_WITH_BETA_PRIOR
        f['fit/zbest'].attrs['ZPHOT_USER'] = pzobj.ZPHOT_USER
        
        if include_fit_coeffs | pzobj.ZML_WITH_BETA_PRIOR:
            name = 'fit_coeffs'
            print(f'h5: fit/{name}')
            attr = getattr(pzobj, name)
            dset = grp.create_dataset(name, data=attr)
            
        # Parameters
        for k in pzobj.param.params:
            grp.attrs[k] = pzobj.param.params[k]
            
        dset = grp.create_dataset('tempfilt', data=pzobj.tempfilt.tempfilt)
        dset = grp.create_dataset('tempfilt_scale', data=pzobj.tempfilt.scale)
        func = pzobj.tempfilt.interpolator_function
        dset.attrs['interpolator_function'] = func.__name__
        
        # Templates
        if include_templates:
            grp = f.create_group("templates")
            grp.attrs['NTEMP'] = pzobj.NTEMP
            for i, templ in enumerate(pzobj.templates):
                grp.attrs[f'TEMPL{i:03d}'] = templ.name
                print(f'h5 templates/{templ.name}')
                dset = grp.create_dataset(f'wave {templ.name}', 
                                    data=templ.wave.astype(pzobj.ARRAY_DTYPE))
                dset = grp.create_dataset(f'flux {templ.name}', 
                                    data=templ.flux.astype(pzobj.ARRAY_DTYPE))
                dset = grp.create_dataset(f'z {templ.name}', 
                                          data=templ.redshifts)


def cat_from_hdf5(h5file):
    """
    Parameters
    ----------
    h5file : str
        HDF5 filename
        
    Returns 
    -------
    cat : `~astropy.table.Table`
        Catalog table generated from HDF5 data
        
    trans : `eazy.param.TranslateFile`
        TranslateFile object
        
    """
    import h5py
    with h5py.File(h5file,'r') as f:
        cat, trans = photoz.PhotoZ._csv_from_arrays(f['cat/id'][:],
                                      f['cat/ra'][:], f['cat/dec'][:], 
                                      f['cat/z_spec'][:], 
                                      f['cat/fnu'], f['cat/efnu_orig'],
                                      f['cat/ok_data'],
                                      f['cat/flux_columns'].asstr(), 
                                      f['cat/err_columns'].asstr(),
                                      f['cat/zp'][:]**0,
                                      f['cat/f_numbers'][:])
    
    return cat, trans


def params_from_hdf5(h5file):
    """
    Read full parameters from HDF5 file
    
    Parameters
    ----------
    h5file : str
        HDF5 filename
    
    Returns
    -------
    params : dict
    """
    from collections import OrderedDict
    import h5py
    
    params = OrderedDict()
    
    with h5py.File(h5file,'r') as f:
        dset = f['fit']
        for k in dset.attrs:
            params[k] = dset.attrs[k]
    
    return params


def templates_from_hdf5(h5file):
    """
    Read list of templates
    """
    import h5py
    templates = []
    with h5py.File(h5file,'r') as f:
        NTEMP = f['templates'].attrs['NTEMP']
        for i in range(NTEMP):
            name = f['templates'].attrs[f'TEMPL{i:03d}']
            
            wave = f[f'templates/wave {name}'][:]
            flux = f[f'templates/flux {name}'][:]
            redshifts = f[f'templates/z {name}'][:]

            templ_i = templates_code.Template(arrays=(wave, flux), 
                                              name=name, 
                                              redshifts=redshifts)
            templates.append(templ_i)
    
    return templates


def initialize_from_hdf5(h5file='test.hdf5'):
    """
    Initialize a `~eazy.photoz.PhotoZ` object from HDF5 data
    
    Parameters
    ----------
    h5file : str
        HDF5 filename from `eazy.hdf5.
    
    Returns
    -------
    pzobj : '~eazy.photoz.PhotoZ'
    
    """
    import h5py
    
    # Parameter dictionary
    params = params_from_hdf5(h5file)

    # Generate a catalog table from H5 data
    cat, trans = cat_from_hdf5(h5file)
        
    # Put catalog in CATALOG_FILE parameter
    params['CATALOG_FILE'] = cat
    
    with h5py.File(h5file, 'r') as f:
        pzobj = photoz.PhotoZ(param_file=None, translate_file=trans,
                                   zeropoint_file=None, 
                                   params=params, load_prior=True, 
                                   load_products=False, 
                                   tempfilt_data=f['fit/tempfilt'][:])
        
        pzobj.tempfilt.scale = f['fit/tempfilt_scale'][:]
        pzobj.chi2_fit = f['fit/chi2_fit'][:]
        pzobj.zp = f['cat/zp'][:]
        
        if 'fit/fit_coeffs' in f:
            pzobj.fit_coeffs = f['fit/fit_coeffs'][:]
        
        pzobj.compute_lnp(prior=f['fit/zml'].attrs['ZML_WITH_PRIOR'], 
                         beta_prior=f['fit/zml'].attrs['ZML_WITH_BETA_PRIOR'])
        
        pzobj.evaluate_zml(prior=f['fit/zml'].attrs['ZML_WITH_PRIOR'], 
                         beta_prior=f['fit/zml'].attrs['ZML_WITH_BETA_PRIOR'])
        
        if f['fit/zbest'].attrs['ZPHOT_USER']:
            pzobj.fit_at_zbest(zbest=f['fit/zbest'], 
                               prior=f['fit/zml'].attrs['ZML_WITH_PRIOR'], 
                         beta_prior=f['fit/zml'].attrs['ZML_WITH_BETA_PRIOR'])
        else:
            pzobj.fit_at_zbest(zbest=None, 
                               prior=f['fit/zml'].attrs['ZML_WITH_PRIOR'], 
                         beta_prior=f['fit/zml'].attrs['ZML_WITH_BETA_PRIOR'])
            
    return pzobj


class Viewer(object):
    def __init__(self, h5file):
        """
        Tool to replicate functionality of `PhotoZ.show_fit` but with 
        data read from a stored HDF5 file rather than a "live" object
        """
        import h5py
        from astropy.cosmology import LambdaCDM
        
        self.h5file = h5file
        
        self.param = params_from_hdf5(h5file)
        
        photoz.PhotoZ.set_zgrid(self)
        self.NZ = len(self.zgrid)
        
        self.templates = templates_from_hdf5(h5file)
        self.NTEMP = len(self.templates)
        
        self.set_attrs_from_hdf5()
        
        self.set_template_error()
        
        self.cosmology = LambdaCDM(H0=self.param['H0'], 
                                   Om0=self.param['OMEGA_M'], 
                                   Ode0=self.param['OMEGA_L'], 
                                   Tcmb0=2.725, Ob0=0.048)
        
        self.set_tempfilt()


    def get_catalog(self):
        """
        """
        cat, trans = cat_from_hdf5(self.h5file)
        return cat


    def set_tempfilt(self):
        """
        """
        import h5py
        with h5py.File(self.h5file, 'r') as f:
            self.tempfilt = photoz.TemplateGrid(self.zgrid, self.templates, 
                                    RES=self.param['FILTERS_RES'], 
                                    f_numbers=self.f_numbers, 
                                    add_igm=self.param['IGM_SCALE_TAU'], 
                                galactic_ebv=self.MW_EBV, 
                                Eb=self.param['SCALE_2175_BUMP'], 
                                n_proc=1, cosmology=self.cosmology, 
                                array_dtype=self.ARRAY_DTYPE, 
                                tempfilt_data=f['fit/tempfilt'][:])
        
            self.tempfilt.scale = f['fit/tempfilt_scale'][:]


    def set_template_error(self):
        """
        Set the Template Error Function 
        
        Parameters
        ----------
        TEF : `eazy.templates.TemplateError` or None
            If not specified, read from `params['TEMP_ERR_FILE']` and scale
            by `params['TEMP_ERR_A2']`.
        
        compute_tef_lnp : bool
            Compute the likelihood normalization correction for the 
            `~eazy.templates.TemplateError` function.
        
        Returns
        -------
        Sets `TEF`, `TEFgrid` and `compute_tef_lnp` attributes
            
        """
        import h5py
        with h5py.File(self.h5file, 'r') as f:
            arrays = (f['fit/tef_x'][:], f['fit/tef_y'][:])
            
        self.TEF = templates_code.TemplateError(arrays=arrays, 
                                             filter_wavelengths=self.pivot, 
                                             scale=self.param['TEMP_ERR_A2'])


    def set_attrs_from_hdf5(self):
        """
        """
        import h5py
        with h5py.File(self.h5file,'r') as f:
            self.NOBJ, self.NFILT = f['cat/fnu'].shape
            self.pivot = f['cat/pivot'][:]
            self.OBJID = f['cat/id'][:]
            self.zp = f['cat/zp'][:]
            self.ext_corr = f['cat/ext_corr'][:]
            self.ext_redden = f['cat/ext_redden'][:]
            self.f_numbers = f['cat/f_numbers'][:]
            self.flux_columns = f['cat/flux_columns'].asstr()[:]
            self.err_columns = f['cat/err_columns'].asstr()[:]
            
        self.idx = np.arange(self.NOBJ, dtype=int)


    def get_object_data(self, ix):
        """
        """
        import h5py
        with h5py.File(self.h5file,'r') as f:
            fnu_i = f['cat/fnu'][ix,:]
            efnu_orig = f['cat/efnu_orig'][ix,:]
            ra_i = f['cat/ra'][ix]
            dec_i = f['cat/dec'][ix]
            z_i = f['fit/zbest'][ix]
            chi2_i = f['fit/chi2_fit'][ix,:]
            zspec_i = f['cat/z_spec'][ix]
            ok_i = f['cat/ok_data'][ix,:]
        
        efnu_i = np.sqrt(efnu_orig**2 + 
                         (self.param['SYS_ERR']*np.maximum(fnu_i, 0.))**2)
        
        return z_i, fnu_i, efnu_i, ra_i, dec_i, chi2_i, zspec_i, ok_i


    def get_lnp(self, ix):
        """
        """
        import h5py
        with h5py.File(self.h5file,'r') as f:
            chi2_i = f['fit/chi2_fit'][ix,:]
            
        return -0.5*(chi2_i - np.nanmin(chi2_i))


    def show_fit(self, id, **kwargs):
        """
        """
        _ = photoz.PhotoZ.show_fit(self, id, **kwargs)
        return _


    def show_fit_plotly(self, id, **kwargs):
        """
        """
        _ = photoz.PhotoZ.show_fit_plotly(self, id, **kwargs)
        return _


    @property 
    def to_flam(self):
        """
        Conversion factor to :math:`10^{-19} erg/s/cm^2/Å`
        """
        to_flam = 10**(-0.4*(self.param['PRIOR_ABZP']+48.6))
        to_flam *= utils.CLIGHT*1.e10/1.e-19/self.pivot**2/self.ext_corr
        return to_flam


    @property 
    def to_uJy(self):
        """
        Conversion of observed fluxes to `~astropy.units.microJansky`
        """
        return 10**(-0.4*(self.param['PRIOR_ABZP']-23.9))


    @property 
    def lc(self):
        """
        Filter pivot wavelengths (deprecated, use `pivot`)
        """     
        return self.pivot


    @property 
    def ARRAY_DTYPE(self):
        """
        Array data type from `ARRAY_NBITS` parameter
        """
        if 'ARRAY_NBITS' in self.param:
            if self.param['ARRAY_NBITS'] == 64:
                ARRAY_DTYPE = np.float64
            else:
                ARRAY_DTYPE = np.float32
        else:
            ARRAY_DTYPE = np.float32
        
        return ARRAY_DTYPE


    @property 
    def MW_EBV(self):    
        """
        Galactic extinction E(B-V)
        """
        if 'MW_EBV' not in self.param:
            return 0. # 0.0354 # MACS0416
        else:
            return self.param['MW_EBV']
    