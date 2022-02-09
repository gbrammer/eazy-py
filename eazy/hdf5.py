"""
Tools for saving/recoving state from HDF5 files
"""
from .photoz import PhotoZ

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
            print('cat meta: ', k, pzobj.cat.meta[k])
            grp.attrs[k] = pzobj.cat.meta[k]

        for name in ['flux_columns','err_columns']:
            print(f'Store cat/{name}')
            attr = getattr(pzobj, name)
            dset = grp.create_dataset(name, data=attr) 

        for name in ['f_numbers','fnu','efnu_orig','ok_data','zp',
                     'ext_corr','ext_redden']:
            print(f'Store cat/{name}')
            attr = getattr(pzobj, name)
            dset = grp.create_dataset(name, data=attr)
        
        grp.attrs['MW_EBV'] = pzobj.param['MW_EBV']
        
        grp = f.create_group("fit")
        for name in ['zml','zbest','chi2_fit','coeffs_best']:
            print(f'Store fit/{name}')
            attr = getattr(pzobj, name)
            dset = grp.create_dataset(name, data=attr)
        
        f['fit/zml'].attrs['ZML_WITH_PRIOR'] = pzobj.ZML_WITH_PRIOR
        f['fit/zml'].attrs['ZML_WITH_BETA_PRIOR'] = pzobj.ZML_WITH_BETA_PRIOR
        f['fit/zbest'].attrs['ZPHOT_USER'] = pzobj.ZPHOT_USER
        
        if include_fit_coeffs | pzobj.ZML_WITH_BETA_PRIOR:
            name = 'fit_coeffs'
            print(f'Store fit/{name}')
            attr = getattr(pzobj, name)
            dset = grp.create_dataset(name, data=attr)
            
        # Parameters
        for k in pzobj.param.params:
            grp.attrs[k] = pzobj.param.params[k]
            
        dset = grp.create_dataset('tempfilt', data=pzobj.tempfilt.tempfilt)
        func = pzobj.tempfilt.interpolator_function
        dset.attrs['interpolator_function'] = func.__name__
        
        # Templates
        if include_templates:
            grp = f.create_group("templates")
            grp.attrs['NTEMP'] = pzobj.NTEMP
            for i, templ in enumerate(pzobj.templates):
                grp.attrs[f'TEMPL{i:03d}'] = templ.name
                print(f'templates/{templ.name}')
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
        cat, trans = PhotoZ._csv_from_arrays(f['cat/id'][:],
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
        pzobj = PhotoZ(param_file=None, translate_file=trans,
                                   zeropoint_file=None, 
                                   params=params, load_prior=True, 
                                   load_products=False, 
                                   tempfilt_data=f['fit/tempfilt'][:])
        
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