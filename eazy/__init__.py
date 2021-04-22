import os

from . import igm
from . import param
from . import templates
from . import filters
from . import photoz

from .version import __version__, __long_version__, __version_hash__

def symlink_eazy_inputs(path='$EAZYCODE', get_hdfn_test_catalog=False):
    """
    Make symbolic links to EAZY inputs
    
    Parameters
    ----------
    path : str
        Full directory path or environment variable pointing to the old eazy
        C-code repository that provides the template and filter files.
        
        If `path.startswith('$')` then treat path as an environment variable.
        
        If you install from the repository that provides the eazy-photozy 
        code as a submodule, then you should be able to run with `path=None` 
        and retrieve the files directly from the repository.  This should
        also work with the `pip` installation.
        
        Another safe way to ensure that the necessary files are avialable is
        to clone the `eazy-photoz` repository and set an environment variable
        to point to it (e.g, 'EAZYCODE'), which you then pass as the `path`
        argument.
    
    Returns
    -------
    Symbolic links to the `FILTER.RES.latest` file and `templates` 
    directory are created in the current working directory (`./`).
    
    """
    
    if path.startswith('$'):
        path = os.getenv(path)
    
    if path is None:
        # Use the code attached to the repository
        path = os.path.join(os.path.dirname(__file__), 'data/')
        
    if not os.path.exists(path):
        print('Couldn\'t find path {0}'.format(path))
        return False
    
    # Templates directory
    if os.path.exists('./templates'):
        os.remove('./templates')
        
    os.symlink(os.path.join(path, 'templates'), './templates')
    print('{0} -> {1}'.format(os.path.join(path, 'templates'), './templates'))
    
    # Filter file
    if os.path.exists('./FILTER.RES.latest'):
        os.remove('./FILTER.RES.latest')
    
    os.symlink(os.path.join(path, 'filters/FILTER.RES.latest'), './FILTER.RES.latest')
    print('{0} -> {1}'.format(os.path.join(path, 'filters/FILTER.RES.latest'), './FILTER.RES.latest'))
    
    if get_hdfn_test_catalog:
        for cat_path in ['inputs', 'hdfn_fs99']:
            parent = os.path.join(path, cat_path, 'hdfn_fs99_eazy.cat')
            translate = os.path.join(path, cat_path, 'zphot.translate')
            if os.path.exists(parent):
                for file in [parent, translate]:
                    os.symlink(file, os.path.basename(file))
                    print('{0} -> {1}'.format(file, os.path.basename(file)))
                    
def get_test_catalog(path=None, path_is_env=True):
    """
    Make symbolic links to EAZY inputs
    
    Parameters
    ----------
    path : str
        Full directory path or environment variable pointing to the old eazy
        C-code repository that provides the template and filter files.
    
    path_is_env : bool
        If True, then `path` is an environment variable pointing to the Eazy
        repository.  If False, then treat as a directory path.
    
    Returns
    -------
    Symbolic links in `./`.
    
    """
    if path_is_env:
        path = os.getenv(path)
    
    if not os.path.exists(path):
        print('Couldn\'t find path {0}'.format(path))
        return False
    
    
    
        