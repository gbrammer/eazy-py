import os

from . import igm
from . import param
from . import templates
from . import filters
from . import photoz

#__version__ = "0.2.0"
from .version import __version__

def symlink_eazy_inputs(path='EAZYCODE', path_is_env=True):
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

def get_test_catalog(path='EAZYCODE', path_is_env=True):
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
    
    
    
        