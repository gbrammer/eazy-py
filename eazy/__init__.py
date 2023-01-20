import os

from . import igm
from . import param
from . import templates
from . import filters
from . import photoz

from .version import __version__

try:
    import dust_attenuation
except ImportError:
    print('Failed to `import dust_attenuation`')
    print('Install my fork with $ pip install ' +
          'git+https://github.com/gbrammer/dust_attenuation.git')

try:
    import dust_extinction
except ImportError:
    print('Failed to `import dust_extinction`')
    print('Install my fork with $ pip install ' +
          'git+https://github.com/gbrammer/dust_extinction.git')


# Hot fix for importing prospector without SPS_HOME variable set
try:
    from prospect.utils.smoothing import smoothspec
except (FileNotFoundError, TypeError):
    if 'SPS_HOME' not in os.environ:
        sps_home = 'xxxxdummyxxxx' #os.path.dirname(__file__)
        print(f'Warning: setting environment variable SPS_HOME={sps_home} '
              'to be able to import prospect.')
        os.environ['SPS_HOME'] = sps_home


def fetch_eazy_photoz():
    """
    If necessary, clone the eazy-photoz repository to get templates and filters
    """
    current_path = os.getcwd()
    
    module_path = os.path.dirname(__file__)
    data_path = os.path.join(module_path, 'data/')
    os.chdir(data_path)

    eazy_photoz = os.path.join(data_path, 'eazy-photoz')
    git_url = 'https://github.com/gbrammer/eazy-photoz.git'

    if not os.path.exists(eazy_photoz):
        os.system(f'git clone {git_url}')
        print(f'cloning {git_url} to {data_path}')

    if not os.path.exists('filters'):
        os.symlink(os.path.join(data_path, 'eazy-photoz','filters'),
                   'filters')

    if not os.path.exists('templates'):
        os.symlink(os.path.join(data_path, 'eazy-photoz','templates'),
                   'templates')

    if not os.path.exists('hdfn_fs99'):
        os.symlink(os.path.join(data_path, 'eazy-photoz','inputs'),
                   'hdfn_fs99')
    
    # Back to working directory
    os.chdir(current_path)


def symlink_eazy_inputs(path='$EAZYCODE', get_hdfn_test_catalog=False, copy=False):
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

    copy : bool
        Copy ``templates`` directory and ``FILTER.RES.latest`` file, rather
        than symlink

    Returns
    -------
    Symbolic links to the `FILTER.RES.latest` file and `templates`
    directory are created in the current working directory (`./`).

    """

    if path.startswith('$'):
        path = os.getenv(path)

    current_path = os.getcwd()

    if path is None:
        # Use the code attached to the repository
        path = os.path.join(os.path.dirname(__file__), 'data/')
        if not os.path.exists(os.path.join(path, 'templates')):
            fetch_eazy_photoz()
    
    os.chdir(current_path)

    if not os.path.exists(path):
        print('Couldn\'t find path {0}'.format(path))
        return False

    # Templates directory
    if os.path.exists('./templates'):
        try:
            os.remove('./templates')
        except PermissionError:
            os.system('rm -rf templates')

    t_path = os.path.join(path, 'templates')
    if copy:
        os.system('cp -R {0} .'.format(t_path))
    else:
        os.symlink(t_path, './templates')

    print('{0} -> {1}'.format(t_path, './templates'))

    # Filter file
    if os.path.exists('./FILTER.RES.latest'):
        os.remove('./FILTER.RES.latest')

    res_path = os.path.join(path, 'filters/FILTER.RES.latest')
    if copy:
        os.system(f'cp {0} .'.format(res_path))
    else:
        os.symlink(res_path, './FILTER.RES.latest')

    print('{0} -> {1}'.format(res_path, './FILTER.RES.latest'))

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
