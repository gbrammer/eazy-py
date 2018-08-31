from distutils.core import setup
from distutils.extension import Extension

import subprocess

import os
import numpy

### For Cython build
# if False:
#     try:
#         from Cython.Build import cythonize
#         USE_CYTHON = True
#     except ImportError:
#         USE_CYTHON = False
# 
#     if not os.path.exists('grizli/utils_c/interp.pyx'):
#         USE_CYTHON = False
#     
#     if USE_CYTHON:
#         cext = '.pyx'
#     else:
#         cext = '.c'
# 
#     print 'C extension: %s' %(cext)
# 
#     extensions = [
#         Extension("grizli/utils_c/interp", ["grizli/utils_c/interp"+cext],
#             include_dirs = [numpy.get_include()],),
#         
#         # Extension("grizli/utils_c/nmf", ["grizli/utils_c/nmf"+cext],
#         #     include_dirs = [numpy.get_include()],),
#     
#         Extension("grizli/utils_c/disperse", ["grizli/utils_c/disperse"+cext],
#             include_dirs = [numpy.get_include()],),
# 
#     ]
# 
#     if USE_CYTHON:
#         extensions = cythonize(extensions)

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

args = 'git describe --tags'
p = subprocess.Popen(args.split(), stdout=subprocess.PIPE)
version = p.communicate()[0].decode("utf-8").strip()

version = '0.2.0'

version_str = """# git describe --tags
__version__ = "{0}"\n""".format(version)

fp = open('eazy/version.py','w')
fp.write(version_str)
fp.close()
print('Git version: {0}'.format(version))

setup(
    name = "eazy",
    version = version,
    author = "Gabriel Brammer",
    author_email = "gbrammer@gmail.com",
    description = "Pythonic photo-zs modeled after EAZY",
    license = "MIT",
    url = "https://github.com/gbrammer/eazy-py",
    download_url = "https://github.com/gbrammer/eazy-py/tarball/"+version,
    packages=['eazy'],
    # requires=['numpy', 'scipy', 'astropy', 'drizzlepac', 'stwcs'],
    # long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 1 - Planning",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    package_data={'eazy': ['data/*', 'data/filters/*', 'data/templates/*',
                           'data/hdfn_fs99/*', 'data/templates/fsps_full/*',
                           'data/templates/uvista_nmf/*']},
    # scripts=['grizli/scripts/flt_info.sh'],
    # ext_modules = extensions,
)
