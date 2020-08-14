import pytest
import os

import numpy as np

from .. import param

def data_path():
    """
    Data path
    """
    path = os.path.join(os.path.dirname(__file__), '../data/')
    return path

def test_param_file():
    """
    Read Param file
    """
    param_file = os.path.join(data_path(), 'zphot.param.default')
    
    # No filename, read default
    pfile1 = param.EazyParam(PARAM_FILE=None)
    
    # Read filename
    pfile2 = param.EazyParam(PARAM_FILE=param_file)
    
    assert(pfile1['Z_MIN'] == pfile2['Z_MIN'])