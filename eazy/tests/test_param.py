import os

from .. import param
from .. import utils


def test_param_file():
    """
    Read Param file
    """
    # No filename, read default
    pfile1 = param.EazyParam(PARAM_FILE=None)

    # Read from file
    param_file = os.path.join(
        os.path.dirname(param.__file__),
        "data", "zphot.param.default"
    )
    pfile2 = param.EazyParam(PARAM_FILE=param_file)

    assert pfile1["Z_MIN"] == pfile2["Z_MIN"]

    # Set new parameter
    pfile1["XPARAM"] = 1.0
    assert "XPARAM" in pfile1.param_names
    assert pfile1["XPARAM"] == 1
