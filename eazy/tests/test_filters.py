import os
import warnings

import numpy as np

from astropy.utils.exceptions import AstropyWarning

from .. import filters, fetch_eazy_photoz, utils

FILTER_RES = None

def test_data_path():
    """
    Data path, download data files if needed
    """
    from .. import filters, fetch_eazy_photoz, utils
    # assert os.path.exists(utils.DATA_PATH)

    if not os.path.exists(
        os.path.join(utils.DATA_PATH, "filters", "FILTER.RES.latest")
    ):
        fetch_eazy_photoz()


def test_array_filter():
    """
    Generate filter from arrays
    """
    wx = np.arange(5400, 5600.0, 1)
    wy = wx * 0.0
    wy[10:-10] = 1

    f1 = filters.FilterDefinition(wave=wx, throughput=wy, name="Tophat 5500")

    assert np.allclose(f1.pivot, 5500, rtol=1.0e-3)
    assert np.allclose(f1.ABVega, 0.016, atol=0.03)
    assert np.allclose(f1.equivwidth, 180)
    assert np.allclose(f1.rectwidth, 180)


def test_pysynphot_filter():
    """
    PySynphot filter bandpass
    """
    try:
        import pysynphot as S
    except:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        v_pysyn = S.ObsBandpass("v")

    v_eazy = filters.FilterDefinition(bp=v_pysyn)

    assert np.allclose(v_pysyn.pivot(), v_eazy.pivot, rtol=0.001)


# def test_data_path():
#     """
#     Data path, download data files if needed
#     """
#     assert os.path.exists(utils.DATA_PATH)
#
#     if not os.path.exists(
#         os.path.join(utils.DATA_PATH, "filters", "FILTER.RES.latest")
#     ):
#         fetch_eazy_photoz()


def test_read_filter_res():
    """
    Read FILTER.RES
    """
    global FILTER_RES

    filter_file = os.path.join(utils.DATA_PATH, "filters/FILTER.RES.latest")
    res = filters.FilterFile(filter_file)

    assert res[155].name.startswith("REST_FRAME/maiz-apellaniz_Johnson_V")
    assert np.allclose(res[155].pivot, 5479.35, rtol=0.001)

    FILTER_RES = res
