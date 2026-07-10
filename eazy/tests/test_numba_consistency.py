"""
Consistency tests: the numba JIT fitting path (`eazy.numba_utils`) must
reproduce the pure-python implementations in `eazy.photoz`.

Deterministic products (chi2, coefficients, model fluxes, zml/zbest) are
compared at (near) machine precision.  Stochastic products (coefficient
draws, rest-frame flux percentiles) are compared statistically, since the
python path samples with `numpy.random` while the numba path either samples
with numba's own RNG or evaluates the sampled quantity analytically.
"""
import os
import warnings

import numpy as np
import pytest

np.seterr(all="ignore")

numba = pytest.importorskip("numba")

from .. import utils
from .. import filters
from .. import photoz
from .. import templates as templates_module

HAS_DATA = os.path.exists(
    os.path.join(utils.DATA_PATH, "filters/FILTER.RES.latest")
) and os.path.exists(
    os.path.join(utils.DATA_PATH, "templates/fsps_full/fsps_QSF_12_v3.param")
)

pytestmark = pytest.mark.skipif(
    not HAS_DATA, reason="eazy-photoz data files not found"
)

NOBJ = 60
UBVJ_TEST = [153, 154, 155, 161]


def _build_photoz(tmpdir, nbits=64, nobj=NOBJ, seed=42):
    """
    Synthetic catalog from mixes of the fsps_QSF_12_v3 templates, with
    edge-case objects (missing photometry, negative fluxes, zero errors).
    """
    rng = np.random.default_rng(seed)

    res = filters.FilterFile(
        os.path.join(utils.DATA_PATH, "filters/FILTER.RES.latest")
    )
    f_numbers = [209, 211, 214, 217, 202, 203, 205, 269, 18, 19]
    f_list = [res[f] for f in f_numbers]
    NFILT = len(f_list)

    templ_list = templates_module.read_templates_file(
        os.path.join(utils.DATA_PATH,
                     "templates/fsps_full/fsps_QSF_12_v3.param")
    )
    NTEMP = len(templ_list)

    z_true = rng.uniform(0.2, 4.5, nobj)
    fnu = np.zeros((nobj, NFILT))
    for i in range(nobj):
        it1, it2 = rng.choice(NTEMP, 2, replace=False)
        w = rng.uniform(0, 1)
        f1 = np.array(templ_list[it1].integrate_filter(f_list, z=z_true[i]))
        f2 = np.array(templ_list[it2].integrate_filter(f_list, z=z_true[i]))
        f1 /= np.maximum(f1.max(), 1e-30)
        f2 /= np.maximum(f2.max(), 1e-30)
        sed = w * f1 + (1 - w) * f2
        sed *= 10 ** rng.uniform(-1, 1) / np.maximum(sed[6], 1e-30)
        fnu[i, :] = sed

    lc = np.array([f.pivot for f in f_list])
    SN = rng.uniform(3, 25, nobj)
    efnu = (fnu[:, 6][:, None] / SN[:, None]) * (lc[None, :] / lc[6]) ** 1.5
    efnu = np.maximum(efnu, 1e-4)
    fnu = fnu + rng.normal(size=fnu.shape) * efnu

    # Edge cases
    fnu[0, :] = -99.0; efnu[0, :] = -99.0        # all missing
    fnu[1, 1:] = -99.0; efnu[1, 1:] = -99.0      # 1 valid band
    fnu[2, 2:] = -99.0; efnu[2, 2:] = -99.0      # 2 valid bands
    fnu[3, :] = rng.normal(0, 1e-3, NFILT)       # pure noise, mostly negative
    efnu[3, :] = 1e-3
    fnu[4, ::2] = -99.0; efnu[4, ::2] = -99.0    # alternating missing
    efnu[5, 3] = 0.0                             # zero uncertainty band

    tab = photoz.Table()
    tab["id"] = np.arange(nobj, dtype=int) + 1
    tab["z_spec"] = z_true
    tab["ra"] = 150.1
    tab["dec"] = 2.5

    translate_file = os.path.join(str(tmpdir), "zphot.translate.test")
    with open(translate_file, "w") as fp:
        for i, fnum in enumerate(f_numbers):
            tab[f"f_{fnum}"] = fnu[:, i]
            tab[f"e_{fnum}"] = efnu[:, i]
            fp.write(f"f_{fnum} F{fnum}\n")
            fp.write(f"e_{fnum} E{fnum}\n")

    cat_file = os.path.join(str(tmpdir), "eazy_numba_test.cat")
    tab.write(cat_file, overwrite=True, format="ascii.commented_header")

    params = {
        "CATALOG_FILE": cat_file,
        "MAIN_OUTPUT_FILE": os.path.join(str(tmpdir), "eazy_numba_test"),
        "MW_EBV": 0.0,
        "Z_STEP": 0.05,
        "Z_MIN": 0.1,
        "Z_MAX": 5.0,
        "SYS_ERR": 0.02,
        "PRIOR_ABZP": 23.9,
        "PRIOR_FILTER": 205,
        "PRIOR_FILE": os.path.join(utils.DATA_PATH,
                                   "templates/prior_F160W_TAO.dat"),
        "FILTERS_RES": os.path.join(utils.DATA_PATH,
                                    "filters/FILTER.RES.latest"),
        "TEMPLATES_FILE": os.path.join(
            utils.DATA_PATH, "templates/fsps_full/fsps_QSF_12_v3.param"),
        "VERBOSITY": 0,
        "FIX_ZSPEC": False,
        "ARRAY_NBITS": nbits,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ez = photoz.PhotoZ(param_file=None, translate_file=translate_file,
                           zeropoint_file=None, params=params,
                           load_prior=True, load_products=False)
    return ez


@pytest.fixture(scope="module")
def ez(tmp_path_factory):
    return _build_photoz(tmp_path_factory.mktemp("numba_consistency"))


@pytest.fixture(scope="module")
def fit_results(ez):
    """
    Run fit_catalog with both branches; return the grids.
    """
    ez.fit_catalog(use_numba=False, n_proc=0, get_best_fit=False,
                   verbose=False)
    chi2_py = np.array(ez.chi2_fit, dtype=np.float64)
    coeffs_py = np.array(ez.fit_coeffs, dtype=np.float64)

    ez.fit_catalog(use_numba=True, get_best_fit=False, verbose=False)
    chi2_nb = np.array(ez.chi2_fit, dtype=np.float64)
    coeffs_nb = np.array(ez.fit_coeffs, dtype=np.float64)

    return chi2_py, coeffs_py, chi2_nb, coeffs_nb


def test_nnls_against_scipy():
    """
    The Lawson-Hanson NNLS kernel must match `scipy.optimize.nnls`.
    """
    from scipy.optimize import nnls as scipy_nnls
    from ..numba_utils import _nnls_numba

    rng = np.random.default_rng(1)
    for trial in range(50):
        m = rng.integers(5, 40)
        n = rng.integers(2, 15)
        A = rng.normal(size=(m, n)) ** 2
        x_true = np.where(rng.uniform(size=n) > 0.5,
                          rng.uniform(0.1, 10, n), 0.0)
        b = A @ x_true + rng.normal(0, 0.01, m)

        x_sp, rnorm_sp = scipy_nnls(A, b)
        x_nb, rnorm_nb = _nnls_numba(A, b)

        assert np.all(x_nb > -1e-12), f"trial {trial}"

        # The numba solution must never be worse than scipy's.  (It can be
        # slightly *better*: scipy >= 1.12 occasionally returns a mildly
        # suboptimal point on nearly collinear problems.)  When the two
        # objectives tie, the fitted vector is unique and must match, even
        # if degenerate coefficient splits differ.
        r_sp = np.linalg.norm(A @ x_sp - b)
        r_nb = np.linalg.norm(A @ x_nb - b)
        assert r_nb <= r_sp * (1 + 1e-8) + 1e-12, f"trial {trial}"

        if np.isclose(r_sp, r_nb, rtol=1e-8, atol=1e-12):
            fit_scale = max(np.abs(A @ x_sp).max(), 1)
            assert np.allclose(A @ x_sp, A @ x_nb, rtol=1e-5,
                               atol=1e-8 * fit_scale), f"trial {trial}"


def test_template_lsq_kernel(ez):
    """
    Direct kernel comparison: template_lsq_numba vs photoz.template_lsq
    for every object at several redshifts, float64.
    """
    from ..numba_utils import template_lsq_numba

    fnu_corr = (ez.fnu * ez.ext_redden * ez.zp).astype(np.float64)
    efnu_corr = (ez.efnu * ez.ext_redden * ez.zp).astype(np.float64)
    missing = ez.fnu < ez.param["NOT_OBS_THRESHOLD"]
    efnu_corr[missing] = ez.param["NOT_OBS_THRESHOLD"] - 9.0

    zp = np.asarray(ez.zp, dtype=np.float64)

    n_fit = 0
    n_coeff_tie = 0
    for z in [0.3, 1.1, 2.4, 4.2]:
        A = ez.tempfilt(z).astype(np.float64)
        TEFz = ez.TEF(z).astype(np.float64)

        for iobj in range(ez.NOBJ):
            chi2_py, coeffs_py, fmodel_py, _ = photoz.template_lsq(
                fnu_corr[iobj], efnu_corr[iobj], A, TEFz, zp, 0,
                "nnls", True, 1.0)
            chi2_nb, coeffs_nb, fmodel_nb = template_lsq_numba(
                fnu_corr[iobj], efnu_corr[iobj], A, TEFz, zp, True, 1.0)

            if not np.isfinite(chi2_py):
                assert not np.isfinite(chi2_nb), \
                    f"finite-pattern mismatch obj {iobj} z={z}"
                continue

            n_fit += 1

            # chi2 and the fitted fluxes on the *valid* bands are the
            # unique outputs of the convex NNLS problem: always equal
            assert np.isclose(chi2_py, chi2_nb, rtol=1e-8), \
                f"chi2 mismatch obj {iobj} z={z}"

            ok_band = ((efnu_corr[iobj] / zp > 0) &
                       np.isfinite(fnu_corr[iobj]))
            fscale = max(np.abs(fmodel_py[ok_band]).max(), 1e-30)
            assert np.allclose(fmodel_py[ok_band], fmodel_nb[ok_band],
                               rtol=1e-5, atol=1e-8 * fscale), \
                f"fmodel mismatch obj {iobj} z={z}"

            # Individual coefficients can differ when templates are
            # degenerate (equal-chi2 ties, established above); such ties
            # must be rare
            scale = max(np.abs(coeffs_py).max(), 1e-30)
            if not np.allclose(coeffs_py, coeffs_nb, rtol=1e-5,
                               atol=1e-8 * scale):
                n_coeff_tie += 1

    assert n_fit > 100
    assert n_coeff_tie / n_fit < 0.05, \
        f"coefficient ties in {n_coeff_tie}/{n_fit} fits"


def test_template_lsq_nan_fluxes(ez):
    """
    NaN photometry must be masked identically (guards against fastmath
    folding away the np.isfinite checks).
    """
    from ..numba_utils import template_lsq_numba

    z = 1.5
    A = ez.tempfilt(z).astype(np.float64)
    TEFz = ez.TEF(z).astype(np.float64)
    zp = np.asarray(ez.zp, dtype=np.float64)

    fnu_i = (ez.fnu * ez.ext_redden * ez.zp)[10].astype(np.float64)
    efnu_i = (ez.efnu * ez.ext_redden * ez.zp)[10].astype(np.float64)
    fnu_i[2] = np.nan
    efnu_i[5] = np.nan

    chi2_py, coeffs_py, _, _ = photoz.template_lsq(
        fnu_i, efnu_i, A, TEFz, zp, 0, "nnls", True, 1.0)
    chi2_nb, coeffs_nb, _ = template_lsq_numba(
        fnu_i, efnu_i, A, TEFz, zp, True, 1.0)

    assert np.isfinite(chi2_nb)
    assert np.isclose(chi2_py, chi2_nb, rtol=1e-8)
    scale = max(np.abs(coeffs_py).max(), 1e-30)
    assert np.allclose(coeffs_py, coeffs_nb, rtol=1e-6, atol=1e-8 * scale)


def test_covariance_kernel(ez):
    """
    The covariance from template_lsq_numba_covar must match the python
    construction: safe_invert of the active-set Hessian of the scaled
    design matrix, unscaled by Anorm.
    """
    from ..numba_utils import template_lsq_numba_covar

    z = 1.5
    A = ez.tempfilt(z).astype(np.float64)
    TEFz = ez.TEF(z).astype(np.float64)
    zp = np.asarray(ez.zp, dtype=np.float64)

    fnu_corr = (ez.fnu * ez.ext_redden * ez.zp).astype(np.float64)
    efnu_corr = (ez.efnu * ez.ext_redden * ez.zp).astype(np.float64)
    missing = ez.fnu < ez.param["NOT_OBS_THRESHOLD"]
    efnu_corr[missing] = ez.param["NOT_OBS_THRESHOLD"] - 9.0

    checked = 0
    for iobj in range(6, ez.NOBJ):
        fnu_i, efnu_i = fnu_corr[iobj], efnu_corr[iobj]
        chi2, coeffs, fmodel, covar = template_lsq_numba_covar(
            fnu_i, efnu_i, A, TEFz, zp, True, 1.0)

        if not np.isfinite(chi2):
            continue

        active = coeffs > 0
        if active.sum() == 0:
            continue

        # Python-path covariance of the unscaled coefficients
        ok_band = (efnu_i / zp > 0) & np.isfinite(fnu_i) & np.isfinite(efnu_i)
        var = efnu_i ** 2 + (TEFz * np.maximum(fnu_i, 0)) ** 2
        rms = np.sqrt(var)
        Anorm = np.linalg.norm((A / rms)[:, ok_band], axis=1, ord=2)
        Anorm[Anorm <= 0] = 1.0
        Ax = ((A.T / Anorm).T / rms).T[ok_band, :]
        mat = Ax[:, active].T @ Ax[:, active]
        covar_py_scaled = utils.safe_invert(mat)
        norm_outer = np.outer(Anorm[active], Anorm[active])
        covar_py = covar_py_scaled / norm_outer

        covar_nb = covar[np.ix_(np.where(active)[0], np.where(active)[0])]

        if np.all(np.isfinite(covar_py)):
            scale = np.abs(covar_py).max()
            assert np.allclose(covar_py, covar_nb, rtol=1e-5,
                               atol=1e-8 * scale), f"covar mismatch {iobj}"
        else:
            assert not np.all(np.isfinite(covar_nb))
        checked += 1

    assert checked > 10


def test_fit_catalog_grids(fit_results):
    """
    chi2 and coefficient grids from fit_catalog must agree, including the
    treatment of objects with < 2 valid bands (chi2 stays 0).
    """
    chi2_py, coeffs_py, chi2_nb, coeffs_nb = fit_results

    assert np.all(np.isfinite(chi2_py) == np.isfinite(chi2_nb))

    finite = np.isfinite(chi2_py)
    assert np.allclose(chi2_py[finite], chi2_nb[finite], rtol=1e-4)

    # Coefficients agree except for rare equal-chi2 degenerate ties
    scale = np.abs(coeffs_py).max()
    mismatch = ~np.isclose(coeffs_py, coeffs_nb, rtol=1e-3,
                           atol=1e-6 * scale).all(axis=-1)
    assert mismatch.mean() < 0.02, f"coeff tie fraction {mismatch.mean()}"
    # ...and at every tie the chi2 must still match (equally optimal)
    assert np.allclose(chi2_py[mismatch & finite], chi2_nb[mismatch & finite],
                       rtol=1e-4)


def test_fit_catalog_zml(ez, fit_results):
    """
    Maximum-likelihood redshifts must be identical.
    """
    chi2_py, coeffs_py, chi2_nb, coeffs_nb = fit_results

    ez.chi2_fit = chi2_py.astype(ez.ARRAY_DTYPE)
    ez.fit_coeffs = coeffs_py.astype(ez.ARRAY_DTYPE)
    ez.evaluate_zml(prior=False, beta_prior=False)
    zml_py = np.array(ez.zml)

    ez.chi2_fit = chi2_nb.astype(ez.ARRAY_DTYPE)
    ez.fit_coeffs = coeffs_nb.astype(ez.ARRAY_DTYPE)
    ez.evaluate_zml(prior=False, beta_prior=False)
    zml_nb = np.array(ez.zml)

    assert np.allclose(zml_py, zml_nb, atol=1e-4)


@pytest.fixture(scope="module")
def zbest_state(ez, fit_results):
    """
    Common state for the fit_at_zbest / rest-frame tests: python chi2 grid
    loaded so both branches see identical zbest.
    """
    chi2_py, coeffs_py, _, _ = fit_results
    ez.chi2_fit = chi2_py.astype(ez.ARRAY_DTYPE)
    ez.fit_coeffs = coeffs_py.astype(ez.ARRAY_DTYPE)
    ez.evaluate_zml(prior=False, beta_prior=False)
    return ez


def test_fit_at_zbest(zbest_state):
    """
    Deterministic outputs of fit_at_zbest agree; stochastic outputs agree
    statistically.
    """
    ez = zbest_state

    ez.fit_at_zbest(get_err=True, use_numba=False, n_proc=0)
    py = dict(coeffs=ez.coeffs_best.copy(), fmodel=ez.fmodel.copy(),
              efmodel=ez.efmodel.copy(), chi2=ez.chi2_best.copy(),
              draws=ez.coeffs_draws.copy(), zbest=ez.zbest.copy())

    ez.fit_at_zbest(get_err=True, use_numba=True, n_proc=0)
    nb = dict(coeffs=ez.coeffs_best.copy(), fmodel=ez.fmodel.copy(),
              efmodel=ez.efmodel.copy(), chi2=ez.chi2_best.copy(),
              draws=ez.coeffs_draws.copy(), zbest=ez.zbest.copy())

    assert np.allclose(py["zbest"], nb["zbest"])

    fin = np.isfinite(py["chi2"])
    assert np.all(fin == np.isfinite(nb["chi2"]))
    assert np.allclose(py["chi2"][fin], nb["chi2"][fin], rtol=1e-4)

    # Coefficients (and model fluxes on unconstrained bands) can differ
    # for rare equal-chi2 degenerate ties; require ties to be rare and
    # equally optimal
    scale = np.abs(py["coeffs"]).max()
    tie = ~np.isclose(py["coeffs"], nb["coeffs"], rtol=1e-3,
                      atol=1e-6 * scale).all(axis=-1)
    assert tie.mean() < 0.05, f"coeff tie fraction {tie.mean()}"
    assert np.allclose(py["chi2"][tie & fin], nb["chi2"][tie & fin],
                       rtol=1e-4)

    # Model fluxes on observed bands agree for the non-tied objects
    fscale = np.abs(py["fmodel"]).max()
    ok = ~tie
    assert np.allclose(py["fmodel"][ok], nb["fmodel"][ok], rtol=1e-3,
                       atol=1e-6 * fscale)

    # Stochastic: draw means/stds per object+template within sampling noise
    ndraws = py["draws"].shape[1]
    std_py = py["draws"].std(axis=1)
    std_nb = nb["draws"].std(axis=1)
    # std-of-std ~ std/sqrt(2 ndraws); allow 6 sigma
    tol = 6.0 / np.sqrt(2 * ndraws)
    both = (std_py > 0) & (std_nb > 0)
    assert both.sum() > 0
    ratio = std_nb[both] / std_py[both]
    assert np.median(np.abs(np.log(ratio))) < tol, \
        f"draw std ratio median {np.median(ratio)}"

    # Same objects/templates have draws in both paths (except at ties,
    # where the equally-optimal active sets differ)
    assert np.all((std_py > 0)[~tie] == (std_nb > 0)[~tie])
    assert np.all((std_py > 0).any(axis=1) == (std_nb > 0).any(axis=1))

    # efmodel: statistical agreement where both defined
    epy, enb = py["efmodel"], nb["efmodel"]
    both = (epy > 0) & (enb > 0)
    assert np.all((epy > 0)[~tie] == (enb > 0)[~tie])
    if both.sum():
        ratio = enb[both] / epy[both]
        assert np.median(np.abs(np.log(ratio))) < tol


def test_rest_frame_fluxes_refit(zbest_state):
    """
    Rest-frame UBVJ from the re-fitting path.  The numba branch evaluates
    the percentiles analytically, the python branch samples them, so the
    comparison allows for python's sampling noise.
    """
    ez = zbest_state
    NDRAWS = 2000

    _, _, fr_py = ez.rest_frame_fluxes(f_numbers=UBVJ_TEST, simple=False,
                                       use_numba=False, ndraws=NDRAWS,
                                       n_proc=0, verbose=0)
    _, _, fr_nb = ez.rest_frame_fluxes(f_numbers=UBVJ_TEST, simple=False,
                                       use_numba=True, n_proc=0, verbose=0)

    fitted = fr_py[:, 0, 2] > ez.param["NOT_OBS_THRESHOLD"]
    assert np.all(fitted == (fr_nb[:, 0, 2] > ez.param["NOT_OBS_THRESHOLD"]))

    sigma_nb = (fr_nb[:, :, 3] - fr_nb[:, :, 1]) / 2.0
    sigma_py = (fr_py[:, :, 3] - fr_py[:, :, 1]) / 2.0

    # Median flux: python p50 has sampling error ~ 1.25 sigma / sqrt(N)
    d50 = np.abs(fr_py[:, :, 2] - fr_nb[:, :, 2])[fitted]
    tol50 = (6 * 1.25 * sigma_nb / np.sqrt(NDRAWS))[fitted]
    tol50 += 1e-4 * np.abs(fr_nb[:, :, 2])[fitted]
    frac_bad = (d50 > tol50).mean()
    assert frac_bad < 0.01, f"p50 outlier fraction {frac_bad}"

    # Width agreement (statistical)
    both = (sigma_py > 0) & (sigma_nb > 0) & fitted[:, None]
    assert both.sum() > 0.5 * fitted.sum() * len(UBVJ_TEST)
    ratio = sigma_nb[both] / sigma_py[both]
    assert np.abs(np.median(np.log(ratio))) < 0.1, \
        f"rf sigma median ratio {np.median(ratio)}"

    # Rest-frame colors from the medians
    UV_py = -2.5 * np.log10(fr_py[:, 0, 2] / fr_py[:, 2, 2])
    UV_nb = -2.5 * np.log10(fr_nb[:, 0, 2] / fr_nb[:, 2, 2])
    ok = fitted & np.isfinite(UV_py) & np.isfinite(UV_nb)
    duv = np.abs(UV_py - UV_nb)[ok]
    assert np.median(duv) < 0.01, f"median U-V diff {np.median(duv)}"


def test_rest_frame_fluxes_simple(zbest_state):
    """
    simple=True: identical draws in, near-identical percentiles out.
    """
    ez = zbest_state
    ez.fit_at_zbest(get_err=True, use_numba=False, n_proc=0)
    ez.get_err = True

    _, _, fs_py = ez.rest_frame_fluxes(f_numbers=UBVJ_TEST, simple=True,
                                       use_numba=False, n_proc=0, verbose=0)
    _, _, fs_nb = ez.rest_frame_fluxes(f_numbers=UBVJ_TEST, simple=True,
                                       use_numba=True, n_proc=0, verbose=0)

    scale = np.abs(fs_py).max()
    assert np.allclose(fs_py, fs_nb, rtol=1e-4, atol=1e-6 * scale)


def test_fit_at_zbest_no_err(zbest_state):
    """
    get_err=False path returns zero efmodel/draws in both branches.
    """
    ez = zbest_state

    ez.fit_at_zbest(get_err=False, use_numba=True, n_proc=0)
    assert np.all(ez.efmodel == 0)

    ez.fit_at_zbest(get_err=False, use_numba=False, n_proc=0)
    assert np.all(ez.efmodel == 0)


def test_numba_seed_reproducible(zbest_state):
    """
    With the same random_seed, repeated numba fit_at_zbest runs give
    identical draws.
    """
    ez = zbest_state

    ez.fit_at_zbest(get_err=True, use_numba=True, n_proc=0)
    draws1 = ez.coeffs_draws.copy()

    ez.fit_at_zbest(get_err=True, use_numba=True, n_proc=0)
    draws2 = ez.coeffs_draws.copy()

    assert np.array_equal(draws1, draws2)
