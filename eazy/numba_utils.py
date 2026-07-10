"""
Numba JIT-compiled implementations of the core EAZY template fitting
functions in `eazy.photoz`.

Each kernel replicates the behavior of its pure-python counterpart:

- `template_lsq_numba` <-> `photoz.template_lsq` (fitter='nnls')
- `fit_by_redshift_numba` <-> `photoz.fit_by_redshift`
- `_single_object_fit_at_zbest_numba` <-> per-object body of
  `photoz._fit_at_zbest_group`
- `_single_object_rest_fluxes_refit` <-> per-object body of
  `photoz._fit_rest_group`, but with the flux percentiles evaluated
  analytically from the fit covariance rather than from random draws
  (the ``ndraws -> inf`` limit of the python implementation)
- `_rest_frame_fluxes_numba_simple` <-> ``simple=True`` branch of
  `photoz.PhotoZ.rest_frame_fluxes`

Internal math is done in float64 regardless of the catalog ``ARRAY_DTYPE``.

Note on fastmath: the kernels use a restricted set of LLVM fastmath flags
that excludes ``nnan``/``ninf``.  Full ``fastmath=True`` lets LLVM assume
no NaN/Inf values exist, which miscompiles the ``np.isfinite`` masking of
missing photometry and the finite-covariance guards.
"""
import numpy as np
import numba

# A small value to prevent division by zero and for float comparisons
EPSILON = np.finfo(np.float64).eps

# Must match photoz.MIN_VALID_FILTERS
MIN_VALID_FILTERS = 1

# fastmath flags that preserve NaN/Inf semantics (no 'nnan'/'ninf')
FASTMATH_SAFE = {'contract', 'reassoc', 'arcp', 'nsz'}


@numba.njit(cache=True)
def seed_numba_rng(seed):
    """
    Seed numba's internal random state (separate from `numpy.random`).
    """
    np.random.seed(seed)


@numba.njit(cache=True, fastmath=FASTMATH_SAFE)
def _nnls_numba(A, b):
    """
    Numba implementation of the Lawson-Hanson NNLS algorithm.

    Solves argmin_x || Ax - b ||_2 for x >= 0.
    """
    m, n = A.shape
    x = np.zeros(n, dtype=np.float64)
    w = np.dot(A.T, (b - np.dot(A, x)))

    # Active and passive sets
    P = np.zeros(n, dtype=np.bool_)
    Z = np.ones(n, dtype=np.bool_)

    max_iter = 3 * n
    for _ in range(max_iter):
        if np.all(w[Z] <= EPSILON):
            break

        j = np.argmax(w[Z])
        j_idx = np.where(Z)[0][j]

        P[j_idx] = True
        Z[j_idx] = False

        # Inner loop is bounded to protect against cycling
        for _inner in range(max_iter):
            Ap = A[:, P]
            s, _, _, _ = np.linalg.lstsq(Ap, b)

            s_full = np.zeros(n, dtype=np.float64)
            s_full[P] = s

            if np.all(s > -EPSILON):
                x = s_full
                w = np.dot(A.T, (b - np.dot(A, x)))
                break

            alpha = np.inf
            active_indices = np.where(P)[0]
            for i in range(len(s)):
                if s[i] < 0:
                    p_idx = active_indices[i]
                    val = x[p_idx] / (x[p_idx] - s[i])
                    if val < alpha:
                        alpha = val

            x += alpha * (s_full - x)

            # Move elements from P to Z where x is zero
            P[(P) & (np.abs(x) < EPSILON)] = False
            Z = ~P

    rnorm = np.linalg.norm(np.dot(A, x) - b)
    return x, rnorm


@numba.njit(cache=True, fastmath=FASTMATH_SAFE)
def _template_lsq_core(fnu_i, efnu_i, Ain, TEFz, zp, renorm_t, hess_threshold):
    """
    Core of `photoz.template_lsq` (fitter='nnls') in float64.

    Returns
    -------
    chi2 : float
    coeffs : array (NTEMP)
        Coefficients of the *input* (unnormalized) templates.
    fmodel : array (NFILT)
    Ax : array (n_ok_band, NTEMP)
        Error-weighted, renormalized design matrix on the valid bands,
        as used for the NNLS solution and its covariance.
    ok_band : bool array (NFILT)
    Anorm : array (NTEMP)
        Template renormalization (ones if ``renorm_t=False``).
    valid : bool
        False if the fit could not be performed (no valid data/templates).
    """
    NTEMP, NFILT = Ain.shape

    fnu = fnu_i.astype(np.float64)
    efnu = efnu_i.astype(np.float64)
    zp64 = zp.astype(np.float64)
    tef = TEFz.astype(np.float64)

    coeffs = np.zeros(NTEMP, dtype=np.float64)
    Anorm = np.ones(NTEMP, dtype=np.float64)
    fmodel = np.zeros(NFILT, dtype=np.float64)

    ok_band = np.zeros(NFILT, dtype=np.bool_)
    nok = 0
    for k in range(NFILT):
        if ((efnu[k] / zp64[k] > 0) & np.isfinite(fnu[k]) &
                np.isfinite(efnu[k])):
            ok_band[k] = True
            nok += 1

    Ax = np.zeros((nok, NTEMP), dtype=np.float64)

    if nok < MIN_VALID_FILTERS:
        return np.inf, coeffs, fmodel, Ax, ok_band, Anorm, False

    var = np.empty(NFILT, dtype=np.float64)
    rms = np.empty(NFILT, dtype=np.float64)
    for k in range(NFILT):
        var[k] = efnu[k] ** 2 + (tef[k] * max(fnu[k], 0.0)) ** 2
        rms[k] = np.sqrt(var[k])

    # Template norms over the valid bands; zero-norm templates are excluded
    # from the fit (photoz.template_lsq: ok_temp)
    ok_temp = np.zeros(NTEMP, dtype=np.bool_)
    for i in range(NTEMP):
        ssq = 0.0
        for k in range(NFILT):
            if ok_band[k]:
                v = Ain[i, k] / rms[k]
                ssq += v * v
        norm_val = np.sqrt(ssq)
        ok_temp[i] = norm_val > 0
        if renorm_t and (norm_val > 0):
            Anorm[i] = norm_val

    if not np.any(ok_temp):
        return np.inf, coeffs, fmodel, Ax, ok_band, Anorm, False

    # Renormalized templates and error-weighted design matrix
    A = np.empty((NTEMP, NFILT), dtype=np.float64)
    for i in range(NTEMP):
        for k in range(NFILT):
            A[i, k] = Ain[i, k] / Anorm[i]

    row = 0
    for k in range(NFILT):
        if ok_band[k]:
            for i in range(NTEMP):
                Ax[row, i] = A[i, k] / rms[k]
            row += 1

    # Prune degenerate templates with large Hessian cross terms
    if hess_threshold < 1.0:
        Hess = np.dot(Ax.T, Ax)
        for i in range(NTEMP):
            if not ok_temp[i]:
                continue
            for j in range(NTEMP):
                if (j != i) and (Hess[i, j] > hess_threshold):
                    ok_temp[j] = False

    n_temp_fit = 0
    for i in range(NTEMP):
        if ok_temp[i]:
            n_temp_fit += 1

    if n_temp_fit == 0:
        return np.inf, coeffs, fmodel, Ax, ok_band, Anorm, False

    Ax_fit = np.empty((nok, n_temp_fit), dtype=np.float64)
    col = 0
    fit_indices = np.empty(n_temp_fit, dtype=np.int64)
    for i in range(NTEMP):
        if ok_temp[i]:
            Ax_fit[:, col] = Ax[:, i]
            fit_indices[col] = i
            col += 1

    b = np.empty(nok, dtype=np.float64)
    row = 0
    for k in range(NFILT):
        if ok_band[k]:
            b[row] = fnu[k] / rms[k]
            row += 1

    coeffs_scaled = np.zeros(NTEMP, dtype=np.float64)
    try:
        coeffs_x, _ = _nnls_numba(Ax_fit, b)
        for col in range(n_temp_fit):
            coeffs_scaled[fit_indices[col]] = coeffs_x[col]
    except Exception:
        # Solver failure: zero coefficients, like photoz.template_lsq
        pass

    fmodel = np.dot(coeffs_scaled, A)
    chi2 = 0.0
    for k in range(NFILT):
        if ok_band[k]:
            chi2 += (fnu[k] - fmodel[k]) ** 2 / var[k]

    for i in range(NTEMP):
        coeffs[i] = coeffs_scaled[i] / Anorm[i]

    return chi2, coeffs, fmodel, Ax, ok_band, Anorm, True


@numba.njit(cache=True, fastmath=FASTMATH_SAFE)
def template_lsq_numba(fnu_i, efnu_i, Ain, TEFz, zp, renorm_t, hess_threshold):
    """
    Numba implementation of the core template least-squares fitting.

    Returns (chi2, coeffs, fmodel); see `photoz.template_lsq`.
    """
    chi2, coeffs, fmodel, Ax, ok_band, Anorm, valid = _template_lsq_core(
        fnu_i, efnu_i, Ain, TEFz, zp, renorm_t, hess_threshold
    )
    return chi2, coeffs, fmodel


@numba.njit(cache=True)
def _safe_invert_numba(mat):
    """
    Numba equivalent of `utils.safe_invert`: return the inverse, or a
    matrix of NaN if ``mat`` is singular by the `numpy.linalg.matrix_rank`
    criterion with the default tolerance.
    """
    n = mat.shape[0]
    s = np.linalg.svd(mat)[1]
    tol = s[0] * n * EPSILON
    rank = 0
    for i in range(n):
        if s[i] > tol:
            rank += 1

    if rank < n:
        return np.full((n, n), np.nan)

    return np.linalg.inv(mat)


@numba.njit(cache=True, fastmath=FASTMATH_SAFE)
def template_lsq_numba_covar(fnu_i, efnu_i, Ain, TEFz, zp, renorm_t,
                             hess_threshold):
    """
    Like `template_lsq_numba`, but also returns the covariance matrix of
    the coefficients of the templates active in the fit (``coeffs > 0``).

    The covariance is computed from the renormalized, error-weighted design
    matrix and then unscaled by ``Anorm``, so it applies directly to the
    returned (unscaled) ``coeffs``, matching the python path where the
    coefficient draws are divided by ``Anorm``.  Entries for inactive
    templates are NaN.
    """
    NTEMP = Ain.shape[0]
    chi2, coeffs, fmodel, Ax, ok_band, Anorm, valid = _template_lsq_core(
        fnu_i, efnu_i, Ain, TEFz, zp, renorm_t, hess_threshold
    )

    covar = np.full((NTEMP, NTEMP), np.nan, dtype=np.float64)
    if not valid:
        return chi2, coeffs, fmodel, covar

    active_indices = np.where(coeffs > 0)[0]
    n_active = len(active_indices)
    if n_active == 0:
        return chi2, coeffs, fmodel, covar

    Ax_active = np.empty((Ax.shape[0], n_active), dtype=np.float64)
    for j in range(n_active):
        Ax_active[:, j] = Ax[:, active_indices[j]]

    hessian = np.dot(Ax_active.T, Ax_active)
    covar_scaled = _safe_invert_numba(hessian)

    for r_idx in range(n_active):
        r = active_indices[r_idx]
        for c_idx in range(n_active):
            c = active_indices[c_idx]
            covar[r, c] = covar_scaled[r_idx, c_idx] / (Anorm[r] * Anorm[c])

    return chi2, coeffs, fmodel, covar


@numba.njit(parallel=True, cache=True)
def fit_by_redshift_numba(A, fnu_corr, efnu_corr, TEFz, zp, renorm_t,
                          hess_threshold):
    """
    Fits all objects at a single redshift in parallel using Numba.

    Replicates `photoz.fit_by_redshift`, including skipping objects with
    fewer than two valid bands (which keep chi2 = 0).
    """
    NOBJ, NFILT = fnu_corr.shape
    NTEMP = A.shape[0]

    chi2 = np.zeros(NOBJ, dtype=np.float64)
    coeffs = np.zeros((NOBJ, NTEMP), dtype=np.float64)

    for iobj in numba.prange(NOBJ):
        nvalid = 0
        for k in range(NFILT):
            if efnu_corr[iobj, k] > 0:
                nvalid += 1
        if nvalid < 2:
            continue

        chi2_obj, coeffs_obj, _ = template_lsq_numba(
            fnu_corr[iobj, :], efnu_corr[iobj, :], A, TEFz, zp,
            renorm_t, hess_threshold
        )
        chi2[iobj] = chi2_obj
        coeffs[iobj, :] = coeffs_obj

    return chi2, coeffs


def fit_by_redshift_dispatcher(fitter_name, A, fnu_corr, efnu_corr, TEFz, zp,
                               renorm_t, hess_threshold):
    """
    Selects the fitting function based on the provided name.
    This allows for easy extension to other numba-based fitters.
    """
    if fitter_name.lower() == 'numba_nnls':
        return fit_by_redshift_numba(
            A, fnu_corr, efnu_corr, TEFz, zp, renorm_t, hess_threshold
        )
    else:
        raise ValueError(
            f"Fitter '{fitter_name}' is not implemented in the dispatcher."
        )


@numba.njit(cache=True)
def multivariate_normal_numba(mean, cov, size):
    """
    Numba-compatible multivariate normal sampler.

    Uses the SVD factorization like `numpy.random.multivariate_normal`, so
    positive *semi*-definite covariance matrices are handled.  Returns NaN
    samples if ``cov`` is not finite.
    """
    n_dim = len(mean)
    samples = np.empty((size, n_dim), dtype=np.float64)

    if not np.all(np.isfinite(cov)):
        samples[:, :] = np.nan
        return samples

    u, s, _ = np.linalg.svd(cov)
    # Factor F with F @ F.T = cov
    F = np.empty((n_dim, n_dim), dtype=np.float64)
    for i in range(n_dim):
        for j in range(n_dim):
            F[i, j] = u[i, j] * np.sqrt(max(s[j], 0.0))

    for i in range(size):
        z = np.random.standard_normal(n_dim)
        samples[i, :] = mean + np.dot(F, z)

    return samples


@numba.njit(cache=True, fastmath=FASTMATH_SAFE)
def _gather_active_covar(covar, active_indices):
    """
    Extract the (n_active, n_active) submatrix of ``covar``; returns
    ``(covar_active, ok)`` with ``ok=False`` if any entry is not finite.
    """
    n_active = len(active_indices)
    covar_active = np.empty((n_active, n_active), dtype=np.float64)
    ok = True
    for r_idx in range(n_active):
        for c_idx in range(n_active):
            cv = covar[active_indices[r_idx], active_indices[c_idx]]
            covar_active[r_idx, c_idx] = cv
            if not np.isfinite(cv):
                ok = False
    return covar_active, ok


@numba.njit(cache=True, fastmath=FASTMATH_SAFE)
def _single_object_fit_at_zbest_numba(fnu_i, efnu_i, A, TEFz, zp, ndraws,
                                      renorm_t, hess_threshold):
    """
    Numba kernel to fit a single object, generate multivariate coefficient
    draws from the fit covariance, and propagate them to the model fluxes.

    Matches the per-object body of `photoz._fit_at_zbest_group`: the draws
    are *not* clipped to be positive.
    """
    ndraws_int = int(ndraws)
    NTEMP, NFILT = A.shape
    chi2, coeffs, fmodel, covar = template_lsq_numba_covar(
        fnu_i, efnu_i, A, TEFz, zp, renorm_t, hess_threshold
    )

    efmodel = np.zeros(NFILT, dtype=np.float64)
    coeffs_draws = np.zeros((ndraws_int, NTEMP), dtype=np.float64)

    if ndraws_int <= 0:
        return chi2, coeffs, fmodel, efmodel, coeffs_draws

    active_indices = np.where(coeffs > 0)[0]
    n_active = len(active_indices)
    if n_active == 0:
        return chi2, coeffs, fmodel, efmodel, coeffs_draws

    covar_active, covar_ok = _gather_active_covar(covar, active_indices)
    if not covar_ok:
        return chi2, coeffs, fmodel, efmodel, coeffs_draws

    draws = multivariate_normal_numba(
        coeffs[active_indices], covar_active, ndraws_int
    )

    if np.all(np.isfinite(draws)):
        for i in range(n_active):
            coeffs_draws[:, active_indices[i]] = draws[:, i]

        model_draws = np.dot(coeffs_draws, A.astype(np.float64))

        for k in range(NFILT):
            p16 = np.percentile(model_draws[:, k], 16)
            p84 = np.percentile(model_draws[:, k], 84)
            efmodel[k] = (p84 - p16) / 2.0

    return chi2, coeffs, fmodel, efmodel, coeffs_draws


@numba.njit(cache=True, fastmath=FASTMATH_SAFE)
def _single_object_rest_fluxes_refit(fnu_i, efnu_i, z, A, zp, rf_tempfilt_iz,
                                     rf_lc, pivot, pad_width, max_err,
                                     renorm_t, hess_threshold,
                                     percentile_sigma_multipliers):
    """
    Numba kernel to perform re-fitting for all rest-frame filters for a
    single object.

    Follows `photoz._fit_rest_group`, except that the flux percentiles are
    computed analytically as ``mean + sigma * norm.ppf(p)`` from the fit
    covariance.  Since the model flux is a linear combination of the
    (Gaussian) coefficient draws, this is the exact ``ndraws -> inf`` limit
    of the python implementation's sampled percentiles.
    """
    NREST = len(rf_lc)
    NFILT = len(pivot)
    NPERC = len(percentile_sigma_multipliers)
    f_rest_i = np.zeros((NREST, NPERC), dtype=np.float64)

    grow_denom = np.log(1.0 / (1.0 + pad_width)) ** 2
    grow = np.empty(NFILT, dtype=np.float64)
    TEFz = np.empty(NFILT, dtype=np.float64)

    for i in range(NREST):
        # Down-weight bands away from the rest-frame filter
        max_grow = 0.0
        for k in range(NFILT):
            x = np.log(rf_lc[i] / (pivot[k] / (1.0 + z)))
            grow[k] = np.exp(-x * x / (2.0 * grow_denom))
            if grow[k] > max_grow:
                max_grow = grow[k]

        for k in range(NFILT):
            TEFz[k] = (2.0 / (1.0 + grow[k] / max_grow) - 1.0) * max_err

        # Coefficients and covariance from the re-weighted fit
        _, coeffs, _, covar = template_lsq_numba_covar(
            fnu_i, efnu_i, A, TEFz, zp, renorm_t, hess_threshold
        )

        rf_templates_i = rf_tempfilt_iz[:, i].astype(np.float64)
        mean_flux = np.dot(coeffs, rf_templates_i)

        # Variance = F^T Cov(c) F over the active templates
        sigma_flux = 0.0
        active_indices = np.where(coeffs > 0)[0]
        if len(active_indices) > 0:
            covar_active, covar_ok = _gather_active_covar(
                covar, active_indices)
            if covar_ok:
                rf_active = rf_templates_i[active_indices]
                var_flux = np.dot(rf_active, np.dot(covar_active, rf_active))
                if var_flux > 0:
                    sigma_flux = np.sqrt(var_flux)

        for p_idx in range(NPERC):
            f_rest_i[i, p_idx] = (mean_flux +
                                  sigma_flux *
                                  percentile_sigma_multipliers[p_idx])

    return f_rest_i


@numba.njit(parallel=True, cache=True)
def _rest_frame_fluxes_numba_simple(coeffs_draws, izbest, rf_tempfilt,
                                    percentiles):
    """
    Numba-accelerated rest-frame flux calculation for the ``simple=True``
    case: percentiles of the stored coefficient draws propagated through
    the rest-frame template fluxes.
    """
    NOBJ, NDRAWS, NTEMP = coeffs_draws.shape
    NREST = rf_tempfilt.shape[2]
    f_rest = np.zeros((NOBJ, NREST, len(percentiles)),
                      dtype=coeffs_draws.dtype)

    for iobj in numba.prange(NOBJ):
        iz = izbest[iobj]
        draws_i = np.ascontiguousarray(coeffs_draws[iobj, :, :])

        for i in range(NREST):
            rf_i = np.ascontiguousarray(rf_tempfilt[iz, :, i])
            dval = np.dot(draws_i, rf_i)
            f_rest[iobj, i, :] = np.percentile(dval, percentiles)

    return f_rest
