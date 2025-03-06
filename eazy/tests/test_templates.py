import os
import pytest

import numpy as np

try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz

from .. import utils
from .. import templates
from .. import filters


@pytest.fixture
def read_template_ascii():
    path = os.path.join(utils.DATA_PATH, "templates")
    ascii_file = os.path.join(path, "fsps_full/fsps_QSF_12_v3_001.dat")
    templ = templates.Template(file=ascii_file)
    return templ


def test_read_template_ascii(read_template_ascii):
    """
    Test interpolation function
    """
    templ = read_template_ascii
    assert templ.name == "fsps_QSF_12_v3_001.dat"
    assert np.allclose(templ.flux.shape, [1, 5994])


@pytest.fixture
def read_template_fits():
    path = os.path.join(utils.DATA_PATH, "templates")
    fits_file = os.path.join(path, "spline_templates_v2/spline_age0.01_av0.0.fits")

    templ = templates.Template(file=fits_file)
    return templ


def test_read_template_fits(read_template_fits):
    """
    Read template FITS file
    """
    templ = read_template_fits
    assert np.allclose(templ.flux.shape, [templ.NZ, 12603])
    assert templ.name == "spline_age0.01_av0.0.fits"


def test_zscale():
    """
    Test zscale method
    """
    wave = np.arange(5000.0, 6000.0)
    flux = np.zeros_like(wave)
    flux[500] = 1.0

    templ = templates.Template(arrays=(wave, flux))

    z = 1.0
    zsc = templ.zscale(z=z, scalar=1.0)

    # wave = wave*(1+z)
    assert np.allclose(zsc.wave, templ.wave * (1 + z))

    # max index unchanged
    assert np.argmax(zsc.flux.flatten()) == 500

    # Still just one non-zero value
    assert (zsc.flux > 0).sum() == (flux > 0).sum()

    # Flux scaled by 1/(1+z)
    assert zsc.flux.max() == 1 / (1 + z)

    # Float and array scale
    for scalar in [2, wave * 0.0 + 2]:
        zsc = templ.zscale(z=z, scalar=scalar)
        assert zsc.flux.max() == np.max(scalar) / (1 + z)


def test_gaussian_templates():
    """
    Test templates.gaussian_templates
    """
    wave = np.arange(5000.0, 6000.0)
    centers = np.arange(5100.0, 5901.0, 100)

    width = 10
    widths = centers * 0 + width

    NW = len(wave)
    NG = len(centers)
    norm = np.sqrt(2 * np.pi * width**2)

    n0 = templates.gaussian_templates(wave, centers=centers, widths=widths, norm=False)

    assert np.allclose(n0.shape, (NW, NG))
    assert np.allclose(n0.max(), 1.0, rtol=1.0e-4)
    assert np.allclose(n0.sum(), norm * NG, rtol=1.0e-4)

    # Normalized
    n1 = templates.gaussian_templates(wave, centers=centers, widths=widths, norm=True)

    assert np.allclose(n1.shape, (NW, NG))
    assert np.allclose(n1.max(), 1.0 / norm, rtol=1.0e-4)
    assert np.allclose(n1.sum(), NG, rtol=1.0e-4)


def test_bspline_templates():
    """
    templates.bspline_templates
    """
    wave = np.arange(5000.0, 6000.0)
    NW = len(wave)

    df = 6

    for df in [6, 8, 12]:
        for log in [True, False]:
            spl = templates.bspline_templates(
                wave,
                degree=3,
                df=df,
                get_matrix=True,
                log=log,
                clip=0.0001,
                minmax=None,
            )

            assert np.allclose(spl.shape, (NW, df))
            assert np.allclose(spl.sum(axis=1), 1.0, rtol=1.0e-4)

    spt = templates.bspline_templates(
        wave, degree=3, df=df, get_matrix=False, log=log, clip=0.0001, minmax=None
    )

    assert len(spt) == df
    keys = list(spt.keys())
    for i, k in enumerate(keys):
        templ = spt[k]
        assert np.allclose(templ.wave, wave)
        assert np.allclose(spl[:, i], np.squeeze(templ.flux))


def test_redshift_dependent():
    """
    Redshift-dependent templates
    """
    wave = np.arange(5000.0, 6000.0)

    # No dependence
    flux = np.ones(len(wave))
    templ = templates.Template(arrays=(wave, flux), redshifts=[0])
    assert templ.zindex(-0.1, redshift_type="nearest") == 0
    assert templ.zindex(0.3, redshift_type="nearest") == 0

    assert templ.zindex(-0.1, redshift_type="floor") == 0
    assert templ.zindex(0.3, redshift_type="floor") == 0

    assert np.allclose(templ.zindex(-0.1, redshift_type="interp"), (0, 1.0))
    assert np.allclose(templ.zindex(0.1, redshift_type="interp"), (0, 1.0))

    # Redshift-dependent
    flux = np.ones((2, len(wave)))
    flux[1, :] = 2

    templ = templates.Template(arrays=(wave, flux), redshifts=[0, 1])

    assert templ.zindex(-0.1, redshift_type="nearest") == 0
    assert templ.zindex(0.3, redshift_type="nearest") == 0
    assert templ.zindex(0.6, redshift_type="nearest") == 1
    assert templ.zindex(2.6, redshift_type="nearest") == 1

    assert templ.zindex(-0.1, redshift_type="floor") == 0
    assert templ.zindex(0.3, redshift_type="floor") == 0
    assert templ.zindex(0.6, redshift_type="floor") == 0
    assert templ.zindex(2.6, redshift_type="floor") == 1

    assert np.allclose(templ.zindex(-0.1, redshift_type="interp"), (0, 1.0))
    assert np.allclose(templ.zindex(0.1, redshift_type="interp"), (0, 0.9))
    assert np.allclose(templ.zindex(0.9, redshift_type="interp"), (0, 0.1))
    assert np.allclose(templ.zindex(1.1, redshift_type="interp"), (1, 1.0))

    assert np.allclose(templ.flux_flam(iz=0, redshift_type="nearest"), 1.0)
    assert np.allclose(templ.flux_flam(iz=1, redshift_type="nearest"), 2.0)

    assert np.allclose(templ.flux_flam(z=-1.0, redshift_type="nearest"), 1.0)
    assert np.allclose(templ.flux_flam(z=0.0, redshift_type="nearest"), 1.0)
    assert np.allclose(templ.flux_flam(z=0.3, redshift_type="nearest"), 1.0)
    assert np.allclose(templ.flux_flam(z=1.5, redshift_type="nearest"), 2.0)

    assert np.allclose(templ.flux_flam(z=-1.0, redshift_type="interp"), 1.0)
    assert np.allclose(templ.flux_flam(z=0.0, redshift_type="interp"), 1.0)
    assert np.allclose(templ.flux_flam(z=0.3, redshift_type="interp"), 1.3)
    assert np.allclose(templ.flux_flam(z=1.5, redshift_type="interp"), 2.0)


def test_integrate_filter():
    """
    Integrating templates through filter throughput
    """
    import astropy.units as u

    # Tophat filter
    wx = np.arange(5400, 5600.0, 1)
    wy = wx * 0.0
    wy[10:-10] = 1

    f1 = filters.FilterDefinition(wave=wx, throughput=wy)

    # Flat-fnu spectrum
    wave = np.arange(1000.0, 9000.0)
    fnu = np.ones((2, len(wave))) * u.microJansky
    fnu[1, :] *= 2

    flam = fnu.to(
        utils.FLAM_CGS,
        equivalencies=u.equivalencies.spectral_density(wave * u.Angstrom),
    )

    templ = templates.Template(arrays=(wave, flam), redshifts=[0, 1])

    fnu_int = templ.integrate_filter(f1, z=0)
    assert np.allclose(fnu_int * utils.FNU_CGS, 1 * u.microJansky)

    fnu_int = templ.integrate_filter(f1, z=0, scale=2.0)
    assert np.allclose(fnu_int * utils.FNU_CGS, 2 * u.microJansky)

    fnu_int = templ.integrate_filter(f1, z=0.3, redshift_type="nearest")
    assert np.allclose(fnu_int * utils.FNU_CGS, 1 * u.microJansky)

    fnu_int = templ.integrate_filter(f1, z=0.3, redshift_type="interp")
    assert np.allclose(fnu_int * utils.FNU_CGS, 1.3 * u.microJansky)

    # Return f-lambda
    for z in [0, 0.2]:
        flam_interp = templ.integrate_filter(
            f1, z=z, flam=True, redshift_type="nearest"
        )

        wz = f1.pivot * (1 + z) * u.Angstrom
        flam_unit = (1 * u.microJansky).to(
            utils.FLAM_CGS, equivalencies=u.equivalencies.spectral_density(wz)
        )

        assert np.allclose(flam_interp * utils.FLAM_CGS, flam_unit)


def test_template_resampling():
    """
    Resampling preserving integrated flux
    """
    try:
        from grizli.utils_c import interp

        interp_grizli = interp.interp_conserve_c
    except:
        interp_grizli = None

    interp_eazy = utils.interp_conserve

    # Template with delta function line
    xtest = np.linspace(6550, 6576, 1024)
    ytest = xtest * 0

    ytest[len(xtest) // 2] = 1

    dx = np.diff(xtest)[0]

    tline = templates.Template(arrays=(xtest, ytest / dx))

    # Different resample grids
    for func in [interp_eazy, interp_grizli]:
        if func is None:
            continue

        for nstep in [16, 32, 64, 128]:
            wlo = np.linspace(6550, 6576, nstep)

            tlo = tline.resample(wlo, in_place=False)

            assert np.allclose(trapz(tlo.flux.flatten(), tlo.wave), 1.0, rtol=1.0e-3)

    # Arbitrarily-spaced wavelengths
    np.random.seed(1)
    for func in [interp_eazy, interp_grizli]:
        if func is None:
            continue

        for nstep in [16, 32, 64, 128]:
            wlo = np.sort(np.random.rand(nstep) * 26 + 6550)

            tlo = tline.resample(wlo, in_place=False)

            assert np.allclose(trapz(tlo.flux.flatten(), tlo.wave), 1.0, rtol=1.0e-3)


def test_template_smoothing():
    """
    Test template smoothing:

        - `eazy.templates.Template.smooth_velocity`
        - `eazy.templates.Template.to_observed_frame`

    """
    from astropy.stats import gaussian_sigma_to_fwhm

    #### Template with delta function line
    xtest = np.linspace(6550, 6576, 1024)
    ytest = xtest * 0
    ytest[len(xtest) // 2] = 1

    dx = np.diff(xtest)[0]

    tline = templates.Template(arrays=(xtest, ytest))

    #### Velocity smoothing
    vel = 100  # sigma
    pixel_sigma = vel / 3.0e5 * 6563.0 / dx

    tsm = tline.smooth_velocity(vel, in_place=False)

    assert np.allclose(
        tsm.flux.max(), 1.0 / np.sqrt(2 * np.pi) / pixel_sigma, rtol=1.0e-3
    )

    assert np.allclose(trapz(tsm.flux.flatten(), tsm.wave), dx, rtol=1.0e-3)

    #### MUSE LSF
    bacon_lsf_fwhm = lambda w: 5.866e-8 * w**2 - 9.187e-4 * w + 6.04
    lsf_sig = bacon_lsf_fwhm(6563) / gaussian_sigma_to_fwhm

    tlsf = tline.to_observed_frame(
        extra_sigma=0, lsf_func="Bacon", smoothspec_kwargs={"fftsmooth": False}
    )

    smax = 1 / np.sqrt(2 * np.pi) / (lsf_sig / dx)
    assert np.allclose(tlsf.flux.max(), smax, rtol=1.0e-3)
    assert np.allclose(trapz(tlsf.flux.flatten(), tlsf.wave), dx, rtol=1.0e-3)

    #### User LSF
    lsf_sig = 2.0
    my_lsf = lambda x: x * 0 + lsf_sig
    tlsf = tline.to_observed_frame(
        extra_sigma=0, lsf_func=my_lsf, smoothspec_kwargs={"fftsmooth": False}
    )

    smax = 1 / np.sqrt(2 * np.pi) / (lsf_sig / dx)
    assert np.allclose(tlsf.flux.max(), smax, rtol=1.0e-3)
    assert np.allclose(trapz(tlsf.flux.flatten(), tlsf.wave), dx, rtol=1.0e-3)

    #### No LSF is the same as smooth_velocity
    tobs = tline.to_observed_frame(
        extra_sigma=vel,
        lsf_func=None,
        to_air=False,
        z=0,
        smoothspec_kwargs={"fftsmooth": False},
        clip_wavelengths=None,
    )

    np.allclose(tobs.flux, tsm.flux, atol=tsm.flux.max() * 1.0e-3)

    #### Resampled
    for nstep in [16, 32, 64, 128]:
        wlo = np.linspace(6550, 6576, nstep)
        tlo = tline.to_observed_frame(
            extra_sigma=0,
            lsf_func="Bacon",
            smoothspec_kwargs={"fftsmooth": False},
            wavelengths=wlo,
        )

        assert np.allclose(trapz(tlo.flux.flatten(), tlo.wave), dx, rtol=1.0e-2)


def test_igm():

    kwarg_sets = [
        dict(igm_model="Asada24", scale_tau=1.0, add_cgm=True),
        dict(igm_model="Asada24", scale_tau=1.0, add_cgm=False),
        dict(igm_model="Inoue14", scale_tau=1.0, add_cgm=False),
    ]

    # Specific test of Inoue14 against result from a particular version
    # (db97f839cf8afe4a22c31c5d6195fd707ba4de32)
    zfix = 3.0
    rest_wave = np.arange(850, 1251, 50)
    igm_val = np.array(
        [
            0.33537573,
            0.54634578,
            0.74207249,
            0.74194787,
            0.79182545,
            0.75792504,
            0.72135181,
            0.68233589,
            1.0,
        ]
    )

    template = templates.Template(arrays=(rest_wave, rest_wave**0.0))
    for kwargs in kwarg_sets:
        igmz = template.igm_absorption(z=zfix, **kwargs)
        assert np.allclose(igmz, igm_val, rtol=1.0e-2)

    # Test other IGM properties
    wave = np.logspace(np.log10(800), np.log10(4000.0), 512)
    template = templates.Template(arrays=(wave, wave**0.0))

    fuv = wave < 1216.0
    fuv_cgm = wave < 2000.0

    for z in [2.0, 6.5]:
        with_cgm, without_cgm, inoue = [
            template.igm_absorption(z=z, **kwargs) for kwargs in kwarg_sets
        ]

        # Asada24(add_cgm=False) = Inoue14
        assert np.allclose(without_cgm, inoue)

        # No CGM in Asada24 at z < 6.0
        if z < 6.0:
            assert np.allclose(with_cgm, without_cgm)
        else:
            assert np.all(with_cgm[fuv_cgm] < without_cgm[fuv_cgm])

    for kwargs in kwarg_sets:
        i1 = template.igm_absorption(z=1.0, **kwargs)
        i2 = template.igm_absorption(z=2.0, **kwargs)
        i7 = template.igm_absorption(z=7.0, **kwargs)

        assert np.allclose(i1[~fuv], 1.0)
        assert np.all(i1[fuv] < 1.0)
        assert np.all(i1 >= i2)
        assert np.all(i2 >= i7)

        kwargs["scale_tau"] = 2.0
        it2 = template.igm_absorption(z=1.0, **kwargs)
        assert np.allclose(np.log(it2[fuv]) / np.log(i1[fuv]), 2.0)

        kwargs["scale_tau"] = 1.0
