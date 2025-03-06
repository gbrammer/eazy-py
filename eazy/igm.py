import os
import numpy as np

# from math import pow as _pow

from . import __file__ as filepath

__all__ = ["Asada24", "Inoue14"]


class Asada24(object):
    def __init__(
        self,
        sigmoid_params=(3.5918, 1.8414, 18.001),
        scale_tau=1.0,
        add_cgm=True,
        **kwargs,
    ):
        """
        Compute IGM+CGM transmission from Asada et al. 2024, in prep.
        The IGM model is from Inoue+ (2014).

        Parameters
        ----------
        sigmoid_params : 3-tuple of float
            Parameters that controll the redshift evolution of the CGM HI gas column
            density. The defaul values are from Asada et al. (2024).

        scale_tau : float
            Scalar multiplied to tau_igm

        add_cgm : bool
            Add the additional LyA damping absorption at z>6 as described in Asada+24.
            If False, the transmission will be identical to Inoue+2014

        .. plot::
            :include-source:

            # Compare two IGM transmissions

            import numpy as np
            import matplotlib.pyplot as plt
            from eazy import igm as igm_module

            igm_A24 = igm_module.Asada24()
            igm_I14 = igm_module.Inoue14()

            redshifts = [6., 7., 8., 9., 10.]
            colors = ['b', 'c', 'purple', 'orange', 'red']

            wave = np.linspace(100,2000,1901) ## wavelength array in the rest-frame
            lyman = wave < 2000


            fig = plt.figure(figsize=(6,5))
            for z, c in zip(redshifts, colors):
                igmz_A24 = wave*0.+1
                igmz_A24[lyman] = igm_A24.full_IGM(z, (wave*(1+z))[lyman])

                igmz_I14 = wave*0.+1
                igmz_I14[lyman] = igm_I14.full_IGM(z, (wave*(1+z))[lyman])

                plt.plot(wave*(1+z), igmz_I14, color=c, ls='dashed')
                plt.plot(wave*(1+z), igmz_A24, color=c, label=r'$z={}$'.format(int(z)))

            plt.xlabel('Observed wavelength [A]')
            plt.ylabel('Transmission')

            plt.legend()

            plt.xlim(5000,17000)
        """
        self._load_data()
        self.sigmoid_params = sigmoid_params
        self.scale_tau = scale_tau
        self.add_cgm = add_cgm

    def __repr__(self):
        attrs = ["sigmoid_params", "add_cgm", "scale_tau"]
        return (
            "<eazy.igm.Asada24 object>("
            + ", ".join([f"{k}={self.__dict__[k]}" for k in attrs])
            + ")"
        )

    @property
    def max_fuv_wave(self):
        """
        Maximum FUV wavelength (Angstroms) where IGM model will have an effect
        """
        if self.add_cgm:
            return 2000.0
        else:
            return 1300.0

    def tau_cgm(self, N_HI, lam, z):
        """
        CGM optical depth given by Totani+06, eqn (1)

        Parameters
        ----------
        N_HI : float
            HI column density [cm-2]

        lam : 1D array
            wavelength array in the observed frame [AA]

        z : float
            Redshift of the source

        Returns
        -------
        1D array of tau_CGM
        """

        lam_rest = lam / (1 + z)
        nu_rest = 3e18 / lam_rest

        tau = np.zeros(len(lam))

        for i, nu in enumerate(nu_rest):
            tau[i] = N_HI * sigma_a(nu)

        return tau

    def lgNHI_z(self, z):
        """
        HI column density as a function of redshift, calibrated in Asada+ 2024.
        Only valid at z>=6

        Parameters
        ----------
        z : float
            Redshift of the source

        Returns
        -------
        log10(HI column density [cm-2])
        """

        lgN_HI = sigmoid(
            z, self.sigmoid_params[0], self.sigmoid_params[1], self.sigmoid_params[2]
        )

        return lgN_HI

    ### IGM part -- identical to Inoue+14
    def _load_data(self):
        path = os.path.join(os.path.dirname(filepath), "data")
        # print path

        LAF_file = os.path.join(path, "LAFcoeff.txt")
        DLA_file = os.path.join(path, "DLAcoeff.txt")

        data = np.loadtxt(LAF_file, unpack=True)
        ix, lam, ALAF1, ALAF2, ALAF3 = data
        self.lam = lam[:, np.newaxis]
        self.ALAF1 = ALAF1[:, np.newaxis]
        self.ALAF2 = ALAF2[:, np.newaxis]
        self.ALAF3 = ALAF3[:, np.newaxis]

        data = np.loadtxt(DLA_file, unpack=True)
        ix, lam, ADLA1, ADLA2 = data
        self.ADLA1 = ADLA1[:, np.newaxis]
        self.ADLA2 = ADLA2[:, np.newaxis]

        return True

    @property
    def NA(self):
        """
        Number of Lyman-series lines
        """
        return self.lam.shape[0]

    def tLSLAF(self, zS, lobs):
        """
        Lyman series, Lyman-alpha forest
        """
        z1LAF = 1.2
        z2LAF = 4.7

        l2 = self.lam  # [:, np.newaxis]
        tLSLAF_value = np.zeros_like(lobs * l2).T

        x0 = lobs < l2 * (1 + zS)
        x1 = x0 & (lobs < l2 * (1 + z1LAF))
        x2 = x0 & ((lobs >= l2 * (1 + z1LAF)) & (lobs < l2 * (1 + z2LAF)))
        x3 = x0 & (lobs >= l2 * (1 + z2LAF))

        tLSLAF_value = np.zeros_like(lobs * l2)
        tLSLAF_value[x1] += ((self.ALAF1 / l2**1.2) * lobs**1.2)[x1]
        tLSLAF_value[x2] += ((self.ALAF2 / l2**3.7) * lobs**3.7)[x2]
        tLSLAF_value[x3] += ((self.ALAF3 / l2**5.5) * lobs**5.5)[x3]

        return tLSLAF_value.sum(axis=0)

    def tLSDLA(self, zS, lobs):
        """
        Lyman Series, DLA
        """
        z1DLA = 2.0

        l2 = self.lam  # [:, np.newaxis]
        tLSDLA_value = np.zeros_like(lobs * l2)

        x0 = (lobs < l2 * (1 + zS)) & (lobs < l2 * (1.0 + z1DLA))
        x1 = (lobs < l2 * (1 + zS)) & ~(lobs < l2 * (1.0 + z1DLA))

        tLSDLA_value[x0] += ((self.ADLA1 / l2**2) * lobs**2)[x0]
        tLSDLA_value[x1] += ((self.ADLA2 / l2**3) * lobs**3)[x1]

        return tLSDLA_value.sum(axis=0)

    def tLCDLA(self, zS, lobs):
        """
        Lyman continuum, DLA
        """
        z1DLA = 2.0
        lamL = 911.8

        tLCDLA_value = np.zeros_like(lobs)

        x0 = lobs < lamL * (1.0 + zS)
        if zS < z1DLA:
            tLCDLA_value[x0] = (
                0.2113 * (1.0 + zS) ** 2.0
                - 0.07661 * (1.0 + zS) ** 2.3 * (lobs[x0] / lamL) ** (-3e-1)
                - 0.1347 * (lobs[x0] / lamL) ** 2.0
            )
        else:
            x1 = lobs >= lamL * (1.0 + z1DLA)

            tLCDLA_value[x0 & x1] = (
                0.04696 * (1.0 + zS) ** 3.0
                - 0.01779 * (1.0 + zS) ** 3.3 * (lobs[x0 & x1] / lamL) ** (-3e-1)
                - 0.02916 * (lobs[x0 & x1] / lamL) ** 3.0
            )
            tLCDLA_value[x0 & ~x1] = (
                0.6340
                + 0.04696 * (1.0 + zS) ** 3.0
                - 0.01779 * (1.0 + zS) ** 3.3 * (lobs[x0 & ~x1] / lamL) ** (-3e-1)
                - 0.1347 * (lobs[x0 & ~x1] / lamL) ** 2.0
                - 0.2905 * (lobs[x0 & ~x1] / lamL) ** (-3e-1)
            )

        return tLCDLA_value

    def tLCLAF(self, zS, lobs):
        """
        Lyman continuum, LAF
        """
        z1LAF = 1.2
        z2LAF = 4.7
        lamL = 911.8

        tLCLAF_value = np.zeros_like(lobs)

        x0 = lobs < lamL * (1.0 + zS)

        if zS < z1LAF:
            tLCLAF_value[x0] = 0.3248 * (
                (lobs[x0] / lamL) ** 1.2
                - (1.0 + zS) ** (-9e-1) * (lobs[x0] / lamL) ** 2.1
            )
        elif zS < z2LAF:
            x1 = lobs >= lamL * (1 + z1LAF)
            tLCLAF_value[x0 & x1] = 2.545e-2 * (
                (1.0 + zS) ** 1.6 * (lobs[x0 & x1] / lamL) ** 2.1
                - (lobs[x0 & x1] / lamL) ** 3.7
            )
            tLCLAF_value[x0 & ~x1] = (
                2.545e-2 * (1.0 + zS) ** 1.6 * (lobs[x0 & ~x1] / lamL) ** 2.1
                + 0.3248 * (lobs[x0 & ~x1] / lamL) ** 1.2
                - 0.2496 * (lobs[x0 & ~x1] / lamL) ** 2.1
            )
        else:
            x1 = lobs > lamL * (1.0 + z2LAF)
            x2 = (lobs >= lamL * (1.0 + z1LAF)) & (lobs < lamL * (1.0 + z2LAF))
            x3 = lobs < lamL * (1.0 + z1LAF)

            tLCLAF_value[x0 & x1] = 5.221e-4 * (
                (1.0 + zS) ** 3.4 * (lobs[x0 & x1] / lamL) ** 2.1
                - (lobs[x0 & x1] / lamL) ** 5.5
            )
            tLCLAF_value[x0 & x2] = (
                5.221e-4 * (1.0 + zS) ** 3.4 * (lobs[x0 & x2] / lamL) ** 2.1
                + 0.2182 * (lobs[x0 & x2] / lamL) ** 2.1
                - 2.545e-2 * (lobs[x0 & x2] / lamL) ** 3.7
            )
            tLCLAF_value[x0 & x3] = (
                5.221e-4 * (1.0 + zS) ** 3.4 * (lobs[x0 & x3] / lamL) ** 2.1
                + 0.3248 * (lobs[x0 & x3] / lamL) ** 1.2
                - 3.140e-2 * (lobs[x0 & x3] / lamL) ** 2.1
            )

        return tLCLAF_value

    def full_IGM(self, z, lobs):
        """Get full IGM+CGM absorption

        Parameters
        ----------
        z : float
            Redshift to evaluate IGM absorption

        lobs : array-like
            Observed-frame wavelength(s) in Angstroms.

        Returns
        -------
        abs : array-like
            IGM+CGM transmission factor

        """

        tau_LS = self.tLSLAF(z, lobs) + self.tLSDLA(z, lobs)
        tau_LC = self.tLCLAF(z, lobs) + self.tLCDLA(z, lobs)
        tau_clip = 0.0

        igmz = np.exp(-self.scale_tau * (tau_LC + tau_LS + tau_clip))

        if self.add_cgm:
            if z < 6:
                tau_C = 0.0
            else:
                NHI = 10 ** (self.lgNHI_z(z))
                tau_C = self.tau_cgm(NHI, lobs, z)

            cgmz = np.exp(-tau_C)
        else:
            cgmz = 1.0

        return igmz * cgmz


class Inoue14(object):
    """
    IGM absorption from Inoue et al. (2014)
    """

    max_fuv_wave = 1300.0

    def __init__(self, scale_tau=1.0, **kwargs):
        """
        IGM absorption from Inoue et al. (2014)

        Parameters
        ----------
        scale_tau : float
            Parameter multiplied to the IGM :math:`\tau` values (exponential
            in the linear absorption fraction).
            I.e., :math:`f_\mathrm{igm} = e^{-\mathrm{scale\_tau} \tau}`.

        Attributes
        ----------
        max_fuv_wave : float
            Maximum FUV wavelength (Angstroms) where IGM model will have an effect
        """
        self._load_data()
        self.scale_tau = scale_tau

    def __repr__(self):
        attrs = ["scale_tau"]
        return (
            "<eazy.igm.Inoue14 object>("
            + ", ".join([f"{k}={self.__dict__[k]}" for k in attrs])
            + ")"
        )

    def _load_data(self):
        path = os.path.join(os.path.dirname(filepath), "data")
        # print path

        LAF_file = os.path.join(path, "LAFcoeff.txt")
        DLA_file = os.path.join(path, "DLAcoeff.txt")

        data = np.loadtxt(LAF_file, unpack=True)
        ix, lam, ALAF1, ALAF2, ALAF3 = data
        self.lam = lam[:, np.newaxis]
        self.ALAF1 = ALAF1[:, np.newaxis]
        self.ALAF2 = ALAF2[:, np.newaxis]
        self.ALAF3 = ALAF3[:, np.newaxis]

        data = np.loadtxt(DLA_file, unpack=True)
        ix, lam, ADLA1, ADLA2 = data
        self.ADLA1 = ADLA1[:, np.newaxis]
        self.ADLA2 = ADLA2[:, np.newaxis]

        return True

    @property
    def NA(self):
        """
        Number of Lyman-series lines
        """
        return self.lam.shape[0]

    def tLSLAF(self, zS, lobs):
        """
        Lyman series, Lyman-alpha forest
        """
        z1LAF = 1.2
        z2LAF = 4.7

        l2 = self.lam  # [:, np.newaxis]
        tLSLAF_value = np.zeros_like(lobs * l2).T

        x0 = lobs < l2 * (1 + zS)
        x1 = x0 & (lobs < l2 * (1 + z1LAF))
        x2 = x0 & ((lobs >= l2 * (1 + z1LAF)) & (lobs < l2 * (1 + z2LAF)))
        x3 = x0 & (lobs >= l2 * (1 + z2LAF))

        tLSLAF_value = np.zeros_like(lobs * l2)
        tLSLAF_value[x1] += ((self.ALAF1 / l2**1.2) * lobs**1.2)[x1]
        tLSLAF_value[x2] += ((self.ALAF2 / l2**3.7) * lobs**3.7)[x2]
        tLSLAF_value[x3] += ((self.ALAF3 / l2**5.5) * lobs**5.5)[x3]

        return tLSLAF_value.sum(axis=0)

    def tLSDLA(self, zS, lobs):
        """
        Lyman Series, DLA
        """
        z1DLA = 2.0

        l2 = self.lam  # [:, np.newaxis]
        tLSDLA_value = np.zeros_like(lobs * l2)

        x0 = (lobs < l2 * (1 + zS)) & (lobs < l2 * (1.0 + z1DLA))
        x1 = (lobs < l2 * (1 + zS)) & ~(lobs < l2 * (1.0 + z1DLA))

        tLSDLA_value[x0] += ((self.ADLA1 / l2**2) * lobs**2)[x0]
        tLSDLA_value[x1] += ((self.ADLA2 / l2**3) * lobs**3)[x1]

        return tLSDLA_value.sum(axis=0)

    def tLCDLA(self, zS, lobs):
        """
        Lyman continuum, DLA
        """
        z1DLA = 2.0
        lamL = 911.8

        tLCDLA_value = np.zeros_like(lobs)

        x0 = lobs < lamL * (1.0 + zS)
        if zS < z1DLA:
            tLCDLA_value[x0] = (
                0.2113 * (1.0 + zS) ** 2.0
                - 0.07661 * (1.0 + zS) ** 2.3 * (lobs[x0] / lamL) ** (-3e-1)
                - 0.1347 * (lobs[x0] / lamL) ** 2.0
            )
        else:
            x1 = lobs >= lamL * (1.0 + z1DLA)

            tLCDLA_value[x0 & x1] = (
                0.04696 * (1.0 + zS) ** 3.0
                - 0.01779 * (1.0 + zS) ** 3.3 * (lobs[x0 & x1] / lamL) ** (-3e-1)
                - 0.02916 * (lobs[x0 & x1] / lamL) ** 3.0
            )
            tLCDLA_value[x0 & ~x1] = (
                0.6340
                + 0.04696 * (1.0 + zS) ** 3.0
                - 0.01779 * (1.0 + zS) ** 3.3 * (lobs[x0 & ~x1] / lamL) ** (-3e-1)
                - 0.1347 * (lobs[x0 & ~x1] / lamL) ** 2.0
                - 0.2905 * (lobs[x0 & ~x1] / lamL) ** (-3e-1)
            )

        return tLCDLA_value

    def tLCLAF(self, zS, lobs):
        """
        Lyman continuum, LAF
        """
        z1LAF = 1.2
        z2LAF = 4.7
        lamL = 911.8

        tLCLAF_value = np.zeros_like(lobs)

        x0 = lobs < lamL * (1.0 + zS)

        if zS < z1LAF:
            tLCLAF_value[x0] = 0.3248 * (
                (lobs[x0] / lamL) ** 1.2
                - (1.0 + zS) ** (-9e-1) * (lobs[x0] / lamL) ** 2.1
            )
        elif zS < z2LAF:
            x1 = lobs >= lamL * (1 + z1LAF)
            tLCLAF_value[x0 & x1] = 2.545e-2 * (
                (1.0 + zS) ** 1.6 * (lobs[x0 & x1] / lamL) ** 2.1
                - (lobs[x0 & x1] / lamL) ** 3.7
            )
            tLCLAF_value[x0 & ~x1] = (
                2.545e-2 * (1.0 + zS) ** 1.6 * (lobs[x0 & ~x1] / lamL) ** 2.1
                + 0.3248 * (lobs[x0 & ~x1] / lamL) ** 1.2
                - 0.2496 * (lobs[x0 & ~x1] / lamL) ** 2.1
            )
        else:
            x1 = lobs > lamL * (1.0 + z2LAF)
            x2 = (lobs >= lamL * (1.0 + z1LAF)) & (lobs < lamL * (1.0 + z2LAF))
            x3 = lobs < lamL * (1.0 + z1LAF)

            tLCLAF_value[x0 & x1] = 5.221e-4 * (
                (1.0 + zS) ** 3.4 * (lobs[x0 & x1] / lamL) ** 2.1
                - (lobs[x0 & x1] / lamL) ** 5.5
            )
            tLCLAF_value[x0 & x2] = (
                5.221e-4 * (1.0 + zS) ** 3.4 * (lobs[x0 & x2] / lamL) ** 2.1
                + 0.2182 * (lobs[x0 & x2] / lamL) ** 2.1
                - 2.545e-2 * (lobs[x0 & x2] / lamL) ** 3.7
            )
            tLCLAF_value[x0 & x3] = (
                5.221e-4 * (1.0 + zS) ** 3.4 * (lobs[x0 & x3] / lamL) ** 2.1
                + 0.3248 * (lobs[x0 & x3] / lamL) ** 1.2
                - 3.140e-2 * (lobs[x0 & x3] / lamL) ** 2.1
            )

        return tLCLAF_value

    def full_IGM(self, z, lobs):
        """Get full Inoue IGM absorption

        Parameters
        ----------
        z : float
            Redshift to evaluate IGM absorption

        lobs : array
            Observed-frame wavelength(s) in Angstroms.

        Returns
        -------
        abs : array
            IGM absorption

        """
        tau_LS = self.tLSLAF(z, lobs) + self.tLSDLA(z, lobs)
        tau_LC = self.tLCLAF(z, lobs) + self.tLCDLA(z, lobs)

        ### Upturn at short wavelengths, low-z
        # k = 1./100
        # l0 = 600-6/k
        # clip = lobs/(1+z) < 600.
        # tau_clip = 100*(1-1./(1+np.exp(-k*(lobs/(1+z)-l0))))
        tau_clip = 0.0

        return np.exp(-self.scale_tau * (tau_LC + tau_LS + tau_clip))

    def build_grid(self, zgrid, lrest):
        """Build a spline interpolation object for fast IGM models

        Returns: self.interpolate
        """

        from scipy.interpolate import CubicSpline

        igm_grid = np.zeros((len(zgrid), len(lrest)))
        for iz in range(len(zgrid)):
            igm_grid[iz, :] = self.full_IGM(zgrid[iz], lrest * (1 + zgrid[iz]))

        self.interpolate = CubicSpline(zgrid, igm_grid)


def sigmoid(x, A, a, c):
    """
    Sigmoid function centered at x=6
    """

    return A / (1 + np.exp(-a * (x - 6))) + c


def sigma_a(nu_rest):
    """
     Lyα absorption cross section for the restframe frequency \nu given by Totani+06, eqn (1)
     for CGM damping wing calculation

    Parameters
    ----------
    nu : float
        rest-frame frequency [Hz]

    Returns
    -------
    Lyα absorption cross section at the restframe frequency \nu_rest [cm2]
    """

    Lam_a = 6.255486e8  ## Hz
    nu_lya = 2.46607e15  ## Hz

    C = 6.9029528e22  ## Constant factor [Ang2/s2]

    sig = (
        C
        * (nu_rest / nu_lya) ** 4
        / (
            4 * np.pi**2 * (nu_rest - nu_lya) ** 2
            + Lam_a**2 * (nu_rest / nu_lya) ** 6 / 4
        )
    )

    s = sig * 1e-16  ## convert AA-2 to cm-2

    return s
