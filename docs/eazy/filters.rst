.. filter-objects:

Bandpass filters in eazy-py
===========================

EAZY uses a single file do define all of the filters that are available to the
redshift fitting code. This is the same file that was defined for the original
EAZY code, and the most up-to-date version of the file is still distributed in
the ``eazy-photoz`` repository: `FILTERS.RES.latest
<https://github.com/gbrammer/eazy-photoz/blob/master/filters/>`_.

This file has lines like 

.. code-block::

    105 hst/wfc3/IR/f140w.dat lambda_c= 1.3924e+04 AB-Vega= 1.072 w95=3759.7
    1      1.15350e+04 0.00000e+00
    2      1.15800e+04 0.00000e+00
    3      1.16250e+04 4.96221e-04
    ...
    105    1.62150e+04 0.00000e+00
    {N} {description} lambda_c={pivot} ...
    1 {w1} {t1}
    2 {w2} {t2}
    ...
    N {wN} {tN}

where ``N`` is the number of samples for a given filter bandpass, ``description`` can be anything that describes the filter and ``pivot`` is the pivot wavelength of the filter.  The value of the pivot wavelength is recomputed internally, but currently the "lambda_c" text is needed to indicate the first line of a new filter.  The tabulated values below the description line are wavelength ``w``, in Angstroms, and throughput ``t`` with arbitrary units/normalization.  The full `~eazy.filters.FilterFile` is then a concatenation of these individual bandpass definitions.

.. note:: 
    Bandpasses are all now assumed to be associated with photon-counting
    devices (e.g., CCDs).
    
The `~eazy.filters.FilterFile` can be read as follows:

    >>> import os
    >>> from eazy import filters, utils
    >>> res = filters.FilterFile(os.path.join(utils.path_to_eazy_data(), 
    >>>                          'filters/FILTER.RES.latest'))
    >>> print(res.NFILT)
    407
    >>> for i in range(res.NFILT):
    >>>     print(f'{i+1} {res.filters[i].name}')
    1 hst/ACS_update_sep07/wfc_f435w_t77.dat obs_before_7-4-06+rebin-5A lambda_c= 4.3179e+03 AB-Vega=-0.104 w95=993.1
    2 hst/ACS_update_sep07/wfc_f475w_t77.dat obs_before_7-4-06+rebin-5A lambda_c= 4.7453e+03 AB-Vega=-0.101 w95=1412.8
    3 hst/ACS_update_sep07/wfc_f555w_t77.dat obs_before_7-4-06+rebin-5A lambda_c= 5.3601e+03 AB-Vega=-0.009 w95=1260.9
    4 hst/ACS_update_sep07/wfc_f606w_t77.dat obs_before_7-4-06+rebin-5A lambda_c= 5.9194e+03 AB-Vega= 0.082 w95=2225.4
    5 hst/ACS_update_sep07/wfc_f775w_t77.dat obs_before_7-4-06+rebin-5A lambda_c= 7.6933e+03 AB-Vega= 0.385 w95=1490.9
    6 hst/ACS_update_sep07/wfc_f814w_t77.dat obs_before_7-4-06+rebin-5A lambda_c= 8.0599e+03 AB-Vega= 0.419 w95=2359.0
    7 hst/ACS_update_sep07/wfc_f850lp_t77.dat obs_before_7-4-06+rebin-5A lambda_c= 9.0364e+03 AB-Vega= 0.519 w95=2096.6
    8 hst/nicmos_f110w.dat synphot-calcband lambda_c= 1.1234e+04 AB-Vega= 0.725 w95=5536.9
    9 hst/nicmos_f160w.dat synphot-calcband lambda_c= 1.6037e+04 AB-Vega= 1.306 w95=3864.6
    10 hst/wfpc2_f300w.dat synphot-calcband lambda_c= 2.9928e+03 AB-Vega= 1.350 w95=1257.0
    11 hst/wfpc2_f336w.dat synphot-calcband lambda_c= 3.3595e+03 AB-Vega= 1.182 w95=601.8
    12 hst/wfpc2_f450w.dat synphot-calcband lambda_c= 4.5573e+03 AB-Vega=-0.086 w95=1313.1
    13 hst/wfpc2_f555w.dat synphot-calcband lambda_c= 5.4429e+03 AB-Vega=-0.004 w95=1993.5
    14 hst/wfpc2_f606w.dat synphot-calcband lambda_c= 6.0013e+03 AB-Vega= 0.098 w95=2224.5
    15 hst/wfpc2_f702w.dat synphot-calcband lambda_c= 6.9171e+03 AB-Vega= 0.260 w95=2094.1
    16 hst/wfpc2_f814w.dat synphot-calcband lambda_c= 7.9960e+03 AB-Vega= 0.412 w95=2366.3
    17 hst/wfpc2_f850lp.dat synphot-calcband lambda_c= 9.1140e+03 AB-Vega= 0.516 w95=1699.0
    18 IRAC/irac_tr1_2004-08-09.dat 3.6micron lambda_c= 3.5569e+04 AB-Vega= 2.781 w95=7139.2
    19 IRAC/irac_tr2_2004-08-09.dat 4.5micron lambda_c= 4.5020e+04 AB-Vega= 3.254 w95=9705.5
    20 IRAC/irac_tr3_2004-08-09.dat 5.8micron lambda_c= 5.7450e+04 AB-Vega= 3.747 w95=13590.7
    21 IRAC/irac_tr4_2004-08-09.dat 8.0micron lambda_c= 7.9158e+04 AB-Vega= 4.387 w95=27892.8
    22 DEEP2-VVDS/mouldB_cfh7403.dat +atm lambda_c= 4.3122e+03 AB-Vega=-0.107 w95=968.6
    23 DEEP2-VVDS/mouldV_cfh7503.dat +atm lambda_c= 5.3564e+03 AB-Vega=-0.011 w95=1001.0
    24 DEEP2-VVDS/mouldR_cfh7603.dat +atm lambda_c= 6.5528e+03 AB-Vega= 0.206 w95=1285.9
    25 DEEP2-VVDS/mouldI_cfh7802.dat +atm lambda_c= 8.1551e+03 AB-Vega= 0.435 w95=2086.8
    26 KPNO/IRIMJ HDF-N+atm lambda_c= 1.2289e+04 AB-Vega= 0.879 w95=2643.0
    27 KPNO/IRIMH HDF-N+atm lambda_c= 1.6444e+04 AB-Vega= 1.359 w95=2767.1
    28 KPNO/IRIMK HDF-N+atm lambda_c= 2.2124e+04 AB-Vega= 1.872 w95=3916.9
    29 KPNO/IRIMKPRIME +atm lambda_c= 2.1648e+04 AB-Vega= 1.834 w95=3130.3
    30 ESO-NTT/SOFI_J.dat +atm lambda_c= 1.2359e+04 AB-Vega= 0.885 w95=2666.4
    31 ESO-NTT/SOFI_Js.dat +atm lambda_c= 1.2484e+04 AB-Vega= 0.909 w95=1571.7
    32 ESO-NTT/SOFI_H.dat +atm lambda_c= 1.6465e+04 AB-Vega= 1.358 w95=2832.6
    33 ESO-NTT/SOFI_Ks.dat +atm lambda_c= 2.1666e+04 AB-Vega= 1.837 w95=2684.8
    34 ESO/isaac_j.res ESO_ETC+atm lambda_c= 1.2356e+04 AB-Vega= 0.886 w95=2667.5
    35 ESO/isaac_js.res ESO_ETC+atm lambda_c= 1.2480e+04 AB-Vega= 0.909 w95=1573.4
    36 ESO/isaac_h.res ESO_ETC+atm lambda_c= 1.6496e+04 AB-Vega= 1.362 w95=2832.0
    37 ESO/isaac_ks.res ESO_ETC+atm lambda_c= 2.1667e+04 AB-Vega= 1.838 w95=2686.2
    38 ESO/WFI_u360specs.txt 2.2m-loiano+atm lambda_c= 3.5279e+03 AB-Vega= 1.069 w95=7865.0
    39 ESO/u35_rebin.dat 2.2m-GOODS-MUSIC+atm lambda_c= 3.5788e+03 AB-Vega= 0.974 w95=623.6
    40 musyc/U_1030_tot.dat lambda_c= 3.5843e+03 AB-Vega= 0.844 w95=784.7
    41 musyc/U_1255_tot.dat lambda_c= 3.7206e+03 AB-Vega= 0.586 w95=628.0
    42 musyc/U_cdfs_tot.dat alsoESO-DPS lambda_c= 3.5006e+03 AB-Vega= 1.033 w95=683.8
    43 musyc/U_hdfs_tot.dat lambda_c= 3.7206e+03 AB-Vega= 0.586 w95=628.0
    44 musyc/B_1030_tot.dat lambda_c= 4.4144e+03 AB-Vega=-0.103 w95=1297.9
    45 musyc/B_1255_tot.dat lambda_c= 4.4186e+03 AB-Vega=-0.102 w95=1319.0
    46 musyc/B_cdfs_tot.dat alsoESO-DPS lambda_c= 4.5910e+03 AB-Vega=-0.122 w95=975.2
    47 musyc/B_hdfs_tot.dat lambda_c= 4.4144e+03 AB-Vega=-0.103 w95=1297.9
    48 musyc/V_1030_tot.dat lambda_c= 5.4175e+03 AB-Vega=-0.006 w95=1263.0
    49 musyc/V_1255_tot.dat lambda_c= 5.4200e+03 AB-Vega=-0.006 w95=1263.6
    50 musyc/V_cdfs_tot.dat alsoESO-DPS lambda_c= 5.3724e+03 AB-Vega=-0.017 w95=919.6
    51 musyc/V_hdfs_tot.dat lambda_c= 5.4175e+03 AB-Vega=-0.006 w95=1263.0
    52 musyc/R_1030_tot.dat lambda_c= 6.5618e+03 AB-Vega= 0.195 w95=2491.9
    53 musyc/R_1255_tot.dat lambda_c= 6.5618e+03 AB-Vega= 0.195 w95=2491.9
    54 musyc/R_cdfs_tot.dat alsoESO-DPS lambda_c= 6.4992e+03 AB-Vega= 0.191 w95=1600.4
    55 musyc/R_hdfs_tot.dat lambda_c= 6.5618e+03 AB-Vega= 0.195 w95=2491.9
    56 musyc/I_1030_tot.dat lambda_c= 7.9768e+03 AB-Vega= 0.428 w95=1593.5
    57 musyc/I_1255_tot.dat lambda_c= 7.9838e+03 AB-Vega= 0.428 w95=1646.5
    58 musyc/I_cdfs_tot.dat alsoESO-DPS lambda_c= 8.6372e+03 AB-Vega= 0.488 w95=2459.3
    59 musyc/I_hdfs_tot.dat lambda_c= 8.0518e+03 AB-Vega= 0.436 w95=1529.1
    60 musyc/z_1030_tot.dat lambda_c= 9.0404e+03 AB-Vega= 0.516 w95=2040.8
    61 musyc/z_1255_tot.dat lambda_c= 9.0404e+03 AB-Vega= 0.516 w95=2040.8
    62 musyc/z_cdfs_tot.dat lambda_c= 9.0404e+03 AB-Vega= 0.516 w95=2040.8
    63 musyc/z_hdfs_tot.dat lambda_c= 9.0404e+03 AB-Vega= 0.516 w95=2040.8
    64 musyc/J_tot.dat hdfs1-1030 lambda_c= 1.2452e+04 AB-Vega= 0.901 w95=1566.2
    65 musyc/H_tot.dat hdfs1-1030 lambda_c= 1.6283e+04 AB-Vega= 1.341 w95=2863.2
    66 musyc/K_tot.dat hdfs1-1030 lambda_c= 2.1307e+04 AB-Vega= 1.803 w95=3231.7
    67 musyc/Ja_tot.dat.2cols hdfs2-1255 lambda_c= 1.2461e+04 AB-Vega= 0.905 w95=1543.3
    68 musyc/Ha_tot.dat.2cols hdfs2-1255 lambda_c= 1.6346e+04 AB-Vega= 1.349 w95=2710.3
    69 musyc/Ka_tot.dat.2cols hdfs2-1255 lambda_c= 2.1517e+04 AB-Vega= 1.821 w95=3123.6
    70 musyc/o3_hdfs_tot.dat lambda_c= 4.9977e+03 AB-Vega=-0.092 w95=76.4
    71 COSMOS/CFHT_filter_i.txt lambda_c= 7.6831e+03 AB-Vega= 0.382 w95=1511.7
    72 COSMOS/CFHT_filter_u.txt lambda_c= 3.8379e+03 AB-Vega= 0.330 w95=1578.3
    73 COSMOS/SDSS_filter_u.txt lambda_c= 3.5565e+03 AB-Vega= 0.938 w95=695.2
    74 COSMOS/SDSS_filter_g.txt lambda_c= 4.7025e+03 AB-Vega=-0.104 w95=1330.5
    75 COSMOS/SDSS_filter_r.txt lambda_c= 6.1766e+03 AB-Vega= 0.140 w95=1177.4
    76 COSMOS/SDSS_filter_i.txt lambda_c= 7.4961e+03 AB-Vega= 0.353 w95=1297.9
    77 COSMOS/SDSS_filter_z.txt lambda_c= 8.9467e+03 AB-Vega= 0.513 w95=1960.3
    78 COSMOS/SUBARU_filter_B.txt lambda_c= 4.4480e+03 AB-Vega=-0.112 w95=1035.4
    79 COSMOS/SUBARU_filter_V.txt lambda_c= 5.4702e+03 AB-Vega=-0.000 w95=993.1
    80 COSMOS/SUBARU_filter_g.txt lambda_c= 4.7609e+03 AB-Vega=-0.101 w95=1317.6
    81 COSMOS/SUBARU_filter_r.txt lambda_c= 6.2755e+03 AB-Vega= 0.154 w95=1379.4
    82 COSMOS/SUBARU_filter_i.txt lambda_c= 7.6712e+03 AB-Vega= 0.380 w95=1488.9
    83 COSMOS/SUBARU_filter_z.txt lambda_c= 9.0282e+03 AB-Vega= 0.514 w95=1410.5
    84 COSMOS/SUBARU_filter_NB816.txt lambda_c= 8.1509e+03 AB-Vega= 0.461 w95=162.0
    85 KPNO/FLAMINGOS.BARR.J.MAN240.ColdWitness.txt +atm lambda_c= 1.2461e+04 AB-Vega= 0.906 w95=1525.1
    86 KPNO/FLAMINGOS.BARR.H.MAN109.ColdWitness.txt +atm lambda_c= 1.6339e+04 AB-Vega= 1.348 w95=2685.1
    87 KPNO/FLAMINGOS.BARR.Ks.MAN306A.ColdWitness.txt COSMOS+atm lambda_c= 2.1542e+04 AB-Vega= 1.825 w95=3035.9
    88 megaprime/cfht_mega_u_cfh9301.dat CFHT-LS+atm lambda_c= 3.8280e+03 AB-Vega= 0.325 w95=771.0
    89 megaprime/cfht_mega_g_cfh9401.dat CFHT-LS+atm lambda_c= 4.8699e+03 AB-Vega=-0.090 w95=1427.7
    90 megaprime/cfht_mega_r_cfh9601.dat CFHT-LS+atm lambda_c= 6.2448e+03 AB-Vega= 0.151 w95=1232.3
    91 megaprime/cfht_mega_i_cfh9701.dat CFHT-LS+atm lambda_c= 7.6756e+03 AB-Vega= 0.382 w95=1501.0
    92 megaprime/cfht_mega_z_cfh9801.dat CFHT-LS+atm lambda_c= 8.8719e+03 AB-Vega= 0.510 w95=1719.4
    93 ESO/fors1_u_bess.res ms1054+atm lambda_c= 3.6825e+03 AB-Vega= 0.766 w95=508.9
    94 ESO/fors1_b_bess.res ms1054+atm lambda_c= 4.3430e+03 AB-Vega=-0.107 w95=1097.4
    95 ESO/fors1_v_bess.res ms1054+atm lambda_c= 5.5226e+03 AB-Vega= 0.011 w95=1473.4
    96 ESO/fors1_r_bess.res ESO_ETC+atm lambda_c= 6.5303e+03 AB-Vega= 0.191 w95=2391.0
    97 ESO/fors1_i_bess.res ESO_ETC+atm lambda_c= 7.8749e+03 AB-Vega= 0.413 w95=1509.2
    98 ESO/fors1_u_gunn.res ESO_ETC+atm lambda_c= 3.6196e+03 AB-Vega= 0.972 w95=452.0
    99 ESO/fors1_g_gunn.res ESO_ETC+atm lambda_c= 5.0867e+03 AB-Vega=-0.053 w95=766.2
    100 ESO/fors1_v_gunn.res ESO_ETC+atm lambda_c= 4.0022e+03 AB-Vega= 0.048 w95=801.4
    101 ESO/fors1_r_gunn.res ESO_ETC+atm lambda_c= 6.5160e+03 AB-Vega= 0.205 w95=960.8
    102 ESO/fors1_z_gunn.res ESO_ETC+atm lambda_c= 8.9782e+03 AB-Vega= 0.513 w95=1733.6
    103 ESO/vimos_u.res ESO_ETC+atm lambda_c= 3.7495e+03 AB-Vega= 0.471 w95=591.2
    104 ESO/wfi_BB_B123_ESO878.res ESO_ETC+atm lambda_c= 4.5922e+03 AB-Vega=-0.100 w95=1233.0
    105 ESO/wfi_BB_U50_ESO877.res ESO_ETC_u35+atm lambda_c= 3.6065e+03 AB-Vega= 0.935 w95=563.2
    106 ESO/wfi_U50_ESO877_exend3200.res ESO_ETC_u35+atm lambda_c= 3.6023e+03 AB-Vega= 0.941 w95=580.3
    107 ESO/wfi_BB_U38_ESO841.res ESO_ETC_u38_narrower_than_COMBO17/C17+atm lambda_c= 3.6865e+03 AB-Vega= 0.768 w95=474.7
    108 COMBO-17.old/epsi_U.dat U38_from_C17_website+atm lambda_c= 3.6616e+03 AB-Vega= 0.817 w95=505.9
    109 NOAO/steidel_Un_k1041bp_aug04.txt_ccd kpno-mosaic+atm lambda_c= 3.5765e+03 AB-Vega= 0.949 w95=623.4
    110 NOAO/steidel_G_k1042bp_aug04.txt_ccd kpno-mosaic+atm lambda_c= 4.8077e+03 AB-Vega=-0.106 w95=812.4
    111 NOAO/steidel_Rs_k1043bp_aug04.txt_ccd kpno-mosaic+atm lambda_c= 6.9408e+03 AB-Vega= 0.278 w95=1098.9
    112 COSMOS/gabasch_H_cosmos.txt Gabasch-H-COSMOS+atm lambda_c= 1.6427e+04 AB-Vega= 1.357 w95=2595.8
    113 CAPAK_v2/u_megaprime_sagem.res cosmos-u lambda_c= 3.8172e+03 AB-Vega= 0.322 w95=731.1
    114 CAPAK_v2/B_subaru.res cosmos-b lambda_c= 4.4480e+03 AB-Vega=-0.112 w95=1035.4
    115 CAPAK_v2/V_subaru.res cosmos-v lambda_c= 5.4702e+03 AB-Vega=-0.000 w95=993.1
    116 CAPAK_v2/r_subaru.res cosmos-r lambda_c= 6.2755e+03 AB-Vega= 0.154 w95=1379.4
    117 CAPAK_v2/i_subaru.res cosmos-i lambda_c= 7.6712e+03 AB-Vega= 0.380 w95=1488.9
    118 CAPAK_v2/z_subaru.res cosmos-z lambda_c= 9.0282e+03 AB-Vega= 0.514 w95=1410.5
    119 CAPAK_v2/flamingos_Ks.res cosmos-k lambda_c= 2.1519e+04 AB-Vega= 1.823 w95=3031.1
    120 CAPAK/galex1500.res FUV lambda_c= 1.5364e+03 AB-Vega= 2.128 w95=372.4
    121 CAPAK/galex2500.res NUV lambda_c= 2.2992e+03 AB-Vega= 1.665 w95=925.3
    122 UKIDSS/B_qe.txt UDS lambda_c= 4.4078e+03 AB-Vega=-0.099 w95=1083.6
    123 UKIDSS/R_qe.txt UDS lambda_c= 6.5083e+03 AB-Vega= 0.200 w95=1193.7
    124 UKIDSS/i_qe.txt UDS lambda_c= 7.6555e+03 AB-Vega= 0.378 w95=1523.9
    125 UKIDSS/z_qe.txt UDS lambda_c= 9.0602e+03 AB-Vega= 0.513 w95=1402.2
    126 UKIDSS/J.txt UDS lambda_c= 1.2502e+04 AB-Vega= 0.912 w95=1599.2
    127 UKIDSS/K.txt UDS lambda_c= 2.2060e+04 AB-Vega= 1.868 w95=3581.2
    128 NEWFIRM/j1_atmos.dat nmbs lambda_c= 1.0460e+04 AB-Vega= 0.636 w95=1470.5
    129 NEWFIRM/j2_atmos.dat nmbs lambda_c= 1.1946e+04 AB-Vega= 0.827 w95=1476.4
    130 NEWFIRM/j3_atmos.dat nmbs lambda_c= 1.2778e+04 AB-Vega= 0.946 w95=1394.3
    131 NEWFIRM/h1_atmos.dat nmbs lambda_c= 1.5601e+04 AB-Vega= 1.281 w95=1657.9
    132 NEWFIRM/h2_atmos.dat nmbs lambda_c= 1.7064e+04 AB-Vega= 1.417 w95=1720.5
    133 NEWFIRM/k.dat nmbs lambda_c= 2.1635e+04 AB-Vega= 1.831 w95=3180.7
    134 NEWFIRM/k_atmos.dat nmbs lambda_c= 2.1684e+04 AB-Vega= 1.836 w95=3128.2
    135 REST_FRAME/Bessel_UX.dat noCCD lambda_c= 3.5860e+03 AB-Vega= 0.816 w95=813.1
    136 REST_FRAME/Bessel_B.dat noCCD lambda_c= 4.3711e+03 AB-Vega=-0.107 w95=1324.8
    137 REST_FRAME/Bessel_V.dat noCCD lambda_c= 5.4776e+03 AB-Vega= 0.003 w95=1368.9
    138 REST_FRAME/Bessel_R.dat noCCD lambda_c= 6.4592e+03 AB-Vega= 0.173 w95=2215.0
    139 REST_FRAME/Bessel_I.dat noCCD lambda_c= 8.0201e+03 AB-Vega= 0.432 w95=1620.1
    140 REST_FRAME/Johnson-Cousins_U.dat lambda_c= 3.6299e+03 AB-Vega= 0.781 w95=690.8
    141 REST_FRAME/Johnson-Cousins_B.dat lambda_c= 4.2936e+03 AB-Vega=-0.056 w95=1311.2
    142 REST_FRAME/Johnson-Cousins_V.dat lambda_c= 5.4698e+03 AB-Vega= 0.000 w95=1368.4
    143 REST_FRAME/Johnson-Cousins_R.dat lambda_c= 6.4712e+03 AB-Vega= 0.180 w95=2333.1
    144 REST_FRAME/Johnson-Cousins_I.dat lambda_c= 7.8726e+03 AB-Vega= 0.414 w95=1328.9
    145 REST_FRAME/Gunn_u.dat incl_atmos_mirrors_ccd lambda_c= 3.5588e+03 AB-Vega= 0.927 w95=704.0
    146 REST_FRAME/Gunn_g.dat incl_atmos_mirrors_ccd lambda_c= 4.6726e+03 AB-Vega=-0.103 w95=1362.4
    147 REST_FRAME/Gunn_r.dat incl_atmos_mirrors_ccd lambda_c= 6.1739e+03 AB-Vega= 0.137 w95=1167.5
    148 REST_FRAME/Gunn_i.dat incl_atmos_mirrors_ccd lambda_c= 7.4926e+03 AB-Vega= 0.353 w95=1320.6
    149 REST_FRAME/Gunn_z.dat incl_atmos_mirrors_ccd lambda_c= 8.8740e+03 AB-Vega= 0.511 w95=1826.2
    150 REST_FRAME/Johnson-Morgan_U.dat 1951ApJ...114..522 lambda_c= 3.4902e+03 AB-Vega= 0.987 w95=872.0
    151 REST_FRAME/Johnson-Morgan_B.dat 1951ApJ...114..522 lambda_c= 4.3610e+03 AB-Vega=-0.100 w95=1143.6
    152 REST_FRAME/Johnson-Morgan_V.dat 1951ApJ...114..522 lambda_c= 5.4755e+03 AB-Vega= 0.016 w95=1462.1
    153 REST_FRAME/maiz-apellaniz_Johnson_U.res 2006AJ....131.1184M lambda_c= 3.5900e+03 AB-Vega= 0.769 w95=862.3
    154 REST_FRAME/maiz-apellaniz_Johnson_B.res 2006AJ....131.1184M lambda_c= 4.3722e+03 AB-Vega=-0.106 w95=1321.3
    155 REST_FRAME/maiz-apellaniz_Johnson_V.res 2006AJ....131.1184M lambda_c= 5.4794e+03 AB-Vega= 0.002 w95=1369.1
    156 SDSS/u.dat DR7+atm lambda_c= 3.5565e+03 AB-Vega= 0.938 w95=695.2
    157 SDSS/g.dat DR7+atm lambda_c= 4.7025e+03 AB-Vega=-0.104 w95=1330.5
    158 SDSS/r.dat DR7+atm lambda_c= 6.1756e+03 AB-Vega= 0.140 w95=1177.4
    159 SDSS/i.dat DR7+atm lambda_c= 7.4900e+03 AB-Vega= 0.352 w95=1297.0
    160 SDSS/z.dat DR7+atm lambda_c= 8.9467e+03 AB-Vega= 0.513 w95=1960.3
    161 2MASS/J.res lambda_c= 1.2358e+04 AB-Vega= 0.885 w95=2311.1
    162 2MASS/H.res lambda_c= 1.6458e+04 AB-Vega= 1.362 w95=2680.1
    163 2MASS/K.res lambda_c= 2.1603e+04 AB-Vega= 1.830 w95=3020.3
    164 COMBO17/C17_420.res lambda_c= 4.1797e+03 AB-Vega=-0.173 w95=319.6
    165 COMBO17/C17_464.res lambda_c= 4.6173e+03 AB-Vega=-0.170 w95=156.5
    166 COMBO17/C17_485.res lambda_c= 4.8603e+03 AB-Vega=-0.057 w95=319.8
    167 COMBO17/C17_518.res lambda_c= 5.1885e+03 AB-Vega=-0.050 w95=179.4
    168 COMBO17/C17_571.res lambda_c= 5.7172e+03 AB-Vega= 0.046 w95=269.5
    169 COMBO17/C17_604.res lambda_c= 6.0450e+03 AB-Vega= 0.107 w95=219.7
    170 COMBO17/C17_646.res lambda_c= 6.4518e+03 AB-Vega= 0.217 w95=277.0
    171 COMBO17/C17_696.res lambda_c= 6.9595e+03 AB-Vega= 0.268 w95=225.9
    172 COMBO17/C17_753.res lambda_c= 7.5310e+03 AB-Vega= 0.363 w95=231.3
    173 COMBO17/C17_815.res lambda_c= 8.1578e+03 AB-Vega= 0.462 w95=262.1
    174 COMBO17/C17_855.res lambda_c= 8.5571e+03 AB-Vega= 0.538 w95=265.4
    175 COMBO17/C17_915.res lambda_c= 9.1409e+03 AB-Vega= 0.505 w95=344.7
    176 COMBO17/C17_U.res lambda_c= 3.6527e+03 AB-Vega= 0.836 w95=511.4
    177 COMBO17/C17_B.res lambda_c= 4.5726e+03 AB-Vega=-0.124 w95=980.2
    178 COMBO17/C17_V.res lambda_c= 5.3709e+03 AB-Vega=-0.017 w95=918.6
    179 COMBO17/C17_R.res lambda_c= 6.4664e+03 AB-Vega= 0.183 w95=1599.3
    180 COMBO17/C17_I.res lambda_c= 8.5538e+03 AB-Vega= 0.483 w95=1874.2
    181 Subaru_MB/IA427.dat lambda_c= 4.2600e+03 AB-Vega=-0.161 w95=223.1
    182 Subaru_MB/IA445.dat lambda_c= 4.4427e+03 AB-Vega=-0.141 w95=219.1
    183 Subaru_MB/IA464.dat lambda_c= 4.6333e+03 AB-Vega=-0.167 w95=237.8
    184 Subaru_MB/IA484.dat lambda_c= 4.8473e+03 AB-Vega=-0.037 w95=249.5
    185 Subaru_MB/IA505.dat lambda_c= 5.0608e+03 AB-Vega=-0.077 w95=259.0
    186 Subaru_MB/IA527.dat lambda_c= 5.2593e+03 AB-Vega=-0.035 w95=281.7
    187 Subaru_MB/IA550.dat lambda_c= 5.4950e+03 AB-Vega= 0.006 w95=305.3
    188 Subaru_MB/IA574.dat lambda_c= 5.7629e+03 AB-Vega= 0.054 w95=303.3
    189 Subaru_MB/IA598.dat lambda_c= 6.0071e+03 AB-Vega= 0.101 w95=330.7
    190 Subaru_MB/IA624.dat lambda_c= 6.2308e+03 AB-Vega= 0.142 w95=336.6
    191 Subaru_MB/IA651.dat lambda_c= 6.4984e+03 AB-Vega= 0.236 w95=359.6
    192 Subaru_MB/IA679.dat lambda_c= 6.7816e+03 AB-Vega= 0.245 w95=371.9
    193 Subaru_MB/IA709.dat lambda_c= 7.0735e+03 AB-Vega= 0.287 w95=358.3
    194 Subaru_MB/IA738.dat lambda_c= 7.3595e+03 AB-Vega= 0.334 w95=355.0
    195 Subaru_MB/IA768.dat lambda_c= 7.6804e+03 AB-Vega= 0.387 w95=388.5
    196 Subaru_MB/IA797.dat lambda_c= 7.9662e+03 AB-Vega= 0.432 w95=403.5
    197 Subaru_MB/IA827.dat lambda_c= 8.2468e+03 AB-Vega= 0.475 w95=367.0
    198 Subaru_MB/IA856.dat lambda_c= 8.5648e+03 AB-Vega= 0.534 w95=379.0
    199 Subaru_MB/IA907.dat lambda_c= 9.0704e+03 AB-Vega= 0.497 w95=452.0
    200 COSMOS/CFHT_filter_Ks.txt lambda_c= 2.1571e+04 AB-Vega= 1.827 w95=3171.1
    201 hst/wfc3/IR/f098m.dat calcband_wfc3-ir-f098m lambda_c= 9.8668e+03 AB-Vega= 0.558 w95=1631.3
    202 hst/wfc3/IR/f105w.dat lambda_c= 1.0545e+04 AB-Vega= 0.641 w95=2781.0
    203 hst/wfc3/IR/f125w.dat lambda_c= 1.2471e+04 AB-Vega= 0.895 w95=2867.0
    204 hst/wfc3/IR/f140w.dat lambda_c= 1.3924e+04 AB-Vega= 1.072 w95=3759.7
    205 hst/wfc3/IR/f160w.dat lambda_c= 1.5396e+04 AB-Vega= 1.250 w95=2743.9
    206 hst/wfc3/UVIS/f218w.dat calcband_wfc3-uvis1-f218w lambda_c= 2.2272e+03 AB-Vega= 1.690 w95=467.2
    207 hst/wfc3/UVIS/f225w.dat lambda_c= 2.3707e+03 AB-Vega= 1.660 w95=691.7
    208 hst/wfc3/UVIS/f275w.dat lambda_c= 2.7086e+03 AB-Vega= 1.501 w95=588.4
    209 hst/wfc3/UVIS/f336w.dat lambda_c= 3.3537e+03 AB-Vega= 1.185 w95=542.7
    210 hst/wfc3/UVIS/f390w.dat lambda_c= 3.9219e+03 AB-Vega= 0.220 w95=979.4
    211 hst/wfc3/UVIS/f438w.dat lambda_c= 4.3256e+03 AB-Vega=-0.154 w95=655.9
    212 hst/wfc3/UVIS/f475w.dat lambda_c= 4.7715e+03 AB-Vega=-0.100 w95=1415.6
    213 hst/wfc3/UVIS/f555w.dat lambda_c= 5.3086e+03 AB-Vega=-0.027 w95=1987.5
    214 hst/wfc3/UVIS/f606w.dat lambda_c= 5.8925e+03 AB-Vega= 0.080 w95=2193.0
    215 hst/wfc3/UVIS/f625w.dat lambda_c= 6.2451e+03 AB-Vega= 0.143 w95=1501.1
    216 hst/wfc3/UVIS/f775w.dat lambda_c= 7.6576e+03 AB-Vega= 0.378 w95=1423.7
    217 hst/wfc3/UVIS/f814w.dat lambda_c= 8.0595e+03 AB-Vega= 0.417 w95=2377.1
    218 REST_FRAME/UV1600.dat Width_350A lambda_c= 1.5967e+03 AB-Vega= 2.033 w95=332.3
    219 REST_FRAME/UV2800.dat Width_350A lambda_c= 2.7942e+03 AB-Vega= 1.452 w95=322.9
    220 WIRCam/cfh8101_J.txt +atm lambda_c= 1.2530e+04 AB-Vega= 0.914 w95=1540.6
    221 WIRCam/cfh8201_H.txt +atm lambda_c= 1.6294e+04 AB-Vega= 1.342 w95=2766.2
    222 WIRCam/cfh8302_Ks.txt +atm lambda_c= 2.1574e+04 AB-Vega= 1.827 w95=3151.1
    223 MOIRCS/MKO_Y_ED537.txt +atm lambda_c= 1.0242e+04 AB-Vega= 0.600 w95=961.0
    224 MOIRCS/J277.txt +atm lambda_c= 1.2517e+04 AB-Vega= 0.913 w95=1571.2
    225 MOIRCS/H117.txt +atm lambda_c= 1.6347e+04 AB-Vega= 1.348 w95=2686.4
    226 MOIRCS/Ks_rot1707wedged120K.txt +atm lambda_c= 2.1577e+04 AB-Vega= 1.828 w95=3043.6
    227 NOAO/k1001bp_jul04.txt MOSAIC-U-2004+atm lambda_c= 3.5929e+03 AB-Vega= 0.842 w95=720.7
    228 NOAO/k1001bp_jul04.txt MOSAIC-U-2004 lambda_c= 3.5667e+03 AB-Vega= 0.882 w95=738.9
    229 LRIS/g_blue_transmission.dat +atm lambda_c= 4.7508e+03 AB-Vega=-0.105 w95=940.4
    230 LRIS/g_blue_transmission.dat lambda_c= 4.7421e+03 AB-Vega=-0.106 w95=942.2
    231 LRIS/Rs_LRISred_transmission.dat +atm lambda_c= 6.8186e+03 AB-Vega= 0.247 w95=1461.1
    232 LRIS/Rs_LRISred_transmission.dat lambda_c= 6.8214e+03 AB-Vega= 0.250 w95=1658.9
    233 hst/ACS_update_sep07/wfc_f435w_t81.dat obs_AFTER_7-4-06+rebin-5A lambda_c= 4.3189e+03 AB-Vega=-0.104 w95=992.7
    234 hst/ACS_update_sep07/wfc_f475w_t81.dat obs_AFTER_7-4-06+rebin-5A lambda_c= 4.7469e+03 AB-Vega=-0.100 w95=1412.3
    235 hst/ACS_update_sep07/wfc_f555w_t81.dat obs_AFTER_7-4-06+rebin-5A lambda_c= 5.3610e+03 AB-Vega=-0.009 w95=1260.7
    236 hst/ACS_update_sep07/wfc_f606w_t81.dat obs_AFTER_7-4-06+rebin-5A lambda_c= 5.9211e+03 AB-Vega= 0.083 w95=2224.8
    237 hst/ACS_update_sep07/wfc_f625w_t81.dat obs_AFTER_7-4-06+rebin-5A lambda_c= 6.3114e+03 AB-Vega= 0.158 w95=1391.2
    238 hst/ACS_update_sep07/wfc_f775w_t81.dat obs_AFTER_7-4-06+rebin-5A lambda_c= 7.6924e+03 AB-Vega= 0.384 w95=1490.9
    239 hst/ACS_update_sep07/wfc_f814w_t81.dat obs_AFTER_7-4-06+rebin-5A lambda_c= 8.0570e+03 AB-Vega= 0.419 w95=2357.9
    240 hst/ACS_update_sep07/wfc_f850lp_t81.dat obs_AFTER_7-4-06+rebin-5A lambda_c= 9.0331e+03 AB-Vega= 0.517 w95=2091.9
    241 hst/wfc3/IR/f110w.dat lambda_c= 1.1534e+04 AB-Vega= 0.755 w95=4722.7
    242 hst/wfc3/UVIS/f475x.dat lambda_c= 4.9414e+03 AB-Vega=-0.051 w95=2458.5
    243 hst/wfc3/UVIS/f600lp.dat lambda_c= 7.4505e+03 AB-Vega= 0.321 w95=3700.3
    244 WISE/RSR-W1.txt lambda_c= 3.3682e+04 AB-Vega= 2.661 w95=9257.5
    245 WISE/RSR-W2.txt lambda_c= 4.6179e+04 AB-Vega= 3.301 w95=10858.4
    246 WISE/RSR-W3.txt lambda_c= 1.2073e+05 AB-Vega= 5.135 w95=82697.2
    247 WISE/RSR-W4.txt lambda_c= 2.2194e+05 AB-Vega= 6.610 w95=59683.6
    248 FOURSTAR/J_cam_optics_sky.txt lambda_c= 1.2405e+04 AB-Vega= 0.892 w95=2118.2
    249 FOURSTAR/J1_cam_optics_sky.txt lambda_c= 1.0540e+04 AB-Vega= 0.653 w95=984.1
    250 FOURSTAR/J2_cam_optics_sky.txt lambda_c= 1.1448e+04 AB-Vega= 0.771 w95=1372.5
    251 FOURSTAR/J3_cam_optics_sky.txt lambda_c= 1.2802e+04 AB-Vega= 0.954 w95=1278.7
    252 FOURSTAR/H_cam_optics_sky.txt lambda_c= 1.6180e+04 AB-Vega= 1.333 w95=2691.7
    253 FOURSTAR/Hlong_cam_optics_sky.txt lambda_c= 1.7020e+04 AB-Vega= 1.412 w95=1524.4
    254 FOURSTAR/Hshort_cam_optics_sky.txt lambda_c= 1.5544e+04 AB-Vega= 1.274 w95=1523.6
    255 FOURSTAR/Ks_cam_optics_sky.txt lambda_c= 2.1538e+04 AB-Vega= 1.824 w95=3202.3
    256 VISTA/Y_system+atmos.dat at80K_forETC+trans_10_10 lambda_c= 1.0217e+04 AB-Vega= 0.596 w95=1025.7
    257 VISTA/J_system+atmos.dat lambda_c= 1.2527e+04 AB-Vega= 0.911 w95=1703.1
    258 VISTA/H_system+atmos.dat lambda_c= 1.6433e+04 AB-Vega= 1.355 w95=2843.9
    259 VISTA/Ks_system+atmos.dat lambda_c= 2.1503e+04 AB-Vega= 1.819 w95=3108.7
    260 ESO/VIMOS/R.dat Avg.Q1-Q4 lambda_c= 6.4427e+03 AB-Vega= 0.182 w95=1332.6
    261 UKIDSS/Table02_online.dat Hewitt_WFCAM_Z lambda_c= 8.8262e+03 AB-Vega= 0.508 w95=945.4
    262 UKIDSS/Table03_online.dat Hewitt_WFCAM_Y lambda_c= 1.0315e+04 AB-Vega= 0.610 w95=1058.5
    263 UKIDSS/Table04_online.dat Hewitt_WFCAM_J lambda_c= 1.2502e+04 AB-Vega= 0.912 w95=1599.2
    264 UKIDSS/Table05_online.dat Hewitt_WFCAM_H lambda_c= 1.6360e+04 AB-Vega= 1.348 w95=2971.7
    265 UKIDSS/Table06_online.dat Hewitt_WFCAM_K lambda_c= 2.2060e+04 AB-Vega= 1.868 w95=3581.2
    266 VLT/hawki_y_ETC.dat +atm lambda_c= 1.0207e+04 AB-Vega=0.597 w95=1036.7
    267 VLT/hawki_j_ETC.dat +atm lambda_c= 1.2574e+04 AB-Vega=0.925 w95=1526.2
    268 VLT/hawki_h_ETC.dat +atm lambda_c= 1.6193e+04 AB-Vega=1.337 w95=2792.0
    269 VLT/hawki_k_ETC.dat +atm lambda_c= 2.1524e+04 AB-Vega=1.826 w95=3209.9
    270 RestUV/Tophat_1400_200.dat lambda_c= 1.3985e+03 AB-Vega=2.519 w95=191.1
    271 RestUV/Tophat_1700_200.dat lambda_c= 1.6989e+03 AB-Vega=1.916 w95=190.7
    272 RestUV/Tophat_2200_200.dat lambda_c= 2.1993e+03 AB-Vega=1.691 w95=191.1
    273 RestUV/Tophat_2700_200.dat lambda_c= 2.6994e+03 AB-Vega=1.504 w95=190.9
    274 RestUV/Tophat_2800_200.dat lambda_c= 2.7996e+03 AB-Vega=1.465 w95=191.2
    275 RestUV/Tophat_5500_200.dat lambda_c= 5.5002e+03 AB-Vega=0.010 w95=190.0
    276 VLT/VST.OCam.B.dat ETC_A-D_withADC lambda_c= 4.4458e+03 AB-Vega=-0.116 w95=987.8
    277 VLT/VST.OCam.V.dat ETC_A-D_withADC lambda_c= 5.5013e+03 AB-Vega=0.008 w95=946.4
    278 VLT/VST.OCam.sdss.u.dat ETCwithADC lambda_c= 3.6482e+03 AB-Vega=0.845 w95=519.5
    279 VLT/VST.OCam.sdss.g.dat ETCwithADC lambda_c= 4.7745e+03 AB-Vega=-0.097 w95=1284.2
    280 VLT/VST.OCam.sdss.r.dat ETCwithADC lambda_c= 6.2899e+03 AB-Vega=0.159 w95=1384.6
    281 VLT/VST.OCam.sdss.i.dat ETCwithADC lambda_c= 7.5021e+03 AB-Vega=0.356 w95=1561.3
    282 VLT/VST.OCam.sdss.z.dat ETCwithADC lambda_c= 8.8424e+03 AB-Vega=0.517 w95=1114.8
    283 Subaru/suprime_Ic.dat lambda_c= 7.9605e+03 AB-Vega=0.432 w95=1428.3
    284 Subaru/suprime_FDCCD_z.res lambda_c= 9.0963e+03 AB-Vega=0.516 w95=1424.0
    285 Subaru/suprime_Rc.dat lambda_c= 6.5054e+03 AB-Vega=0.203 w95=1192.2
    286 IMACS/sloan_u_blue01.data lambda_c= 3.6280e+03 AB-Vega=0.954
    287 IMACS/sloan_g_blue01.data lambda_c= 4.7553e+03 AB-Vega=-0.100
    288 IMACS/sloan_r_red01.data lambda_c= 6.5012e+03 AB-Vega=0.205
    289 IMACS/sloan_i_red01.data lambda_c= 7.7230e+03 AB-Vega=0.392
    290 IMACS/sloan_z_red01.data lambda_c= 9.7105e+03 AB-Vega=0.570
    291 ESO/WFI-V89_843.raw +atm lambda_c= 5.3758e+03 AB-Vega=-0.014
    292 ESO/WFI-Rc162_844.raw +atm lambda_c= 6.4938e+03 AB-Vega=0.193
    293 DECam/DECam_u.txt lambda_c= 3.8566e+03 AB-Vega=0.340
    294 DECam/DECam_g.txt lambda_c= 4.8200e+03 AB-Vega=-0.090
    295 DECam/DECam_r.txt lambda_c= 6.4230e+03 AB-Vega=0.182
    296 DECam/DECam_i.txt lambda_c= 7.8067e+03 AB-Vega=0.406
    297 DECam/DECam_z.txt lambda_c= 9.1585e+03 AB-Vega=0.517
    298 DECam/DECam_Y.txt lambda_c= 9.8668e+03 AB-Vega=0.564
    299 MOSFIRE/mosfire_Y.txt lambda_c= 1.0474e+04 AB-Vega=0.640
    300 MOSFIRE/mosfire_J.txt lambda_c= 1.2517e+04 AB-Vega=0.913
    301 MOSFIRE/mosfire_H.txt lambda_c= 1.6340e+04 AB-Vega=1.347
    302 MOSFIRE/mosfire_Ks.txt lambda_c= 2.1468e+04 AB-Vega=1.821
    303 MOSFIRE/J2_center_corr.txt lambda_c= 1.1816e+04 AB-Vega=0.815
    304 MOSFIRE/J3_center_corr.txt lambda_c= 1.2881e+04 AB-Vega=0.969
    305 MOSFIRE/H1_center_corr.txt lambda_c= 1.5558e+04 AB-Vega=1.279
    306 MOSFIRE/H2_center_corr.txt lambda_c= 1.7089e+04 AB-Vega=1.423
    307 HST/wfpc2,1,f675w,f675w,a2d7.dat lambda_c= 6.7175e+03 AB-Vega=0.246
    308 niriss-f090w primary_area=244789.2cm2 ABZP=28.2710 lambda_c=0.9009um
    309 niriss-f115w primary_area=244789.2cm2 ABZP=28.3295 lambda_c=1.1495um
    310 niriss-f150w primary_area=244789.2cm2 ABZP=28.1836 lambda_c=1.4929um
    311 niriss-f200w primary_area=244789.2cm2 ABZP=28.2602 lambda_c=1.9926um
    312 niriss-f140m primary_area=244789.2cm2 ABZP=27.3944 lambda_c=1.4035um
    313 niriss-f158m primary_area=244789.2cm2 ABZP=27.4494 lambda_c=1.5857um
    314 g_HSC.txt http://cosmos.astro.caltech.edu/page/filterset lambda_c=4798.2 AB-Vega=-0.089
    315 r_HSC.txt http://cosmos.astro.caltech.edu/page/filterset lambda_c=6218.4 AB-Vega=0.223
    316 i_HSC.txt http://cosmos.astro.caltech.edu/page/filterset lambda_c=7727.0 AB-Vega=0.468
    317 z_HSC.txt http://cosmos.astro.caltech.edu/page/filterset lambda_c=8908.2 AB-Vega=0.522
    318 y_HSC.txt http://cosmos.astro.caltech.edu/page/filterset lambda_c=9775.1 AB-Vega=0.740
    319 NB816_HSC.txt http://cosmos.astro.caltech.edu/page/filterset lambda_c=8176.7 AB-Vega=0.476
    320 NB921_HSC.txt http://cosmos.astro.caltech.edu/page/filterset lambda_c=9213.2 AB-Vega=0.647
    321 VISTA-NB118 atm lambda_c= 1.1909e+04 AB-Vega= xxxx
    322 COSMOS/SUBARU_filter_NB711.txt lambda_c= 7.1202e+03 AB-Vega= xxxx
    323 scuba2/450 lambda_c= 4496191.778981589
    324 scuba2/850 lambda_c= 8543544.971510125
    325 mips/24 lambda_c= 2.37589e+05
    326 mips/70 lambda_c= 7.19852e+05
    327 mips/160 lambda_c= 1.56427e+06
    328 herschel/pacs/70 lambda_c= 7.15415e+05
    329 herschel/pacs/100 lambda_c= 1.02007e+06
    330 herschel/pacs/160 lambda_c= 1.65356e+06
    331 herschel/spire/200 lambda_c= 2.50429e+06
    332 herschel/spire/350 lambda_c= 3.51368e+06
    333 herschel/spire/500 lambda_c= 5.07844e+06
    334 PAN-STARRS/PS1.g lambda_c= 4.84911e+03 AB-Vega= -0.087
    335 PAN-STARRS/PS1.r lambda_c= 6.20119e+03 AB-Vega= 0.144
    336 PAN-STARRS/PS1.i lambda_c= 7.53496e+03 AB-Vega= 0.366
    337 PAN-STARRS/PS1.z lambda_c= 8.67418e+03 AB-Vega= 0.513
    338 PAN-STARRS/PS1.y lambda_c= 9.62777e+03 AB-Vega= 0.544
    339 HST/WFC3_UVIS1.F350LP lambda_c= 5.85733e+03 AB-Vega= 0.159
    340 HST/WFC3_UVIS1.F645N lambda_c= 6.45340e+03 AB-Vega= 0.196
    341 HST/WFC3_UVIS1.F657N lambda_c= 6.56659e+03 AB-Vega= 0.332
    342 HST/WFC3_UVIS1.F665N lambda_c= 6.65587e+03 AB-Vega= 0.243
    343 HST/WFC3_UVIS1.F673N lambda_c= 6.76601e+03 AB-Vega= 0.245
    344 HST/WFC3_IR.F128N lambda_c= 1.28366e+04 AB-Vega= 1.047
    345 HST/WFC3_IR.F130N lambda_c= 1.30104e+04 AB-Vega= 0.984
    346 HST/WFC3_IR.F132N lambda_c= 1.31935e+04 AB-Vega= 1.005
    347 HST/WFC3_IR.F164N lambda_c= 1.64500e+04 AB-Vega= 1.401
    348 CFHT_Megaprime.u_sdss.dat-cfh9302 lambda_c= 3.54017e+03
    349 hst/wfc3,uvis2,f200lp lambda_c= 4.88918e+03
    350 jwst_niriss_f090w v20160902163017 lambda_c= 9.0247e+03
    351 jwst_niriss_f115w v20160902163017 lambda_c= 1.1496e+04
    352 jwst_niriss_f150w v20160902163017 lambda_c= 1.4935e+04
    353 jwst_niriss_f200w v20160902163017 lambda_c= 1.9930e+04
    354 jwst_niriss_f140m v20160902163017 lambda_c= 1.4040e+04
    355 jwst_niriss_f158m v20160902163017 lambda_c= 1.5819e+04
    356 jwst_niriss_f277w v20160902163017 lambda_c= 2.7643e+04
    357 jwst_niriss_f356w v20160902163017 lambda_c= 3.5930e+04
    358 jwst_niriss_f444w v20160902163017 lambda_c= 4.4277e+04
    359 jwst_niriss_f380m v20160902163017 lambda_c= 3.8252e+04
    360 jwst_niriss_f430m v20160902163017 lambda_c= 4.2838e+04
    361 jwst_niriss_f480m v20160902163017 lambda_c= 4.8152e+04
    362 jwst_nircam_f070w v20160902164019 lambda_c= 7.0432e+03
    363 jwst_nircam_f090w v20160902164019 lambda_c= 9.0229e+03
    364 jwst_nircam_f115w v20160902164019 lambda_c= 1.1543e+04
    365 jwst_nircam_f150w v20160902164019 lambda_c= 1.5007e+04
    366 jwst_nircam_f200w v20160902164019 lambda_c= 1.9886e+04
    367 jwst_nircam_f150w2 v20160902164019 lambda_c= 1.6589e+04
    368 jwst_nircam_f140m v20160902164019 lambda_c= 1.4054e+04
    369 jwst_nircam_f162m v20160902164019 lambda_c= 1.6272e+04
    370 jwst_nircam_f182m v20160902164019 lambda_c= 1.8452e+04
    371 jwst_nircam_f210m v20160902164019 lambda_c= 2.0955e+04
    372 jwst_nircam_f164n v20160902164019 lambda_c= 1.6446e+04
    373 jwst_nircam_f187n v20160902164019 lambda_c= 1.8739e+04
    374 jwst_nircam_f212n v20160902164019 lambda_c= 2.1212e+04
    375 jwst_nircam_f277w v20160902164019 lambda_c= 2.7623e+04
    376 jwst_nircam_f356w v20160902164019 lambda_c= 3.5682e+04
    377 jwst_nircam_f444w v20160902164019 lambda_c= 4.4037e+04
    378 jwst_nircam_f322w2 v20160902164019 lambda_c= 3.2318e+04
    379 jwst_nircam_f250m v20160902164019 lambda_c= 2.5037e+04
    380 jwst_nircam_f300m v20160902164019 lambda_c= 2.9895e+04
    381 jwst_nircam_f335m v20160902164019 lambda_c= 3.3623e+04
    382 jwst_nircam_f360m v20160902164019 lambda_c= 3.6243e+04
    383 jwst_nircam_f410m v20160902164019 lambda_c= 4.0821e+04
    384 jwst_nircam_f430m v20160902164019 lambda_c= 4.2813e+04
    385 jwst_nircam_f460m v20160902164019 lambda_c= 4.6302e+04
    386 jwst_nircam_f480m v20160902164019 lambda_c= 4.8156e+04
    387 jwst_nircam_f323n v20160902164019 lambda_c= 3.2369e+04
    388 jwst_nircam_f405n v20160902164019 lambda_c= 4.0517e+04
    389 jwst_nircam_f466n v20160902164019 lambda_c= 4.6544e+04
    390 jwst_nircam_f470n v20160902164019 lambda_c= 4.7078e+04
    391 jwst_miri_f1065c v20171115151044 lambda_c= 1.0595e+05
    392 jwst_miri_f1140c v20171115151044 lambda_c= 1.1304e+05
    393 jwst_miri_f1550c v20171115151044 lambda_c= 1.5512e+05
    394 jwst_miri_f2300c v20171115151044 lambda_c= 2.2661e+05
    395 jwst_miri_f0560w v20171115151044 lambda_c= 5.6326e+04
    396 jwst_miri_f0770w v20171115151044 lambda_c= 7.6364e+04
    397 jwst_miri_f1000w v20171115151044 lambda_c= 9.9471e+04
    398 jwst_miri_f1130w v20171115151044 lambda_c= 1.1309e+05
    399 jwst_miri_f1280w v20171115151044 lambda_c= 1.2809e+05
    400 jwst_miri_f1500w v20171115151044 lambda_c= 1.5048e+05
    401 jwst_miri_f1800w v20171115151044 lambda_c= 1.7970e+05
    402 jwst_miri_f2100w v20171115151044 lambda_c= 2.0801e+05
    403 jwst_miri_f2550w v20171115151044 lambda_c= 2.5230e+05
    404 wfc3,ir,f139m lambda_c= 1.38376e+04
    405 hst/wfc3,uvis2,f621m lambda_c= 6.21894e+03
    406 Gemini F2 K-blue 4.3mm H20 & 1.5am mirrors + QE lambda_c= 2.0742e+04 AB-Vega= 1.767 w95=2294.5
    407 Gemini F2 K-red 4.3mm H20 & 1.5am mirrors + QE lambda_c= 2.3046e+04 AB-Vega= 1.959 w95=2593.2

Defining new filters
~~~~~~~~~~~~~~~~~~~~
Additional filters can be added to the main filter file.  There is a helper to generate the appropriate format for a filter defined from arrays:

    >>> wx = np.arange(5400, 5600., 5)
    >>> wy = wx*0.
    >>> wy[10:-10] = 1
    >>> f1 = filters.FilterDefinition(wave=wx, throughput=wy, name='Tophat 5500')
    >>> print(f1.for_filter_file())
    40 Tophat 5500 lambda_c= 5.4974e+03 AB-Vega= 0.016 w95=95.0
         1 5.40000e+03 0.00000e+00
         2 5.40500e+03 0.00000e+00
         3 5.41000e+03 0.00000e+00
         4 5.41500e+03 0.00000e+00
         5 5.42000e+03 0.00000e+00
         6 5.42500e+03 0.00000e+00
         7 5.43000e+03 0.00000e+00
         8 5.43500e+03 0.00000e+00
         9 5.44000e+03 0.00000e+00
        10 5.44500e+03 0.00000e+00
        11 5.45000e+03 1.00000e+00
        12 5.45500e+03 1.00000e+00
        13 5.46000e+03 1.00000e+00
        14 5.46500e+03 1.00000e+00
        15 5.47000e+03 1.00000e+00
        16 5.47500e+03 1.00000e+00
        17 5.48000e+03 1.00000e+00
        18 5.48500e+03 1.00000e+00
        19 5.49000e+03 1.00000e+00
        20 5.49500e+03 1.00000e+00
        21 5.50000e+03 1.00000e+00
        22 5.50500e+03 1.00000e+00
        23 5.51000e+03 1.00000e+00
        24 5.51500e+03 1.00000e+00
        25 5.52000e+03 1.00000e+00
        26 5.52500e+03 1.00000e+00
        27 5.53000e+03 1.00000e+00
        28 5.53500e+03 1.00000e+00
        29 5.54000e+03 1.00000e+00
        30 5.54500e+03 1.00000e+00
        31 5.55000e+03 0.00000e+00
        32 5.55500e+03 0.00000e+00
        33 5.56000e+03 0.00000e+00
        34 5.56500e+03 0.00000e+00
        35 5.57000e+03 0.00000e+00
        36 5.57500e+03 0.00000e+00
        37 5.58000e+03 0.00000e+00
        38 5.58500e+03 0.00000e+00
        39 5.59000e+03 0.00000e+00
        40 5.59500e+03 0.00000e+00

API
~~~
.. automodapi:: eazy.filters
    :no-inheritance-diagram:
