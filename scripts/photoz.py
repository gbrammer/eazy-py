def rest_frame_seds_selections():
    """
    """
    ok = (zout['z_phot'] > 0.4) & (zout['z_phot'] < 2)
    col = (VJ < 1.5) & (UV > 1.5)
    # Quiescent
    idx = col & ok & (np.log10(sSFR) < -11.5)
    idx = col & ok & (np.log10(sSFR) > -10.5)
    idx = col & ok & (np.log10(sSFR) > -9.5)

    idx = ok & (VJ > 1.8)
     
    ## Red
    UWise = f_rest[:,0,2]/f_rest[:,2,2]
    idx, label, c = ok & (np.log10(UWise) > -1) & (np.log10(sSFR) > -10), 'U22_blue', 'b'

    idx, label, c = ok & (np.log10(UWise) < -1.8) & (np.log10(UWise) > -2.2) & (np.log10(sSFR) > -10), 'U22_mid', 'g'

    idx, label, c = ok & (np.log10(UWise) < -2.4) & (np.log10(sSFR) > -10), 'U22_red', 'r'
    
    # Quiescent
    idx, label, c = ok & (np.log10(zout['MLv']) > 0.4) & (np.log10(sSFR) < -11.9), 'Q', 'r'
    
    # Dusty
    idx, label, c = ok & (np.log10(zout['MLv']) > 0.6) & (np.log10(sSFR) < -10.5), 'MLv_lo', 'brown'

    idx, label, c = ok & (np.log10(zout['MLv']) > 0.6) & (np.abs(np.log10(sSFR)+10.5) < 0.5), 'MLv_mid', 'k'

    idx, label, c = ok & (np.log10(zout['MLv']) > 0.6) & (np.log10(sSFR) > -9.5), 'MLv_hi', 'green'
    
    # post-SB    
    #idx, label, c = (UV < 1.6) & ok & (np.log10(sSFR) < -11) & (VJ < 1), 'post-SB', 'orange'
    
    # star-forming    
    idx, label, c = ok & (UV < 0.6) & (VJ < 0.5), 'SF0', 'purple'
    
    idx, label, c = ok & (np.abs(UV-0.8) < 0.2) & (np.abs(VJ-0.6) < 0.2), 'SF1', 'b'

    idx, label, c = ok & (np.abs(UV-1.2) < 0.2) & (np.abs(VJ-1.0) < 0.2), 'SF2', 'orange'

    idx, label, c = ok & (np.abs(UV-1.6) < 0.2) & (np.abs(VJ-1.6) < 0.2), 'SF3', 'pink'    