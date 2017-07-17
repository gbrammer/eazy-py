import numpy as np
import os

__all__ = ["FilterDefinition", "FilterFile", "ParamFilter"]

class FilterDefinition:
    def __init__(self, name=None, wave=None, throughput=None, bp=None, EBV=0, Rv=3.1):
        """
        Placeholder for the filter definition information.
        """
        self.name = name
        self.wave = wave
        self.throughput = throughput
        self.Aflux = 1.
        
        # pysynphot Bandpass
        if bp is not None:
            self.wave = np.cast[np.double](bp.wave)
            self.throughput =  np.cast[np.double](bp.throughput)
            self.name = bp.name
            
            #self.get_extinction(EBV=EBV, Rv=Rv)
        
        self.norm = 1.
        if self.throughput is not None:
            self.norm = np.trapz(self.throughput/self.wave, self.wave)
        
    def get_extinction(self, EBV=0, Rv=3.1):
        try:
            import specutils.extinction
            import astropy.units as u
            HAS_SPECUTILS = True
        except:
            HAS_SPECUTILS = False
        
        Av = EBV*Rv
        if HAS_SPECUTILS:
            f99 = specutils.extinction.ExtinctionF99(EBV*Rv)
            self.Alambda = f99(self.wave*u.angstrom)
        else:
            self.Alambda = milkyway_extinction(lamb=self.wave, Rv=Rv)*Av
        
        self.Aflux = 10**(-0.4*self.Alambda)
        
    def extinction_correction(self, EBV, Rv=3.1, mag=True, source_lam=None, source_flux = None):
        """
        Get the MW extinction correction within the filter.  
        
        Optionally supply a source spectrum.
        """
        try:
            import specutils.extinction
            import astropy.units as u
            HAS_SPECUTILS = True
        except:
            HAS_SPECUTILS = False
             
        if self.wave is None:
            print('Filter not defined.')
            return False
        
        if source_flux is None:
            source_flux = self.throughput*0.+1
        else:
            source_flux = np.interp(self.wave, source_lam, source_flux, left=0, right=0)
            
        Av = EBV*Rv
        if HAS_SPECUTILS:
            f99 = specutils.extinction.ExtinctionF99(EBV*3.1)
            if (self.wave.min() < 910) | (self.wave.max() > 6.e4):
                Alambda = 0.
            else:
                Alambda = f99(self.wave*u.angstrom)
        else:
            Alambda = milkyway_extinction(lamb = self.wave, Rv=Rv)*Av
            
        delta = np.trapz(self.throughput*source_flux*10**(-0.4*Alambda), self.wave) / np.trapz(self.throughput*source_flux, self.wave)
        
        if mag:
            return 2.5*np.log10(delta)
        else:
            return 1./delta
    
    def ABVega(self):
        """
        Compute AB-Vega conversion
        """
        try:
            import pysynphot as S
        except:
            print('Failed to import "pysynphot"')
            return False
        
        vega=S.FileSpectrum(S.locations.VegaFile)
        abmag=S.FlatSpectrum(0,fluxunits='abmag')
        #xy, yy = np.loadtxt('hawki_y_ETC.dat', unpack=True)
        bp = S.ArrayBandpass(wave=self.wave, throughput=self.throughput, name='')
        ovega = S.Observation(vega, bp)
        oab = S.Observation(abmag, bp)
        return -2.5*np.log10(ovega.integrate()/oab.integrate())
    
    def pivot(self):
        """
        PySynphot pivot wavelength
        """
        try:
            import pysynphot as S
        except:
            print('Failed to import "pysynphot"')
            return False
            
        self.bp = S.ArrayBandpass(wave=self.wave, throughput=self.throughput, name='')
        return self.bp.pivot()
        
    def rectwidth(self):
        """
        Synphot filter rectangular width
        """
        try:
            import pysynphot as S
        except:
            print('Failed to import "pysynphot"')
            return False
            
        self.bp = S.ArrayBandpass(wave=self.wave, throughput=self.throughput, name='')
        return self.bp.rectwidth()

    def ctw95(self):
        """
        95% cumulative throughput width
        http://www.stsci.edu/hst/acs/analysis/bandwidths/#keywords
        
        """
        
        dl = np.diff(self.wave)
        filt = np.cumsum((self.wave*self.throughput)[1:]*dl)
        ctw95 = np.interp([0.025, 0.975], filt/filt.max(), self.wave[1:])
        return np.diff(ctw95)
            
        
class FilterFile:
    def __init__(self, file='FILTER.RES.v8.R300'):
        """
        Read a EAZY (HYPERZ) filter file.
        """
        fp = open(file)
        lines = fp.readlines()
        fp.close()
        
        filters = []
        wave = []
        for line in lines:
            if 'lambda_c' in line:
                if len(wave) > 0:
                    new_filter = FilterDefinition(name=header,
                                                  wave=np.cast[float](wave), 
                                            throughput=np.cast[float](trans))
                    # new_filter.name = header
                    # new_filter.wave = np.cast[float](wave)
                    # new_filter.throughput = np.cast[float](trans)
                    filters.append(new_filter)
                    
                header = ' '.join(line.split()[1:])
                wave = []
                trans = []
            else:
                lspl = np.cast[float](line.split())
                wave.append(lspl[1])
                trans.append(lspl[2])
        # last one
        # new_filter = FilterDefinition()
        # new_filter.name = header
        # new_filter.wave = np.cast[float](wave)
        # new_filter.throughput = np.cast[float](trans)
        new_filter = FilterDefinition(name=header,
                                      wave=np.cast[float](wave), 
                                throughput=np.cast[float](trans))

        filters.append(new_filter)
           
        self.filters = filters
        self.NFILT = len(filters)
    
    def names(self, verbose=True):
        """
        Print the filter names.
        """
        if verbose:
            for i in range(len(self.filters)):
                print('{0:5d} {1}'.format(i+1, self.filters[i].name))
        else:
            string_list = ['{0:5d} {1}\n'.format(i+1, self.filters[i].name) for i in range(len(self.filters))]
            return string_list
            
    def write(self, file='xxx.res', verbose=True):
        """
        Dump the filter information to a filter file.
        """
        fp = open(file,'w')
        for filter in self.filters:
            fp.write('{0:6d} {1}\n'.format(len(filter.wave), filter.name))
            for i in range(len(filter.wave)):
                fp.write('{0:6d} {1:.5e} {2:.5e}\n'.format(i+1, filter.wave[i], filter.throughput[i]))
        
        fp.close()
        
        string_list = self.names(verbose=False)
        fp = open(file+'.info', 'w')
        fp.writelines(string_list)
        fp.close()
        
        if verbose:
            print('Wrote <{0}[.info]>'.format(file))
            
    def search(self, search_string, case=True, verbose=True):
        """ 
        Search filter names for `search_string`.  If `case` is True, then
        match case.
        """
        import re
        
        if not case:
            search_string = search_string.upper()
        
        matched = []
        
        for i in range(len(self.filters)):
            filt_name = self.filters[i].name
            if not case:
                filt_name = filt_name.upper()
                
            if re.search(search_string, filt_name) is not None:
                if verbose:
                    print('{0:5d} {1}'.format(i+1, self.filters[i].name))
                matched.append(i)
        
        return np.array(matched)
        
class ParamFilter(FilterDefinition):
    def __init__(self, line='#  Filter #20, RES#78: COSMOS/SUBARU_filter_B.txt - lambda_c=4458.276253'):
        
        self.lambda_c = float(line.split('lambda_c=')[1])
        self.name = line.split()[4]
        self.fnumber = int(line.split('RES#')[1].split(':')[0])
        self.cnumber = int(line.split('Filter #')[1].split(',')[0])
