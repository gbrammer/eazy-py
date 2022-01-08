import numpy as np
import os

from astropy.table import Table
from . import utils

__all__ = ["FilterDefinition", "FilterFile", "ParamFilter"]

VEGA_FILE = os.path.join(utils.path_to_eazy_data(),
                         'alpha_lyr_stis_008.fits')
                         
VEGA = Table.read(VEGA_FILE)
for c in VEGA.colnames:
    VEGA[c] = VEGA[c].astype(float)
    
class FilterDefinition:
    def __init__(self, name=None, wave=None, throughput=None, bp=None, photon_counter=True):
        """
        Bandpass object
        
        Parameters
        ----------
        name : str
            Label name
        
        wave : array
            Wavelength array, in `astropy.units.Angstrom`.
        
        throughput : array
            Throughput, arbitrary normalization
        
        bp : optional, `pysynphot.obsbandpass` object
            `pysynphot` filter bandpass
        
        photon_counter : bool
            Filter is associated with a photon-counting detector (e.g. CCDs).  
            Often FIR devices are energy-counting, e.g., MIPS, Herschel.
            
        """
        self.name = name
        self.wave = wave
        self._throughput = throughput
        self.Aflux = 1.
        self.photon_counter = photon_counter

        if isinstance(name, str):
            # Set energy-counting filters
            ir_filts = ['mips/24','mips/70','mips/160',
                        'scuba2/450','scuba2/850',
                        'herschel/pacs/70', 'herschel/pacs/100', 
                        'herschel/pacs/160', 'herschel/spire/200', 
                        'herschel/spire/350', 'herschel/spire/500']
            
            if name.split()[0] in ir_filts:
                self.photon_counter = False

        # pysynphot Bandpass
        if bp is not None:
            self.wave = np.cast[np.double](bp.wave)
            self._throughput =  np.cast[np.double](bp.throughput)
            self.name = bp.name
        
        # Set throughput accounting for photon_counter
        self.set_throughput()


    def set_throughput(self):
        """
        Set throughput accounting for `photon_counter` attribute
        """
        if self._throughput is None:
            self.throughput = None
            self.norm = 1.
        else:
            if self.photon_counter:
                self.throughput = self._throughput
            else:
                _wfact = self.wave/np.mean(self.wave)
                self.throughput =  self._throughput/_wfact
            
            self.norm = np.trapz(self.throughput/self.wave, self.wave)


    def __repr__(self):
        return self.name.__repr__()


    def __str__(self):
        return self.name.__str__()


    def get_extinction(self, EBV=0, Rv=3.1):
        """
        Extinction factor 
        """
        import astropy.units as u
        
        f99 = utils.GalacticExtinction(EBV=EBV, Rv=Rv)
        self.Alambda = f99(self.wave)
        self.Aflux = 10**(-0.4*self.Alambda)


    def extinction_correction(self, EBV, Rv=3.1, mag=True, source_lam=None, source_flux=None):
        """
        Get the MW extinction correction within the filter.  
        
        Optionally supply a source spectrum.
        """
        import astropy.units as u
        try:
            import grizli.utils_c
            interp = grizli.utils_c.interp.interp_conserve_c
        except ImportError:
            interp = utils.interp_conserve
             
        if self.wave is None:
            print('Filter not defined.')
            return False
        
        if source_flux is None:
            src = self.throughput*0.+1
        else:
            src = interp(self.wave, source_lam, source_flux, left=0, right=0)
        
        if (self.wave.min() < 910) | (self.wave.max() > 6.e4):
            Alambda = 0.
        else:
            f99 = utils.GalacticExtinction(EBV=EBV, Rv=Rv)
            Alambda = f99(self.wave)
        
        src_red = np.trapz(self.throughput*src*10**(-0.4*Alambda)/self.wave, 
                           self.wave)               
        src_nored = np.trapz(self.throughput*src/self.wave, self.wave)
        
        delta = src_red/src_nored
        
        if mag:
            return 2.5*np.log10(delta)
        else:
            return 1./delta


    @property    
    def ABVega(self):
        """
        Compute AB-Vega conversion
        """
        from astropy.constants import c
        import astropy.units as u
        try:
            import grizli.utils_c
            interp = grizli.utils_c.interp.interp_conserve_c
        except ImportError:
            interp = utils.interp_conserve
        
        # Union of throughput and Vega spectrum arrays
        full_x = np.hstack([self.wave, VEGA['WAVELENGTH']])
        full_x = full_x[np.argsort(full_x)]

        # Vega spectrum, units of f-lambda flux density, cgs
        # Interpolate to wavelength grid, no extrapolation
        vega_full = interp(full_x, VEGA['WAVELENGTH'], VEGA['FLUX'], 
                              left=0, right=0)
                              
        thru_full = interp(full_x, self.wave, self.throughput, 
                              left=0, right=0)        
        
        # AB = 0, same units
        absp = 3631*1e-23*c.to(u.m/u.s).value*1.e10/full_x**2
        
        # Integrate over the bandpass, flam dlam
        num = np.trapz(vega_full*thru_full, full_x)
        den = np.trapz(absp*thru_full, full_x)
        
        return -2.5*np.log10(num/den)


    @property    
    def pivot(self):
        """
        Pivot wavelength
        
        http://pysynphot.readthedocs.io/en/latest/properties.html
        """
        integrator = np.trapz
        
        num = integrator(self.wave, self.wave*self.throughput)
        den = integrator(self.wave, self.throughput/self.wave)
        pivot = np.sqrt(num/den)
        return pivot
    
    @property    
    def equivwidth(self):
        """
        Filter equivalent width

        http://pysynphot.readthedocs.io/en/latest/properties.html
        """
        return np.trapz(self.throughput, self.wave)
    
    
    @property            
    def rectwidth(self):
        """
        Filter rectangular width

        http://pysynphot.readthedocs.io/en/latest/properties.html
        """
        
        rect = self.equivwidth / self.throughput.max()
        return rect
    
    
    @property    
    def ctw95(self):
        """
        95% cumulative throughput width
        
        http://www.stsci.edu/hst/acs/analysis/bandwidths/#keywords
        
        """
        
        dl = np.diff(self.wave)
        filt = np.cumsum((self.wave*self.throughput)[1:]*dl)
        ctw95 = np.interp([0.025, 0.975], filt/filt.max(), self.wave[1:])
        return np.diff(ctw95)[0]
    
    
    def for_filter_file(self, row_str='{i:6} {wave:.5e} {thru:.5e}'):
        """
        Return a string that can be put in the EAZY filter file
        """    
        header = '{0} {1} lambda_c= {2:.4e} AB-Vega= {3:.3f} w95={4:.1f}'
        N = len(self.wave)
        lines = [header.format(N, self.name.split('lambda_c')[0], 
                               self.pivot, self.ABVega, self.ctw95)]
        
        lines += [row_str.format(i=i+1, wave=w, thru=t)
                  for i, (w, t) in enumerate(zip(self.wave, self.throughput))]
        
        return '\n'.join(lines)


class FilterFile:
    def __init__(self, file='FILTER.RES.latest', path='./'):
        """
        Read a EAZY filter file.
        
        .. plot::
            :include-source:
        
            import matplotlib.pyplot as plt
            from eazy.filters import FilterFile
            
            res = FilterFile(path=None)
            print(len(res.filters))
            
            bp = res[205]
            print(bp)
            
            fig, ax = plt.subplots(1,1,figsize=(6,4))
            
            ax.plot(bp.wave, bp.throughput, label=bp.name.split()[0])
            
            ax.set_xlabel('wavelength, Angstroms')
            ax.set_ylabel('throughput')
            ax.legend()
            ax.grid()
            
            fig.tight_layout(pad=0.5)
            
        
        """
        if path is None:
            file_path = os.path.join(os.getenv('EAZYCODE'), 'filters', file)
        else:
            file_path = os.path.join(path, file)
            
        with open(file_path, 'r') as fp:
            lines = fp.readlines()
        
        self.filename = file_path
        
        filters = []
        wave = []
        trans = []
        header = ''
        
        for line in lines:
            if 'lambda_c' in line:
                if len(wave) > 0:
                    # Make filter from lines already read in
                    new_filter = FilterDefinition(name=header,
                                                  wave=np.cast[float](wave), 
                                            throughput=np.cast[float](trans))
                    # new_filter.name = header
                    # new_filter.wave = np.cast[float](wave)
                    # new_filter.throughput = np.cast[float](trans)
                    filters.append(new_filter)

                # Initialize filter
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


    @property 
    def NFILT(self):
        """
        Number of filters in the list
        """
        return len(self.filters)


    def __getitem__(self, i1):
        """
        Return unit-indexed filter, e.g., 161 = 2mass-j
        """
        return self.filters[i1-1]


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


    def search(self, search_string, case=False, verbose=True):
        """ 
        Search filter names for ``search_string``.  If ``case`` is True, then
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
