import os
import collections
import numpy as np

__all__ = ["EazyParam", "TranslateFile", "read_param_file"]

def read_param_file(param_file=None, verbose=True):
    """
    Load a param file and add default parameters if any missing
    """
    param = EazyParam(param_file, verbose=True)
    if param_file is not None:
        # Read defaults
        defaults = EazyParam(None, verbose=False)
        for k in defaults.param_names:
            if k not in param.param_names:
                param[k] = defaults[k]
                if verbose:
                    print(f'Parameter default: {k} = {defaults[k]}')

    return param


class EazyParam(object):
    def __init__(self, PARAM_FILE=None, verbose=True):
        """
        Read an Eazy zphot.param file.

        Example: 

            >>> if os.path.exists('zphot.param'):
            ...     params = EazyParam(PARAM_FILE='zphot.param')
            ...     print(params['Z_STEP'])

        Defaults are in `eazy/data/zphot.param.default <https://github.com/gbrammer/eazy-py/blob/master/eazy/data/zphot.param.default>`_
        
        Parameters
        ----------
        param_file : str
            Name of parameter file.  If None, then will get 
            `data/zphot.param.default` from within the module.
        
        Attributes
        ----------
        params : `collections.OrderedDict`
            Parameter dictionary.  Don't modify this directly but rather use
            `__getitem__` and `__setitem__` methods.
        
        param_names
        
        formats : list
            List indicating if parameters are interpreted as string ('s') or 
            scalar ('f') values.  
            
        """
        if PARAM_FILE is None:
            PARAM_FILE = os.path.join(os.path.dirname(__file__), 
                                      'data/zphot.param.default')
        
        if verbose:
            print('Read default param file: '+PARAM_FILE)
            
        self.filename = PARAM_FILE
        self.param_path = os.path.dirname(PARAM_FILE)
           
        f = open(PARAM_FILE,'r')
        self.lines = f.readlines()
        f.close()

        self.params = collections.OrderedDict()
        self.formats = collections.OrderedDict()
        
        self._process_params()


    @property 
    def param_names(self):
        """
        Keywords of the `params` dictionary
        """
        return list(self.params.keys())


    def _process_params(self):
        """
        Process parameter dictionary
        """
        params = collections.OrderedDict()
        formats = collections.OrderedDict()
        
        #self.param_names = []
        for line in self.lines:
            if not line.strip().startswith('#'):
                lsplit = line.split()
                if lsplit.__len__() >= 2:
                    params[lsplit[0]] = lsplit[1]
                    #self.param_names.append(lsplit[0])
                    try:
                        flt = float(lsplit[1])
                        formats[lsplit[0]] = 'f'
                        params[lsplit[0]] = flt
                    except:
                        formats[lsplit[0]] = 's'
                    
        self.params = params
        self.formats = formats


    @property
    def to_mJy(self):
        """
        Return catalog conversion factor to mJy based on ``PRIOR_ABZP``.
        """
        return 10**(-0.4*(self.params['PRIOR_ABZP']-23.9))/1000.


    def write(self, file=None):
        """
        Write to an ascii file
        """
        if file == None:
            print('No output file specified...')
        else:
            fp = open(file,'w')
            for param in self.param_names:
                fp.write('{0:25s} {1}\n'.format(param, self.params[param]))

            fp.close()


    def __getitem__(self, param_name):
        """
        Get item from ``params`` dict and return None if parameter not found.
        """
        if param_name.upper() not in self.param_names:
            print(f'Parameter {param_name} not found.  Check `param_names`'              
                    ' attribute.')
            return None
        else:
            return self.params[param_name.upper()]


    def __setitem__(self, param_name, value):
        """
        Set item in ``params`` dict.
        """
        self.params[param_name.upper()] = value


    def verify_params(self):
        """
        Some checks on the parameters
        """
        
        assert(self['Z_MAX'] > self['Z_MIN'])
        
        for k in ['TEMPLATES_FILE', 'TEMP_ERR_FILE', 'CATALOG_FILE', 
                  'FILTERS_RES']:
            if isinstance(self[k], str):
                if not os.path.exists(self[k]):
                    raise FileNotFoundError(f'{k} ({self[k]}) not found')        

        assert(int(self['ARRAY_NBITS']) in [32,64])
        
        # Positive
        for k in ['TEMP_ERR_A2', 'SYS_ERR', 'IGM_SCALE_TAU', 'MW_EBV', 
                  'OMEGA_M', 'OMEGA_L']:
            if self[k] < 0:
                raise ValueError(f'{k} ({self[k]}) must be >= 0')
        
        # Positive nonzero
        for k in ['Z_STEP','H0', 'RF_PADDING']:
            if self[k] < 0:
                raise ValueError(f'{k} ({self[k]}) must be > 0')


    @property 
    def kwargs(self):
        """
        Dictionary with lower-case parameter names for passing as ``**kwargs``
        """
        kws = collections.OrderedDict()
        for k in self.param_names:
            kws[k.lower()] = self.params[k]
        
        return kws
        
        
class TranslateFile():
    def __init__(self, file='zphot.translate'):
        """
        File for translating catalog columns to associate bandbasses to them
        
        The `file` has format
                
        .. code-block::

            flux_irac_ch1  F18
            err_irac_ch1   E18
            ...
            
        or a CSV table with format
        
        .. code-block::
        
            column, trans, error
            flux_irac_ch1, F18
            err_irac_ch1,  E18, 1.0
            ...
        
        where `flux_irac_ch1` is a column in the catalog table corresponding 
        to the IRAC 3.6 µm flux density. ``F18`` indicates that this is a 
        *flux density* column and is associated with filter number 18 in the 
        `~eazy.params.filters.FilterFile`.
        
        ``E18`` indicates an uncertainty column, and filters must have both 
        flux density and uncertainty columns to be considered.
        
        The original catalog could have had column names ``F18`` and ``E18``
        and not needed a translate file but it is generally preferable to have
        more descriptive column names that aren't necessarily tied to a
        particular `eazy` filter file.
        
        Note, similarly, that columns like `F{N}` and `E{N}` are treated as 
        these types of flux and uncertainty columns.  If they correspond to 
        something else, they should be "translated" to avoid confusion
        
        """
        from astropy.table import Table
        
        self.file = file
        self.trans = collections.OrderedDict()
        self.error = collections.OrderedDict()

        if hasattr(file, 'colnames'):
            tr = file
            self.file = 'input_table.translate'
        
            if 'error' not in tr.colnames:
                tr['error'] = 1.0
        
            if tr.colnames != ['column', 'trans', 'error']:
                msg = f"table translate_file file must have columns"
                msg += f" 'column', 'trans' [, 'error'].  The file {file}"
                msg += f' has columns {tr.colnames}.'
                raise ValueError(msg)
        
            for i, k in enumerate(tr['column']):
                self.trans[k] = tr['trans'][i]
                self.error[k] = tr['error'][i]

        elif file.endswith('csv'):
            tr = Table.read(file)
                
            if 'error' not in tr.colnames:
                tr['error'] = 1.0
            
            if tr.colnames != ['column', 'trans', 'error']:
                msg = f"csv translate_file file must have columns"
                msg += f" 'column', 'trans' [, 'error'].  The file {file}"
                msg += f' has columns {tr.colnames}.'
                raise ValueError(msg)
            
            for i, k in enumerate(tr['column']):
                self.trans[k] = tr['trans'][i]
                self.error[k] = tr['error'][i]
                            
        else:
            lines = open(file).readlines()

            for line in lines:
                spl = line.split()
                if (line.strip() == '') | (len(spl) < 2):
                    continue
            
                key = spl[0]
                self.trans[key] = spl[1]
                if len(spl) == 3:
                    self.error[key] = float(spl[2])
                else:
                    self.error[key] = 1.


    def change_error(self, filter=88, value=1.e8):
        """
        Modify uncertainties based on error scaling factors in translate file
        """
        if isinstance(filter, str):
            if 'f_' in filter:
                err_filt = filter.replace('f_','e_')
            else:
                err_filt = 'e'+filter

            if err_filt in self.error:
                self.error[err_filt] = value
                return True
        
        if isinstance(filter, int):
            for key in self.trans.keys():
                if self.trans[key] == 'E{0:0d}'.format(filter):
                    self.error[key] = value
                    return True
        
        print('Filter {0} not found in list.'.format(str(filter)))


    def write(self, file=None, show_ones=False):
        """
        Write to an ascii file
        """
        lines = []
        for key in self.error:
            line = '{0}  {1}'.format(key, self.trans[key])
            if self.trans[key].startswith('E') & ((self.error[key] != 1.0) | show_ones):
                line += '  {0:.1f}'.format(self.error[key])

            lines.append(line+'\n')

        if file is None:
            file = self.file
        
        if file:
            fp = open(file,'w')
            fp.writelines(lines)
            fp.close()
        else:
            for line in lines:
                print(line[:-1])


    def to_csv(self):
        """
        Generate CSV string
        """
        rows = 'column,trans,error\n'
        for k in self.error:
            rows += f'{k},{self.trans[k]},{self.error[k]}\n'
        return rows
        