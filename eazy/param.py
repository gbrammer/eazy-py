import os
import collections
import numpy as np

__all__ = ["EazyParam", "TranslateFile", "load_param_file"]

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


class EazyParam():
    """
    Read an Eazy zphot.param file.
    
    Example: 
    
        >>> if os.path.exists('zphot.param'):
        ...     params = EazyParam(PARAM_FILE='zphot.param')
        ...     print(params['Z_STEP'])

    """    
    def __init__(self, PARAM_FILE=None, verbose=True):
        
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
        Keywords of the ``param`` dictionary
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
                if isinstance(self.params[param], str):
                    fp.write('{0:25s} {1}\n'.format(param, self.params[param]))
                else:
                    fp.write('{0:25s} {1}\n'.format(param, self.params[param]))
                    #str = '%-25s %'+self.formats[param]+'\n'
            #
            fp.close()


    def __getitem__(self, param_name):
        """
        Get item from ``params`` dict and return None if parameter not found.
        """
        if param_name not in self.param_names:
            print(f'Parameter {param_name} not found.  Check `param_names`'              
                    ' attribute.')
            return None
        else:
            return self.params[param_name]


    def __setitem__(self, param_name, value):
        """
        Set item in ``params`` dict.
        """
        # if param_name not in self.param_names:
        #     print('xxx append param', param_name)
        #     self.param_names.append(param_name)

        self.params[param_name] = value


class TranslateFile():
    def __init__(self, file='zphot.translate'):
        """
        File for translating filter columns for parsing as 
        """
        self.file=file
        self.ordered_keys = []
        lines = open(file).readlines()
        self.trans = collections.OrderedDict()
        self.error = collections.OrderedDict()
        for line in lines:
            spl = line.split()
            if (line.strip() == '') | (len(spl) < 2):
                continue
            
            key = spl[0]
            self.ordered_keys.append(key)
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

            if err_filt in self.ordered_keys:
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
        for key in self.ordered_keys:
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
