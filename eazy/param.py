import os
import collections
import numpy as np

from .filters import FilterDefinition, FilterFile, ParamFilter
from .templates import TemplateError, Template

__all__ = ["EazyParam", "TranslateFile"]

class EazyParam():
    """
    Read an Eazy zphot.param file.
    
    Example: 
    
    >>> params = EazyParam(PARAM_FILE='zphot.param')
    >>> params['Z_STEP']
    '0.010'

    """    
    def __init__(self, PARAM_FILE=None, read_filters=False,
                 read_templates=False):
        
        if PARAM_FILE is None:
            PARAM_FILE = os.path.join(os.path.dirname(__file__), 'data/zphot.param.default')
            print('Read default param file: '+PARAM_FILE)
            
        self.filename = PARAM_FILE
        self.param_path = os.path.dirname(PARAM_FILE)
        
        f = open(PARAM_FILE,'r')
        self.lines = f.readlines()
        f.close()
        
        self._process_params()
        
        filters = []
        templates = []
        for line in self.lines:
            if line.startswith('#  Filter'):
                filters.append(ParamFilter(line))
            if line.startswith('#  Template'):
                templates.append(line.split()[3])
                
        self.NFILT = len(filters)
        self.filters = filters
        self.template_files = templates
        
        if read_filters:
            RES = FilterFile(self.params['FILTERS_RES'])
            for i in range(self.NFILT):
                filters[i].wave = RES.filters[filters[i].fnumber-1].wave
                filters[i].throughput = RES.filters[filters[i].fnumber-1].throughput
        
        if read_templates:
            self.templates = self.read_templates(templates_file=self.params['TEMPLATES_FILE'])
            
    def read_templates(self, templates_file=None):
        
        lines = open(templates_file).readlines()
        templates = []
        
        for line in lines:
            if line.strip().startswith('#'):
                continue
            
            template_file = line.split()[1]
            templ = Template(file=template_file)
            templ.wave *= float(line.split()[2])
            templ.set_fnu()
            templates.append(templ)
        
        return templates
            
    def show_templates(self, interp_wave=None, ax=None, fnu=False):
        if ax is None:
            ax = plt
        
        for templ in self.templ:
            if fnu:
                flux = templ.flux_fnu
            else:
                flux = templ.flux
                
            if interp_wave is not None:
                y0 = np.interp(interp_wave, templ.wave, flux)
            else:
                y0 = 1.
            
            plt.plot(templ.wave, flux / y0, label=templ.name)
            
    def _process_params(self):
        params = collections.OrderedDict()
        formats = collections.OrderedDict()
        self.param_names = []
        for line in self.lines:
            if line.strip().startswith('#') is False:
                lsplit = line.split()
                if lsplit.__len__() >= 2:
                    params[lsplit[0]] = lsplit[1]
                    self.param_names.append(lsplit[0])
                    try:
                        flt = float(lsplit[1])
                        formats[lsplit[0]] = 'f'
                        params[lsplit[0]] = flt
                    except:
                        formats[lsplit[0]] = 's'
                    
        self.params = params
        #self.param_names = params.keys()
        self.formats = formats
    
    def list_filters(self):
        for filter in self.filters:
            print(' F{0:d}, {1}, lc={2}'.format(filter.fnumber, filter.name, filter.lambda_c))

    def to_mJy(self):
        """
        Return conversion factor to mJy
        """
        return 10**(-0.4*(self.params['PRIOR_ABZP']-23.9))/1000.
        
    def write(self, file=None):
        if file == None:
            print('No output file specified...')
        else:
            fp = open(file,'w')
            for param in self.param_names:
                if isinstance(self.params[param], np.str):
                    fp.write('{0:25s} {1}\n'.format(param, self.params[param]))
                else:
                    fp.write('{0:25s} {1}\n'.format(param, self.params[param]))
                    #str = '%-25s %'+self.formats[param]+'\n'
            #
            fp.close()
            
    def __getitem__(self, param_name):
        """
    __getitem__(param_name)

        >>> cat = mySexCat('drz.cat')
        >>> print cat['NUMBER']

        """
        if param_name not in self.param_names:
            print('Column {0} not found.  Check `column_names` attribute.'.format(param_name))
            return None
        else:
            #str = 'out = self.%s*1' %column_name
            #exec(str)
            return self.params[param_name]
    
    def __setitem__(self, param_name, value):
        self.params[param_name] = value
    
class TranslateFile():
    def __init__(self, file='zphot.translate'):
        self.file=file
        self.ordered_keys = []
        lines = open(file).readlines()
        self.trans = collections.OrderedDict()
        self.error = collections.OrderedDict()
        for line in lines:
            spl = line.split()
            key = spl[0]
            self.ordered_keys.append(key)
            self.trans[key] = spl[1]
            if len(spl) == 3:
                self.error[key] = float(spl[2])
            else:
                self.error[key] = 1.
            #
            
    def change_error(self, filter=88, value=1.e8):
        
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
