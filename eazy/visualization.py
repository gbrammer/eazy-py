"""
Scripts for more interactive visualization of SEDs, etc.
"""
import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from . import utils

__all__ = ['EazyExplorer']

CUTOUT_URL = "https://grizli-cutout.herokuapp.com/thumb?all_filters=True&size=4&scl=1.0&asinh=True&filters=f115w-clear,f277w-clear,f444w-clear&rgb_scl=1.5,0.74,1.3&pl=2&ra={ra}&dec={dec}"

EXTRA_SLIDER_KWS = {'column':'id', 'label':'ID',
                    'min':0, 'max':10, 'step':1,
                    'value':[1,5], 'checked':[],
                    }

class EazyExplorer(object):
    def __init__(self, photoz, zout, extra_zout_columns=[], selection=None, 
                 extra_plots={}, extra_slider_kws=EXTRA_SLIDER_KWS):
        """
        Generating a tool for interactive visualization of `eazy` outputs with 
        the `dash` + `plotly` libraries.
        
        Parameters
        ----------
        photoz : `~eazy.photoz.PhotoZ`
            The main ``PhotoZ`` object.
        
        zout : `astropy.table.Table`
            The `zout` output table with galaxy parameters.  At
            a minimum, it must have columns ``id``, ``z_phot``, ``z_spec``, 
            ``z_phot_chi2``, ``nusefilt``, ``restU``, ``restV``, ``restJ``, 
            ``sfr``, ``mass``, ``ra``, ``dec``.
        
        extra_zout_columns : list
            Additional columns from ``zout`` to copy to the app 
            ``df`` `pandas.DataFrame` object.
        
        selection : array-like
            Selection array on `zout` for catalog subset, can be integer 
            indices, or a boolean array with the same length as `zout`
        
        extra_plots : dict
            Extra scatter plot definitions following
            
            >>> extra_plots = {'PlotName': (xcol, ycol, xlabel, ylabel, xr, yr)}
            
            where ``xcol`` and ``ycol`` are `str` column names in `zout`,
            ``xlabel`` and ``ylabel`` are `str` used for the plot labels and
            ``xr`` and ``yr`` are the default plot range tuples. Only
            ``xcol`` and ``ycol`` are required, and the others are computed if
            not provided.  Note that ``xcol`` and ``ycol`` are added 
            automatically from `zout` if necessary and don't need to be 
            specified in `extra_zout_columns`.
            
        """
        import pandas as pd
        from astropy.table import Table
        
        try:
            import dash
            from dash import  dcc, html
            import plotly.express as px
            
        except ImportError:
            print('Failed to import dash & plotly, so the interactive tool'
                  'won\'t work.\n'
                  'Install with `pip install "dash>=2.5.1"` and also '
                  '`pip install jupyter_dash` for running a server '
                  'through jupyter')
                  
        uv = -2.5*np.log10(zout['restU']/zout['restV'])
        vj = -2.5*np.log10(zout['restV']/zout['restJ'])
        ssfr = zout['sfr']/zout['mass']
        
        for c in ['z_phot', 'z_spec']:
            if hasattr(zout[c], 'mask'):
                zout[c].fill_value = -0.1
                
        def fill_masked(_data, fill_value=None):
            if hasattr(_data, 'mask'):
                if fill_value is None:
                    return _data.filled()
                else:
                    return _data.filled(fill_value)
            else:
                return _data
        
        self.extra_slider_kws = extra_slider_kws
        if 'flux_radius' in zout.colnames:
            self.extra_slider_kws = {'column':'flux_radius', 'label':'R50',
                                     'min':0, 'max':20, 'step':0.2,
                                     'value':[1,10], 'checked':[],
                                     }
                                
        #df = pd.DataFrame()
        df = Table()
        df['id'] = fill_masked(zout['id'])
        df['nusefilt'] = fill_masked(zout['nusefilt'])
        df['uv'] = fill_masked(uv)
        df['vj'] = fill_masked(vj)
        df['ssfr'] = fill_masked(np.log10(ssfr))
        df['mass'] = fill_masked(np.log10(zout['mass']))
        df['z_phot'] = fill_masked(zout['z_phot'])
        df['z_spec'] = np.clip(fill_masked(zout['z_spec']), -0.1, 12)
        df['ra'] = fill_masked(photoz.RA)
        df['dec'] = fill_masked(photoz.DEC)
        df['chi2'] = fill_masked(zout['z_phot_chi2']/zout['nusefilt'])
        
        self.zp = photoz.zp*1
        
        for c in extra_zout_columns:
            if c in zout.colnames:
                df[c] = fill_masked(zout[c])
        
        col = self.extra_slider_kws['column']
        if col in zout.colnames:
            if col not in df.columns:
                df[col] = fill_masked(zout[col])
        else:
            if col not in df.columns:
                df[col] = df['id']
        
        _red_ix = np.argmax(photoz.pivot*(photoz.pivot < 3.e4))
        self.DEFAULT_FILTER = photoz.flux_columns[_red_ix]
                
        ZP = photoz.param['PRIOR_ABZP']*1.
        fmin = 10**(-0.4*(33-ZP))
        fmax = 10**(-0.4*(12-ZP))
        #print('flux limits', fmin, fmax)
        
        for i, f in enumerate(photoz.flux_columns):
            key = f'mag_{f}'
            df[key] = ZP - 2.5*np.log10(np.clip(fill_masked(photoz.cat[f]*self.zp[i]), 
                                                fmin, fmax))
            # if hasattr(photoz.cat[f], 'mask'):
            #     
            #     df[key] = ZP - 2.5*np.log10(np.clip(photoz.cat[f].filled(-99), 
            #                                         fmin, fmax))
            # else:
            #     df[key] = ZP - 2.5*np.log10(np.clip(photoz.cat[f], 
            #                                         fmin, fmax))
        
        df['mag'] = df[f'mag_{self.DEFAULT_FILTER}']
        df['mag1mag2'] = df['mag']*0
        
        self.extra_plots = {}
        
        for k in extra_plots:
            _plot_args = extra_plots[k]
            xcol, ycol = _plot_args[:2]
            
            if xcol not in df.columns:
                df[xcol] = fill_masked(zout[xcol])
                    
            if ycol not in df.columns:
                df[ycol] = fill_masked(zout[ycol])

            if len(_plot_args) == 2:
                self.extra_plots[k] = (xcol, ycol, xcol, ycol, 
                                       np.nanmin(xcol), np.nanmax(ycol))
            elif len(_plot_args) == 4:
                self.extra_plots[k] = (*_plot_args, 
                                       np.nanmin(xcol), np.nanmax(ycol))
            elif len(_plot_args) == 6:
                self.extra_plots[k] = _plot_args
            else:
                print(f'Expected 2,4 or 6 elements in extra_plots[{k}], '
                      f'found {len(_plot_args)}')
        
        if selection is not None:
            self.df = df[selection].to_pandas()
        else:
            self.df = df.to_pandas()

        self.zout = zout
        self.photoz = photoz
        
        self.ZMAX = photoz.zgrid.max()
        self.MAXNFILT = photoz.nusefilt.max()
        

    @property
    def ra_bounds(self):
        return (self.df['ra'].max(), self.df['ra'].min())


    @property 
    def dec_bounds(self):
        return (self.df['dec'].min(), self.df['dec'].max())

    
    def get_filters_from_api(self):
        """
        Query available HST/JWST filters from the heroku API
        """
        import urllib
        import urllib.request
        import json 
        import PIL.Image
        
        _ra = self.ra_bounds
        _dec = self.dec_bounds
        
        cosd = np.cos(np.mean(_dec)/180*np.pi)
        dx = np.diff(_ra)[0]*cosd
        dy = np.diff(_dec)[0]
        si = np.maximum(dx, dy)*3600
        
        furl = 'https://grizli-cutout.herokuapp.com/overlap?'
        furl += f'ra={np.mean(_ra)}&dec={np.mean(_dec)}&size={si}'
        with open('/tmp/thumb.log','a') as fp:
            fp.write(furl+'\n')
        
        with urllib.request.urlopen(furl) as url:
            olap = json.loads(url.read().decode())
        
        return olap
    
    
    def make_dash_app(self, template='plotly_white', server_mode='external', port=8050, app=None, app_type='jupyter', plot_height=680, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'], infer_proxy=False, slider_width=140, cutout_hdu=None, cutout_rgb=None, cutout_size=10, api_filters=None, api_size=2, api_args='', PLOT_TYPES=['zphot-zspec', 'Mag-redshift', 'Mag-color', 'redshift-color', 'Mass-redshift', 'UVJ', 'RA/Dec', 'UV-redshift', 'chi2-redshift'], get_content=False, fitsmap_url=None, cutout_url=CUTOUT_URL):
        """
        Create a Plotly/Dash app for interactive exploration
        
        Parameters
        ----------
        template : str
            `plotly` style `template <https://plotly.com/python/templates/#specifying-themes-in-graph-object-figures>`_.
        
        server_mode, port : str, int
            If not `None`, the app server is started with 
            `app.run_server(mode=server_mode, port=port)`.
        
        app_type : str
            If ``jupyter`` then `app = jupyter_dash.JupyterDash()`, else
            `app = dash.Dash()`
            
        plot_height : int
            Height in pixels of the scatter and SED+P(z) plot windows.
        
        infer_proxy : bool
            Run `JupyterDash.infer_jupyter_proxy_config()`, before app 
            initilization, e.g., for running on GoogleColab.
            
        Returns
        -------
        app : object
            App object following `app_type`.
            
        """
        import dash
        from dash import  dcc, html, callback
        from dash.dependencies import Input, Output
        import plotly.express as px
        
        import matplotlib.pyplot as plt
        from urllib.parse import urlparse, parse_qsl, urlencode
        import astropy.wcs as pywcs
        
        self.template = template
        self.plot_height = plot_height
        self.slider_width = slider_width
        self.cutout_size = cutout_size
        
        pivots = {}
        for f, w in zip(self.photoz.flux_columns, self.photoz.pivot):
            pivots[f] = w
        
        if app is None:
            if app_type == 'dash':
                app = dash.Dash(__name__, 
                                external_stylesheets=external_stylesheets)
            else:
                from jupyter_dash import JupyterDash
                if infer_proxy:
                    JupyterDash.infer_jupyter_proxy_config()
                
                app = JupyterDash(__name__, 
                                  external_stylesheets=external_stylesheets)
        
        for _t in self.extra_plots:
            PLOT_TYPES.append(_t)
            
        COLOR_TYPES = ['z_phot', 'z_spec', 'mass', 'sSFR', 'chi2']
        self.COLOR_TYPES = COLOR_TYPES
        self.PLOT_TYPES = PLOT_TYPES
        self.cutout_data = None
        self.cutout_wcs = None
        self.api_filters = api_filters
        self.fitsmap_url = fitsmap_url
        self.cutout_url = cutout_url
        
        #_title = f"{self.photoz.param['MAIN_OUTPUT_FILE']}"
        #_subhead = f"Nobj={self.photoz.NOBJ}  Nfilt={self.photoz.NFILT}"
        _title = [html.Strong(self.photoz.param['MAIN_OUTPUT_FILE']), 
                  ' / N', html.Sub('obj'), f'={self.photoz.NOBJ}', 
                  ' / N', html.Sub('filt'), f'={self.photoz.NFILT}', 
                  ]
        
        slider_row_style={'width': '90%', 'float':'left', 
                          'margin-left':'10px'}          
        slider_container = {'width': f'{slider_width}px', 
                            'margin-left':'-25px'}
        check_kwargs = dict(style={'text-align':'center', 
                                   'height':'14pt', 
                                   'margin-top':'-20px'})
        
        # bool_options = {'has_zspec': 'z_spec > 0', 
        #                 'use': 'Use == 1'}
        
        if cutout_hdu is not None:
            self.cutout_wcs = pywcs.WCS(cutout_hdu.header, relax=True)
            if cutout_rgb is None:
                self.cutout_data = cutout_hdu.data
            else:
                self.cutout_data = np.flipud(plt.imread(cutout_rgb))
                
            print('xxx', self.cutout_data.shape)
            
            cutout_div = html.Div([
                             dcc.Graph(id='cutout-figure', 
                                       style={})
                                 ], style={'right':'70px', 
                                           'width':'120px',
                                           'height':'120px',
                                        'border':'1px solid rgb(200,200,200)',
                                           'top':'10px', 
                                           'position':'absolute'})
            cutout_target = 'figure'
            
        elif api_filters is not None:
            
            cutout_div = html.Div([
                             dcc.Graph(id='cutout-figure', 
                                       style={})
                                 ], style={'right':'70px', 
                                           'width':'120px',
                                           'height':'120px',
                                        'border':'1px solid rgb(200,200,200)',
                                           'top':'10px', 
                                           'position':'absolute'})
            cutout_target = 'figure'
            self.cutout_data = None
            
        else:
            cutout_div = html.Div(id='cutout-figure', 
                                   style={'left':'1px', 
                                           'width':'1px',
                                           'height':'1px',
                                           'bottom':'1px', 
                                           'position':'absolute'})
            self.cutout_data = None
            cutout_target = 'children'
                
        
        ####### Main content
        content = [
            # Selectors
            html.Div([
                dcc.Location(id='url', refresh=False), 
                
                html.Div([
                    html.Div(_title, id='title-bar', 
                             style={'float':'left', 'margin-top':'4pt'}),
                    
                    html.Div([
                        html.Div([dcc.Dropdown(id='plot-type',
                                     options=[{'label': i, 'value': i}
                                              for i in PLOT_TYPES],
                                     value=PLOT_TYPES[0], 
                                     clearable=False,
                                     style={'width':'120px', 
                                            'margin-right':'5px',
                                            'margin-left':'5px',
                                            'font-size':'8pt'}),
                        ], style={'float':'left'}),
                        
                        html.Div([dcc.Dropdown(id='color-type',
                                     options=[{'label': i, 'value': i}
                                              for i in COLOR_TYPES],
                                     value='sSFR', 
                                     clearable=False,
                                     style={'width':'80px', 
                                            'margin-right':'5px',
                                            'font-size':'8pt'}),
                        ], style={'display':'inline-block', 
                                  'margin-left':'10px'}),
                    ], style={'float':'right'}),
                ], style=slider_row_style),
                
                html.Div([
                    html.Div([dcc.Dropdown(id='mag-filter',
                                 options=[{'label': i, 'value': i}
                                          for i in self.photoz.flux_columns],
                                 value=self.DEFAULT_FILTER, 
                                 style={'width': f'{slider_width-45}px', 
                                        'margin-right':'20px',
                                        'font-size':'8pt'},
                                 clearable=False),
                    ], style={'float':'left'}),
                                        
                    html.Div([
                        dcc.RangeSlider(id='mag-slider',
                                        min=12, max=32, step=0.2,
                                        value=[18, 27],
                                        updatemode='drag',
                                        tooltip={"placement":'left'}, 
                                        marks=None), 

                        dcc.Checklist(id='mag-checked', 
                                      options=[{'label':'AB mag', 
                                                'value':'checked'}], 
                                      value=['checked'], **check_kwargs),
                              
                    ], style=dict(display='inline-block',
                                  **slider_container)),
                    
                    # Mag2 for color
                    html.Div([dcc.Dropdown(id='mag2-filter',
                                 options=[{'label': i, 'value': i}
                                          for i in self.photoz.flux_columns],
                                 value=self.photoz.flux_columns[0], 
                                 style={'width': f'{slider_width-45}px', 
                                        'margin-right':'20px',
                                        'font-size':'8pt'},
                                 clearable=False),
                    ], style={'float':'left'}),
                    
                    html.Div([
                        dcc.RangeSlider(id='color-slider',
                                        min=-2, max=5, step=0.1,
                                        value=[-1, 3],
                                        updatemode='drag',
                                        tooltip={"placement":'left'}, 
                                        marks=None), 

                        dcc.Checklist(id='color-checked', 
                                      options=[{'label':'Mag1 - Mag2', 
                                                'value':'checked'}], 
                                      value=[], **check_kwargs),
                              
                    ], style=dict(display='inline-block',
                                  **slider_container)),
                    
                    html.Div([
                        dcc.RangeSlider(id='nfilt-slider',
                                        min=1, max=self.MAXNFILT, step=1,
                                        value=[3, self.MAXNFILT],
                                        updatemode='drag',
                                        tooltip={"placement":'left'},
                                        marks=None),

                        dcc.Checklist(id='nfilt-checked', 
                                      options=[{'label':'nfilt', 
                                                'value':'checked'}], 
                                      value=['checked'], **check_kwargs),

                    ], style=dict(display='inline-block', 
                                  **slider_container)),
                    
                ], style=slider_row_style),
                
                html.Div([
                    html.Div([
                        dcc.RangeSlider(id='zphot-slider',
                                        min=-0.5, max=self.ZMAX, step=0.1,
                                        value=[0, self.ZMAX],
                                        updatemode='drag',
                                        tooltip={"placement":'left'},
                                        marks=None),
                          
                         dcc.Checklist(id='zphot-checked', 
                                       options=[{'label':'z_phot', 
                                                 'value':'checked'}], 
                                       value=['checked'], **check_kwargs),
                
                    ], style=dict(float='left', **slider_container)), 

                    html.Div([
                        dcc.RangeSlider(id='zspec-slider',
                                        min=-0.5, max=self.ZMAX, step=0.1,
                                        value=[-0.5, 6.5],
                                        updatemode='drag',
                                        tooltip={"placement":'left'},
                                        marks=None),
                          
                        dcc.Checklist(id='zspec-checked', 
                                      options=[{'label':'z_spec', 
                                                'value':'checked'}], 
                                      value=['checked'], **check_kwargs),

                    ], style=dict(display='inline-block',
                                  **slider_container)),

                    html.Div([
                        dcc.RangeSlider(id='mass-slider',
                                        min=7, max=13, step=0.1,
                                        value=[8, 11.8],
                                        updatemode='drag',
                                        tooltip={"placement":'left'},
                                        marks=None),

                        dcc.Checklist(id='mass-checked', 
                                      options=[{'label':'mass', 
                                                'value':'checked'}], 
                                      value=['checked'], **check_kwargs),

                    ], style=dict(display='inline-block', 
                                  **slider_container)),
                    
                    html.Div([
                        dcc.RangeSlider(id='chi2-slider',
                                        min=0, max=20, step=0.1,
                                        value=[0, 6],
                                        updatemode='drag',
                                        tooltip={"placement":'left'},
                                        marks=None),

                        dcc.Checklist(id='chi2-checked', 
                                      options=[{'label':'chi2',
                                                'value':'checked'}], 
                                      value=[], **check_kwargs),

                    ], style=dict(display='inline-block', 
                                  **slider_container)),
                    
                    # Customizable slider
                    html.Div([
                        dcc.RangeSlider(id='extra-slider',
                                        min=self.extra_slider_kws['min'], 
                                        max=self.extra_slider_kws['max'],
                                        step=self.extra_slider_kws['step'],
                                        value=self.extra_slider_kws['value'],
                                        updatemode='drag',
                                        tooltip={"placement":'left'},
                                        marks=None),

                        dcc.Checklist(id='extra-checked', 
                                      options=[{'label':self.extra_slider_kws['label'], 
                                                'value':'checked'}], 
                                      value=self.extra_slider_kws['checked'], 
                                      **check_kwargs),

                    ], style=dict(display='inline-block', 
                                  **slider_container)),
                    
                    # Boolean dropdown
                    # dcc.Dropdown(id='bool-checks',
                    #              options=[{'label': self.bool_options[k], 
                    #                        'value': k}
                    #                       for k in self.bool_options],
                    #              value=[], 
                    #              multi=True,
                    #              style={'width':'100px', 
                    #                     'display':'inline-block',
                    #                     'margin-left':'0px', 
                    #                     'font-size':'8pt'},
                    #              clearable=True),
                    
                ], style=slider_row_style), 

            ], style={'float':'left','width': '55%'}),

            # Object-level controls
            html.Div([
                html.Div([
                    html.Div('ID / RA,Dec.', style={'float':'left', 
                                                       'width':'100px', 
                                                       'margin-top':'5pt'}), 
                    
                    dcc.Input(id='id-input', type='text',
                          style={'width':'120px',
                                 'padding':'2px', 
                                 'display':'inline',
                                 'font-size':'8pt'}), 
                    html.Div(children='', id='match-sep', 
                           style={'margin':'5pt',
                           'display':'inline', 
                                  'width':'50px',
                                  'font-size':'8pt'}),
                    
                    dcc.RadioItems(id='sed-unit-selector',
                                             options=[{'label': i, 'value': i}
                                             for i in ['Fλ', 'Fν', 'νFν']],
                                             value='Fλ',
                                             labelStyle={'display':'inline', 
                                                         'padding':'3px', 
                                                         },
                                             style={'display':'inline',
                                                    'width':'130px'})
                                                              
                ],  style={'width':'260pix', 'float':'left', 
                           'margin-right':'20px'}),
                
            ]), 
            html.Div([           
                # html.Div([
                # ],  style={'width':'120px', 'float':'left'}), 

                html.Div(id='object-info', children='ID: ', 
                        style={'margin':'auto','margin-top':'10px', 
                               'font-size':'10pt'})

            ], style={'float':'right', 'width': '45%'}),

            # Plots
            html.Div([# Scatter plot
                dcc.Graph(id='sample-selection-scatter', 
                          hoverData={'points': [{'customdata':
                                             (self.df['id'][0], 
                                              1.0, -9.0)}]}, 
                          style={'width':'95%'})
            ], style={'float':'left', 'height':'70%', 'width':'49%'}), 

            html.Div([# SED
                dcc.Graph(id='object-sed-figure',
                          style={'width':'95%'})
            ], style={'float':'right', 'width':'49%', 'height':'70%'}),
            
            cutout_div
        ]

        if get_content:
            return content
        
        ##### Callback functions
        @app.callback(
             Output('url', 'search'),
            [Input('plot-type', 'value'),
             Input('color-type', 'value'),
             Input('mag-filter', 'value'),
             Input('mag-slider', 'value'),
             Input('mag2-filter', 'value'),
             Input('color-slider', 'value'),
             Input('mass-slider', 'value'),
             Input('chi2-slider', 'value'),
             Input('nfilt-slider', 'value'),
             Input('zphot-slider', 'value'),
             Input('zspec-slider', 'value'),
             Input('id-input', 'value')])
        def update_url_state(plot_type, color_type, mag_filter, mag_range, mag2_filter, color_range, mass_range, chi2_range, nfilt_range, zphot_range, zspec_range, id_input):
            search = f'?plot_type={plot_type}&color_type={color_type}'
            search += f'&mag_filter={mag_filter}'
            search += f'&mag={mag_range[0]},{mag_range[1]}'
            search += f'&mag2_filter={mag2_filter}'
            search += f'&color={color_range[0]},{color_range[1]}'
            search += f'&mass={mass_range[0]},{mass_range[1]}'
            search += f'&chi2={chi2_range[0]},{chi2_range[1]}'
            search += f'&nfilt={nfilt_range[0]},{nfilt_range[1]}'
            search += f'&zphot={zphot_range[0]},{zphot_range[1]}'
            search += f'&zspec={zspec_range[0]},{zspec_range[1]}'
            if id_input is not None:
                search += f"&id={id_input.replace(' ', '%20')}"
                
            return search


        @app.callback([Output('plot-type', 'value'),
                       Output('color-type', 'value'),
                       Output('mag-filter', 'value'),
                       Output('mag-slider', 'value'),
                       Output('mag2-filter', 'value'),
                       Output('color-slider', 'value'),
                       Output('mass-slider', 'value'),
                       Output('chi2-slider', 'value'),
                       Output('nfilt-slider', 'value'),
                       Output('zphot-slider', 
                                                'value'),
                       Output('zspec-slider', 
                                                'value'),
                       Output('id-input', 'value'),
                      ],[
                       Input('url', 'search')
                      ])
        def set_state_from_url(search):
            plot_type = PLOT_TYPES[0]
            color_type = 'sSFR'
            mag_filter = self.DEFAULT_FILTER
            mag_range = [18, 27]
            mag2_filter = self.DEFAULT_FILTER
            color_range = [-0.5, 3]
            mass_range = [8, 11.6]
            chi2_range = [0, 4]
            nfilt_range = [1, self.MAXNFILT]
            zphot_range = [0, self.ZMAX]
            zspec_range = [-0.5, 6.5]
            id_input = None

            # if '?' not in href:
            if not search:
                return (plot_type, color_type, mag_filter, mag_range,
                        mag2_filter, color_range,
                        mass_range, chi2_range, nfilt_range,
                        zphot_range, zspec_range,
                        id_input)

            # search = href.split('?')[1]
            params = search.split('&')

            for p in params:
                if 'plot_type' in p:
                    val = p.split('=')[1]
                    if val in PLOT_TYPES:
                        plot_type = val

                elif 'color_type' in p:
                    val = p.split('=')[1]
                    if val in COLOR_TYPES:
                        color_type = val
                        
                elif 'mag_filter' in p:
                    val = p.split('=')[1]
                    if val in self.photoz.flux_columns:
                        mag_filter = val
                    
                elif 'mag=' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            mag_range = vals
                    except ValueError:
                        pass
                        
                elif 'mag2_filter' in p:
                    val = p.split('=')[1]
                    if val in self.photoz.flux_columns:
                        mag2_filter = val

                elif 'color=' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            color_range = vals
                    except ValueError:
                        pass

                elif 'mass' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            mass_range = vals
                    except ValueError:
                        pass
                
                elif 'nfilt=' in p:
                    try:
                        vals = [int(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            nfilt_range = vals
                    except ValueError:
                        pass
                
                elif 'zspec' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            zspec_range = vals
                    except ValueError:
                        pass
                        
                elif 'zphot' in p:
                    try:
                        vals = [float(v) for v in p.split('=')[1].split(',')]
                        if len(vals) == 2:
                            zphot_range = vals
                    except ValueError:
                        pass
                        
                elif 'id' in p:
                    try:
                        id_input = p.split('=')[1].replace('%20', ' ')
                    except ValueError:
                        id_input = None
                    
                    if not id_input:
                        id_input = None
                        
            return (plot_type, color_type, mag_filter, mag_range,
                    mag2_filter, color_range,
                    mass_range, chi2_range, nfilt_range,
                    zphot_range, zspec_range,
                    id_input)

        @app.callback(
            Output('nfilt-checked', 'label'),
            Input('sample-selection-scatter', 'selectedData'))
        def update_selected_data(selectedData):
            
            if selectedData is not None:
                #print('selectedData', len(selectedData['points']))
                
                #print('y0', self.df['in_selectedData'].sum())
                
                if len(selectedData['points']) == 1:
                    # Reset
                    self.df['in_selectedData'] = self.df['z_phot'] > 0
                    print('One point selected: reset selection!')
                    
                elif len(selectedData['points']) > 0:
                    selected_ids = []
                    for p in selectedData['points']:
                        if "customdata" in p.keys():
                            selected_ids.append(p['customdata'][0])
                    
                    #print(len(selected_ids), p)
                    
                    if len(selected_ids) == 1:
                        # Reset
                        self.df['in_selectedData'] = self.df['z_phot'] > 0
                        print('One point selected: reset selection!')
                        
                    elif len(selected_ids) > 0:
                        self.df['in_selectedData'] = np.isin(self.df['id'], 
                                                             selected_ids)
                
                #print('y1', self.df['in_selectedData'].sum())
                
            else:
                # print('selectedData None')
                self.df['in_selectedData'] = self.df['z_phot'] > 0
                
            return 'nfilt' #f"N: {self.df['in_selectedData'].sum()}"
        
        
        @app.callback(
            Output('sample-selection-scatter', 'figure'),
            [Input('plot-type', 'value'),
             Input('color-type', 'value'), 
             Input('mag-filter', 'value'),
             Input('mag-slider', 'value'),
             Input('mag-checked', 'value'),
             Input('mag2-filter', 'value'),
             Input('color-slider', 'value'),
             Input('color-checked', 'value'),
             Input('mass-slider', 'value'),
             Input('mass-checked', 'value'),
             Input('extra-slider', 'value'),
             Input('extra-checked', 'value'),
             Input('chi2-slider', 'value'),
             Input('chi2-checked', 'value'),
             Input('nfilt-slider', 'value'),
             Input('nfilt-checked', 'value'),
             Input('zphot-slider', 'value'),
             Input('zphot-checked', 'value'),
             Input('zspec-slider', 'value'),
             Input('zspec-checked', 'value'),
             Input('id-input', 'value')
         ])
        def update_selection(plot_type, color_type, mag_filter, mag_range, mag_checked, mag2_filter, color_range, color_checked, mass_range, mass_checked, extra_range, extra_checked, chi2_range, chi2_checked, nfilt_range, nfilt_checked, zphot_range, zphot_checked, zspec_range, zspec_checked, id_input):
            """
            Apply slider selections
            """
            sel = np.isfinite(self.df['z_phot'])
            if 'checked' in zphot_checked:
                sel &= (self.df['z_phot'] > zphot_range[0]) 
                sel &= (self.df['z_phot'] < zphot_range[1])
            
            if 'checked' in zspec_checked:
                sel &= (self.df['z_spec'] > zspec_range[0]) 
                sel &= (self.df['z_spec'] < zspec_range[1])
            
            if 'checked' in mass_checked:
                sel &= (self.df['mass'] > mass_range[0])
                sel &= (self.df['mass'] < mass_range[1])

            if 'checked' in chi2_checked:
                sel &= (self.df['chi2'] >= chi2_range[0])
                sel &= (self.df['chi2'] <= chi2_range[1])

            if 'checked' in nfilt_checked:
                sel &= (self.df['nusefilt'] >= nfilt_range[0])
                sel &= (self.df['nusefilt'] <= nfilt_range[1])

            if 'checked' in extra_checked:
                _col = self.extra_slider_kws['column']
                sel &= (self.df[_col] >= extra_range[0])
                sel &= (self.df[_col] <= extra_range[1])
            
            #print('redshift: ', sel.sum())
            
            if mag_filter is None:
                mag_filter = self.DEFAULT_FILTER
            
            if mag2_filter is None:
                mag2_filter = self.DEFAULT_FILTER
            
            #self.self.df['mag'] = self.ABZP 
            #self.self.df['mag'] -= 2.5*np.log10(self.photoz.cat[mag_filter])
            mag_col = 'mag_'+mag_filter            
            if 'checked' in mag_checked:
                sel &= (self.df[mag_col] > mag_range[0]) 
                sel &= (self.df[mag_col] < mag_range[1])
            
            mag2_col = 'mag_'+mag2_filter
            
            fblue, fred = get_sorted_mag_columns(mag_filter, mag2_filter)
            
            mag1mag2 = self.df['mag_'+fblue] - self.df['mag_'+fred]
            
            if 'checked' in color_checked:
                sel &= (mag1mag2 > color_range[0]) 
                sel &= (mag1mag2 < color_range[1])
            
            self.df['mag'] = self.df[mag_col]
            self.df['mag1mag2'] = mag1mag2
            
            #print('mag: ', sel.sum())
            
            if plot_type == 'zphot-zspec':
                sel &= self.df['z_spec'] > 0
            
            #print('zspec: ', sel.sum())
            
            if id_input is not None:
                id_i, dr_i = parse_id_input(id_input)
                if id_i is not None:
                    self.df['is_selected'] = self.df['id'] == id_i
                    sel |= self.df['is_selected']
                else:
                    self.df['is_selected'] = False
            else:
                self.df['is_selected'] = False
            
            xsel = self.df['in_selectedData'][sel]
            dff = self.df[sel]
            
            # Color-coding by color-type pulldown
            if color_type == 'z_phot':
                color_kwargs = dict(color=np.clip(dff['z_phot'][xsel], 
                                                  *zphot_range),
                                    color_continuous_scale='portland')
            elif color_type == 'z_spec':
                color_kwargs = dict(color=np.clip(dff['z_spec'][xsel], 
                                                  *zspec_range), 
                                    color_continuous_scale='portland')
            elif color_type == 'mass':
                color_kwargs = dict(color=np.clip(dff['mass'][xsel], *mass_range), 
                                    color_continuous_scale='magma_r')
            elif color_type == 'chi2':
                color_kwargs = dict(color=np.clip(dff['chi2'][xsel], *chi2_range), 
                                    color_continuous_scale='viridis')            
            else:
                color_kwargs = dict(color=np.clip(dff['ssfr'][xsel], -12., -8.), 
                                    color_continuous_scale='portland_r')
            
            # Scatter plot  
            plot_defs = {'Mass-redshift':['z_phot','mass',
                                     'z<sub>phot</sub>', 'log Stellar mass', 
                                     (-0.1, self.ZMAX), (7.5, 12.5)], 
                         'Mag-redshift': ['z_phot','mag',
                                'z<sub>phot</sub>', f'AB mag ({mag_filter})', 
                                (-0.1, self.ZMAX), (18, 28)],
                         'Mag-color': ['mag','mag1mag2',
                                f'AB mag ({mag_filter})',
                                f'{fblue} - {fred}'.replace('_tot_1',''),
                                (18, 28),
                                (-0.5, 3)],
                         'redshift-color': ['z_phot','mag1mag2',
                                f'z_phot',
                                f'{fblue} - {fred}'.replace('_tot_1',''),
                                (-0.1, self.ZMAX),
                                (-0.5, 3)],
                         'RA/Dec': ['ra','dec',
                                    'R.A.', 'Dec.', 
                                    self.ra_bounds, self.dec_bounds], 
                         'zphot-zspec': ['z_spec','z_phot',
                                   'z<sub>spec</sub>', 'z<sub>phot</sub>', 
                                    (0, 4.5), (0, 4.5)], 
                         'UVJ': ['vj','uv',
                                  '(V-J)', '(U-V)', 
                                  (-0.1, 2.5), (-0.1, 2.5)], 
                         'UV-redshift': ['z_phot','uv',
                                 'z<sub>phot</sub>', '(U-V)<sub>rest</sub>', 
                                 (0, 4), (-0.1, 2.50)], 
                         'chi2-redshift': ['z_phot','chi2',
                                 'z<sub>phot</sub>', 'chi<sup>2</sup>',
                                 (0, 4), (0.1, 30)]
                         }
            
            if plot_type in self.extra_plots:
                args = [*self.extra_plots[plot_type], {}, color_kwargs]
            elif plot_type in plot_defs:
                args = [*plot_defs[plot_type], {}, color_kwargs]
            else:
                args = [*plot_defs['zphot-zspec'], {}, color_kwargs]
            
            if args[0] == 'mag':
                args[2] = f'AB mag ({mag_filter})'
            
            if args[1] == 'mag':
                args[3] = f'AB mag ({mag_filter})'
                
            fig = update_sample_scatter(dff, *args)
            
            # Update ranges for some parameters
            if ('Mass' in plot_type) & ('checked' in mass_checked):
                fig.update_yaxes(range=mass_range)

            if ('Mag' in plot_type) & ('checked' in mag_checked):
                if args[0] == 'mag':
                    fig.update_xaxes(range=mag_range)
                else:
                    fig.update_yaxes(range=mag_range)
                    
            if ('color' in plot_type) & ('checked' in color_checked):
                if args[0] == 'mag1mag2':
                    fig.update_xaxes(range=color_range)
                else:
                    fig.update_yaxes(range=color_range)
                    
            if ('redshift' in plot_type) & ('checked' in zphot_checked):
                if args[0] == 'z_phot':
                    fig.update_xaxes(range=zphot_range)
                else:
                    fig.update_yaxes(range=zphot_range)
                    
            if ('zspec' in plot_type) & ('checked' in zspec_checked):
                if args[0] == 'z_spec':
                    fig.update_xaxes(range=zspec_range)
                else:
                    fig.update_yaxes(range=zspec_range)
            
            return fig


        def update_sample_scatter(dff, xcol, ycol, x_label, y_label, x_range, y_range, extra,  color_kwargs):
            """
            Make scatter plot
            """
            import plotly.graph_objects as go
            
            # print('update_sample_scatter xxx', xcol, len(dff[xcol]))
            is_sel = dff['in_selectedData']
            fig = px.scatter(data_frame=dff[is_sel], x=xcol, y=ycol, 
                             custom_data=['id','z_phot','mass','ssfr','mag'], 
                             **color_kwargs)
                            
            htempl = '(%{x:.2f}, %{y:.2f}) <br>'
            htempl += 'id: %{customdata[0]:0d}  z_phot: %{customdata[1]:.2f}'
            htempl += '<br> mag: %{customdata[4]:.1f}  '
            htempl += 'mass: %{customdata[2]:.2f}  ssfr: %{customdata[3]:.2f}'
                
            fig.update_traces(hovertemplate=htempl, opacity=0.7)

            if (~is_sel).sum() > 0:
                _xsel = go.Scatter(x=dff[~is_sel][xcol],
                                   y=dff[~is_sel][ycol], 
                                   mode="markers", 
                                   marker=dict(color='rgba(180,180,180,0.2)',
                                               size=5, symbol='circle'),
                                   hoverinfo='skip',
                               )
                fig.add_trace(_xsel)
                fig.data = fig.data[::-1]

            if dff['is_selected'].sum() > 0:
                dffs = dff[dff['is_selected']]
                _sel = go.Scatter(x=dffs[xcol], y=dffs[ycol],
                                  mode="markers+text",
                                  text=[f'{id}' for id in dffs['id']],
                                  textposition="bottom center",
                                  marker=dict(color='rgba(250,0,0,0.5)', 
                                              size=20, 
                                              symbol='circle-open'))
                                  
                fig.add_trace(_sel)
            
            
            fig.update_xaxes(range=x_range, title_text=x_label)
            fig.update_yaxes(range=y_range, title_text=y_label)

            fig.update_layout(template=template, 
                              autosize=True, showlegend=False, 
                              margin=dict(l=0,r=0,b=0,t=20,pad=0,
                                          autoexpand=True))
            
            if plot_height is not None:
                fig.update_layout(height=plot_height)
                
            fig.update_traces(marker_showscale=False, 
                              selector=dict(type='scatter'))
            fig.update_coloraxes(showscale=False)
            
            if (xcol, ycol) == ('z_spec','z_phot'):
                _one2one = go.Scatter(x=[0, 8], y=[0,8],
                                  mode="lines",
                                  marker=dict(color='rgba(250,0,0,0.5)'))
                fig.add_trace(_one2one)
                
            fig.add_annotation(text=f'N = {len(dff)} / {len(self.df)}',
                          xref="x domain", yref="y domain",
                          x=0.98, y=0.05, showarrow=False)
            
            return fig


        def heroku_thumbnail(id_i):
            """
            Thumbnail from grizli API
            """
            import urllib
            import urllib.request
            import json 
            import PIL.Image
            
            ix = np.where(self.df['id'] == id_i)[0][0]
            ri, di = self.df['ra'][ix], self.df['dec'][ix]
            
            turl = f'https://grizli-cutout.herokuapp.com/thumb?'
            turl += f'ra={ri}&dec={di}&size={api_size}'
            turl += f'{api_args}&filters={api_filters}'   
            print(turl)
            # with open('/tmp/thumb.log','a') as fp:
            #     fp.write(turl+'\n')
                
            req = urllib.request.urlopen(turl)
                                                        
            thumb = np.array(PIL.Image.open(req))
            
            return thumb

        
        def api_cutout_figure(id_i):
            """
            Thumbnail from grizli API
            """
            try:
                cutout = heroku_thumbnail(id_i)
            except:
                cutout = np.zeros((10, 10))
            
            sh = cutout.shape
            
            fig = px.imshow(cutout, origin='upper')

            fig.update_coloraxes(showscale=False)
            fig.update_layout(width=120, height=120, 
                              margin=dict(l=0,r=0,b=0,t=0,pad=0,
                                              autoexpand=True))

            fig.update_xaxes(range=(-0.5, sh[1]-0.5), 
                             visible=False, showticklabels=False)
            fig.update_yaxes(range=(-0.5, sh[0]-0.5),
                             visible=False, showticklabels=False)
            
            return fig


        def hdu_cutout_figure(id_i):
            """
            SED cutout
            """

            ix = np.where(self.df['id'] == id_i)[0]
            ri, di = self.df['ra'][ix], self.df['dec'][ix]
            xi, yi = np.squeeze(self.cutout_wcs.all_world2pix([ri], [di], 0))
            xp = int(np.round(xi))
            yp = int(np.round(yi))
            slx = slice(xp-cutout_size,xp+cutout_size+1)
            sly = slice(yp-cutout_size,yp+cutout_size+1)

            try:
                if self.cutout_data.ndim == 2:
                    cutout = self.cutout_data[sly, slx]
                    fig = px.imshow(cutout, color_continuous_scale='gray_r', 
                                origin='lower')
                else:
                    cutout = self.cutout_data[sly, slx, :]
                    fig = px.imshow(cutout, origin='lower')
            except:
                cutout = np.zeros((2*cutout_size, 2*cutout_size))
                
                fig = px.imshow(cutout, color_continuous_scale='gray_r', 
                            origin='lower')

            fig.update_coloraxes(showscale=False)
            fig.update_layout(width=120, height=120, 
                              margin=dict(l=0,r=0,b=0,t=0,pad=0,
                                              autoexpand=True))

            fig.update_xaxes(range=(0, 2*cutout_size), 
                             visible=False, showticklabels=False)
            fig.update_yaxes(range=(0, 2*cutout_size),
                             visible=False, showticklabels=False)

            return fig

        
        def get_sorted_mag_columns(mag_filter, mag2_filter):
            """
            """
            if pivots[mag_filter] > pivots[mag2_filter]:
                return mag2_filter, mag_filter
            else:
                return mag_filter, mag2_filter
        
        def get_map_link(ra, dec):
            """
            Return an html.A object with a link to a FITSMap or LegacySurvey map
            """
            from dash import html
            
            if self.fitsmap_url is None:
                link = html.A('LegacySurvey', 
                              href=utils.show_legacysurvey(ra, dec, layer='ls-dr9'))
            else:
                href = self.fitsmap_url.format(ra=ra, dec=dec)
                link = html.A('FitsMap', href=href)
            
            return link
        
        
        def parse_id_input(id_input):
            """
            Parse input as id or (ra dec)
            """
            if id_input in ['None', None, '']:
                return None, None
            
            inp_split = id_input.replace(',',' ').split()
            
            if len(inp_split) == 1:
                return int(inp_split[0]), None
                
            ra, dec = np.asarray(inp_split, dtype=float)
            
            cosd = np.cos(self.df['dec']/180*np.pi)
            dx = (self.df['ra'] - ra)*cosd
            dy = (self.df['dec'] - dec)
            dr = np.sqrt(dx**2+dy**2)*3600.
            imin = np.nanargmin(dr)
            
            return self.df['id'][imin], dr[imin]
            

        @app.callback([Output('object-sed-figure', 'figure'),
                       Output('object-info', 'children'), 
                       Output('match-sep', 'children'), 
                       Output('cutout-figure', cutout_target)], 
                      [Input('sample-selection-scatter', 
                                               'hoverData'), 
                       Input('sed-unit-selector', 'value'),
                       Input('id-input', 'value')])
        def update_object_sed(hoverData, sed_unit, id_input):
            """
            SED + p(z) plot
            """
            id_i, dr_i = parse_id_input(id_input)
            if id_i is None:
                id_i = hoverData['points'][0]['customdata'][0]
            else:
                if id_i not in self.zout['id']:
                    id_i = hoverData['points'][0]['customdata'][0]
            
            if dr_i is None:
                match_sep = ''
            else:
                match_sep = f'{dr_i:.1f}"'
                        
            show_fnu = {'Fλ':0, 'Fν':1, 'νFν':2}
            
            layout_kwargs = dict(template=template, 
                                 autosize=True, 
                                 showlegend=False, 
                                 margin=dict(l=0,r=0,b=0,t=20,pad=0,
                                               autoexpand=True))
                              
            fig = self.photoz.show_fit_plotly(id_i,
                                              show_fnu=show_fnu[sed_unit], 
                                              vertical=True, 
                                              panel_ratio=[0.6, 0.4],
                                              show=False,
                                              layout_kwargs=layout_kwargs)
            
            if plot_height is not None:
                fig.update_layout(height=plot_height)
            
            
            ix = self.df['id'] == id_i
            if ix.sum() == 0:
                object_info = 'ID: N/A'
            else:
                ix = np.where(ix)[0][0]
                ra, dec = self.df['ra'][ix], self.df['dec'][ix]
                object_info = [f'ID: {id_i}  |  α, δ = {ra:.6f} {dec:.6f} ',
                               # ' | ', html.A('ESO',
                               #               href=utils.eso_query(ra, dec,
                               #                             radius=1.0,
                               #                             unit='s')),
                               ' | ', html.A('CDS', 
                                             href=utils.cds_query(ra, dec, 
                                                           radius=1.0,
                                                           unit='s')),
                               ' | ', html.A('Cutout',
                                          href=self.cutout_url.format(ra=ra, dec=dec)),
                               ' | ', get_map_link(ra, dec),
                               html.Br(), 
                               f"z_phot: {self.df['z_phot'][ix]:.3f}  ", 
                               f" | z_spec: {self.df['z_spec'][ix]:.3f}", 
                               html.Br(),  
                               f"mag: {self.df['mag'][ix]:.2f}  ", 
                               f" | mass: {self.df['mass'][ix]:.2f} ",
                               f" | sSFR: {self.df['ssfr'][ix]:.2f}", 
                               html.Br()]
            
            
            if self.cutout_data is not None:
                cutout_fig = hdu_cutout_figure(id_i)
            elif api_filters is not None:
                cutout_fig = api_cutout_figure(id_i)
            else:
                cutout_fig = ['']
                
            return fig, object_info, match_sep, cutout_fig
        
        ####### App layout
        app.layout = html.Div(content)
        
        if server_mode is not None:
            # app.run_server(mode=server_mode, port=port)
            app.run(jupyter_mode=server_mode, port=port)
        
        return app    


