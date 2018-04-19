import os
import numpy as np
import matplotlib.pyplot as plt

def path_to_eazy_data():
    return os.path.join(os.path.dirname(__file__), 'data')
    
def running_median(xi, yi, NBIN=10, use_median=True, use_nmad=True, reverse=False, bins=None):
    """
    Running median/biweight/nmad
    """
    import numpy as np
    import astropy.stats
    
    NPER = xi.size // NBIN
    if bins is None:
        so = np.argsort(xi)
        if reverse:
            so = so[::-1]
        
        bx = np.linspace(0,len(xi),NBIN+1)
        bins = np.interp(bx, np.arange(len(xi)), xi[so])
        if reverse:
            bins = bins[::-1]
    
    NBIN = len(bins)-1
           
    xm = np.arange(NBIN)*1.
    xs = xm*0
    ym = xm*0
    ys = xm*0
    N = np.arange(NBIN)
        
    #idx = np.arange(NPER, dtype=int)
    for i in range(NBIN):
        in_bin = (xi > bins[i]) & (xi <= bins[i+1])
        N[i] = in_bin.sum() #N[i] = xi[so][idx+NPER*i].size
        
        if use_median:
            xm[i] = np.median(xi[in_bin]) # [so][idx+NPER*i])
            ym[i] =  np.median(yi[in_bin]) # [so][idx+NPER*i])
        else:
            xm[i] = astropy.stats.biweight_location(xi[in_bin]) # [so][idx+NPER*i])
            ym[i] = astropy.stats.biweight_location(yi[in_bin]) # [so][idx+NPER*i])
            
        if use_nmad:
            mad = astropy.stats.median_absolute_deviation
            ys[i] = 1.48*mad(yi[in_bin]) # [so][idx+NPER*i])
        else:
            ys[i] = astropy.stats.biweight_midvariance(yi[in_bin]) # [so][idx+NPER*i])
            
    return xm, ym, ys, N

def nmad(arr):
    import astropy.stats
    return 1.48*astropy.stats.median_absolute_deviation(arr)

def log_zgrid(zr=[0.7,3.4], dz=0.01):
    """Make a logarithmically spaced redshift grid
    
    Parameters
    ----------
    zr : [float, float]
        Minimum and maximum of the desired grid
    
    dz : float
        Step size, dz/(1+z)
    
    Returns
    -------
    zgrid : array-like
        Redshift grid
    
    """
    zgrid = np.exp(np.arange(np.log(1+zr[0]), np.log(1+zr[1]), dz))-1
    return zgrid
    
def clipLog(im, lexp=1000, cmap=[-1.4914, 0.6273], scale=[-0.1,10]):
    """
    Return normalized array like DS9 log
    """
    import numpy as np
    
    contrast, bias = cmap
    clip = (np.clip(im, scale[0], scale[1])-scale[0])/(scale[1]-scale[0])
    clip_log = np.clip((np.log10(lexp*clip+1)/np.log10(lexp)-bias)*contrast+0.5, 0, 1)
    
    return clip_log

def get_irsa_dust(ra=53.1227, dec=-27.805089, type='SandF'):
    """
    Get Galactic dust reddening from NED/IRSA at a given position
    http://irsa.ipac.caltech.edu/applications/DUST/docs/dustProgramInterface.html
    
    Parameters
    ----------
    ra, dec : float
        RA/Dec in decimal degrees.
        
    type : 'SFD' or 'SandF'
        Dust model, with        
            SandF = Schlafly & Finkbeiner 2011 (ApJ 737, 103) 
              SFD = Schlegel et al. 1998 (ApJ 500, 525)
    
    Returns
    -------
    ebv : float
        Color excess E(B-V), in magnitudes
    
    """
    import os
    import tempfile   
    import urllib.request
    from astropy.table import Table
    from lxml import objectify
    
    query = 'http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={0:.4f}+{1:.4f}+equ+j2000'.format(ra, dec)
    
    req = urllib.request.Request(query)
    response = urllib.request.urlopen(req)
    resp_text = response.read().decode('utf-8')
    
    root = objectify.fromstring(resp_text)
    stats = root.result.statistics

    if type == 'SFD':
        return float(str(stats.refPixelValueSFD).split()[0])
    else:
        return float(str(stats.refPixelValueSandF).split()[0])
        
def fill_between_steps(x, y, z, ax=None, *args, **kwargs):
    """
    Make `fill_between` work like linestyle='steps-mid'.
    """
    so = np.argsort(x)
    mid = x[so][:-1] + np.diff(x[so])/2.
    xfull = np.append(np.append(x, mid), mid+np.diff(x[so])/1.e6)
    yfull = np.append(np.append(y, y[:-1]), y[1:])
    zfull = np.append(np.append(z, z[:-1]), z[1:])
    
    so = np.argsort(xfull)
    if ax is None:
        ax = plt.gca()
    
    ax.fill_between(xfull[so], yfull[so], zfull[so], *args, **kwargs)

class emceeChain():     
    def __init__(self, chain=None, file=None, param_names=[],
                       burn_fraction=0.5, sampler=None):
        
        self.param_names = []
        
        if chain is not None:
            self.chain = chain
        
        if file is not None:
            if 'fits' in file.lower():
                self.load_fits(file=file)            
            else:
                self.load_chain(file=file)
                     
        self.process_chain(param_names = param_names,
                           burn_fraction=burn_fraction)
        #
        if sampler is not None:
            from numpy import unravel_index
            max_ix = unravel_index(sampler.lnprobability.argmax(), sampler.lnprobability.shape)
            self.map = self.chain[max_ix[0], max_ix[1],:]
            self.is_map = True
        else:
            self.map = self.median
            self.is_map = False
            
    def process_chain(self, param_names=[], burn_fraction=0.5):
        """
        Define parameter names and get parameter statistics
        """
        self.nwalkers, self.nstep, self.nparam = self.chain.shape
                
        if param_names == []:            
            if self.param_names == []:
                for i in range(self.nparam):
                    param_names.append('a%d' %(i+1))
               
                self.param_names = param_names
        
        else:
            if len(param_names) != self.nparam:
                print('param_names must have N=%d (or zero) entries' %(self.nparam))
                return False
                        
            self.param_names = param_names
                
        self.param_dict = {}
        for i in range(self.nparam):
            self.param_dict[self.param_names[i]] = i
        
        self.nburn = int(np.round(burn_fraction*self.nstep))
        self.stats = {}
        self.median = np.zeros(self.nparam)
                
        for param in self.param_names:
            pid = self.param_dict[param]
            self.stats[param] = self.get_stats(pid, burn=self.nburn)
            self.median[pid] = self.stats[param]['q50']
            
    def get_stats(self, pid, burn=0, raw=False):
        """
        Get percentile statistics for a parameter in the chain
        """
        if raw:
            pchain = pid*1.
        else:
            pchain = self.chain[:,burn:,pid].flatten()
        
        stats = {}
        stats['q05'] = np.percentile(pchain, 5)
        stats['q16'] = np.percentile(pchain, 16)
        stats['q50'] = np.percentile(pchain, 50)
        stats['q84'] = np.percentile(pchain, 84)
        stats['q95'] = np.percentile(pchain, 95)
        stats['mean'] = np.mean(pchain)
        stats['std'] = np.std(pchain)
        stats['width'] = (stats['q84']-stats['q16'])/2.
        return stats
        
    def show_chain(self, param='a1', chain=None, alpha=0.15, color='blue', scale=1, diff=0, ax = None, add_labels=True, hist=False, autoscale=True, *args, **kwargs):
        """
        Make a plot of the chain for a given parameter.
        
        For plotting, multiply the parameter by `scale` and subtract `diff`.
        
        """
        if chain is None:
            pid = self.param_dict[param]
            chain = self.chain[:,:,pid]
        
        if ax is not None:
            plotter = ax
            xlabel = ax.set_xlabel
            ylabel = ax.set_ylabel
            ylim = ax.set_ylim
        else:
            plotter = plt
            xlabel = plt.xlabel
            ylabel = plt.ylabel
            ylim = plt.ylim
            
        if hist:
            h = plotter.hist(chain[:,self.nburn:].flatten(), alpha=alpha, color=color, *args, **kwargs)
            if add_labels:
                ylabel('N')
                xlabel(param)
        else:
            for i in range(self.nwalkers):
                p = plotter.plot(chain[i,:]*scale-diff, alpha=alpha, color=color, *args, **kwargs)
            if add_labels:
                xlabel('Step')
                ylabel(param)
            #
            if autoscale:
                ylim(self.stats[param]['q50']*scale + np.array([-8,8])*self.stats[param]['width']*scale)
            
    def save_chain(self, file='emcee_chain.pkl', verbose=True):
        """
        Save the chain to a Pkl file
        """
        import cPickle as pickle
        
        fp = open(file,'wb')
        pickle.dump(self.nwalkers, fp)
        pickle.dump(self.nstep, fp)
        pickle.dump(self.nparam, fp)
        pickle.dump(self.param_names, fp)
        pickle.dump(self.chain, fp)
        fp.close()
        
        if verbose:
            print('Wrote %s.' %(file))
        
    def load_chain(self, file='emcee_chain.pkl'):
        """
        Read the chain from the pickle file
        """
        import cPickle as pickle
        
        fp = open(file, 'rb')
        self.nwalkers = pickle.load(fp)
        self.nstep = pickle.load(fp)
        self.nparam = pickle.load(fp)
        self.param_names = pickle.load(fp)
        self.chain = pickle.load(fp)
        fp.close()
    
    def save_fits(self, file='emcee_chain.fits', verbose=True):
        """
        Make a FITS file of an EMCEE chain
        """
        header = pyfits.Header()
        header.update('NWALKERS', self.nwalkers)
        header.update('NSTEP', self.nstep)
        header.update('NPARAM', self.nparam)
        
        hdu = [pyfits.PrimaryHDU(header=header)]
        
        for param in self.param_names:
            header.update('PARAM', param)
            hdu.append(pyfits.ImageHDU(data=self.__getitem__(param), header=header, name=param))
        
        hduList = pyfits.HDUList(hdu)
        hduList.writeto(file, clobber=True, output_verify='silentfix')
    
        if verbose:
            print('Wrote %s.' %(file))
    
    def load_fits(self, file='emcee_chain.fits'):
        im = pyfits.open(file)
        self.nwalkers = im[0].header['NWALKERS']
        self.nstep = im[0].header['NSTEP']
        self.nparam = im[0].header['NPARAM']
        self.param_names = []
        self.chain = np.ones((self.nwalkers, self.nstep, self.nparam))
        for i in range(self.nparam):
            self.param_names.append(im[i+1].header['PARAM'])
            self.chain[:,:,i] = im[i+1].data
        
        im.close()
    
    def parameter_correlations(self, size=8, shrink=5, show=None, file=None):
        if show is None:
            show = self.param_names
        
        NP = len(show)
        fig = unicorn.plotting.plot_init(square=True, aspect=1, xs=size, left=0.05, right=0.01, bottom=0.01, top=0.01, NO_GUI=False, use_tex=False, fontsize=7)
        fig.subplots_adjust(wspace=0.0,hspace=0.0)
        
        counter = 0
        for i in range(NP):
            for j in range(NP):
                counter = counter + 1
                ax = fig.add_subplot(NP, NP, counter)
                a = ax.plot(self[show[i]][:,self.nburn::shrink].flatten(), self[show[j]][:,self.nburn::shrink].flatten(), alpha=0.03, color='black', linestyle='None', marker=',')
                a = ax.set_xlim(self.stats[show[i]]['q50']-3*self.stats[show[i]]['std'], self.stats[show[i]]['q50']+3*self.stats[show[i]]['std'])
                a = ax.set_ylim(self.stats[show[j]]['q50']-3*self.stats[show[j]]['std'], self.stats[show[j]]['q50']+3*self.stats[show[j]]['std'])
                if i == j:
                    a = ax.text(0.5, 0.92, show[i], fontsize=8, color='red', horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

        if file is not None:
            fig.savefig(file)
    
    def draw_random(self, N=10):
        """
        Draw random sets of parameters from the chain
        """
        #ok_walk = self.sampler.acceptance_fraction > min_acceptance
        iwalk = np.cast[int](np.random.rand(N)*self.nwalkers)
        istep = self.nburn + np.cast[int](np.random.rand(N)*(self.nstep-self.nburn))
        draw = self.chain[iwalk, istep, :]
        return draw
        
    def __getitem__(self, param):
        pid = self.param_dict[param]
        return self.chain[:,:,pid]
        
    def contour(self, p1, p2, labels=None, levels=[0.683, 0.955], colors=None, limits=None, bins=20, ax=None, fill=False, **kwargs):
        """
        Plot sigma contours
        """
        import astroML.plotting
        
        if isinstance(p1, str):
            trace1 = self.__getitem__(p1)[:,self.nburn:].flatten()
            pname1 = p1
        else:
            trace1 = p1.flatten()
            pname1 = ''
        
        if isinstance(p2, str):
            trace2 = self.__getitem__(p2)[:,self.nburn:].flatten()
            pname2 = p2
        else:
            trace2 = p2.flatten()
            pname2 = ''
        
        if labels is None:
            labels = [pname1, pname2]
            
        if limits is None:
            limits = [(t.min(), t.max()) for t in [trace1, trace2]]
        
        bins = [np.linspace(limits[i][0], limits[i][1], bins + 1)
                for i in range(2)]
        
        
        H, xbins, ybins = np.histogram2d(trace1, trace2, bins=(bins[0], bins[1]))
        H[H == 0] = 1E-16
        Nsigma = astroML.plotting.mcmc.convert_to_stdev(np.log(H))
        
        if ax is None:
            ax = plt
        
        ax.contour(0.5 * (xbins[1:] + xbins[:-1]),
                   0.5 * (ybins[1:] + ybins[:-1]),
                   Nsigma.T, levels=levels, **kwargs)
        
        if fill:
            if ax is plt:
                col = plt.gca().collections
            else:
                col = ax.collections
            
            n_levels = len(levels)
            if colors is None:
                dc = 1./(n_levels+1)
                colors = ['%.2f' %((i+1)*dc) for i in np.arange(n_levels)]
                print(colors)

            for i in range(n_levels): 
                print(colors[i])
                col[i].set_facecolor(colors[i])
                col[i].set_edgecolor(colors[i])
                col[i].set_zorder(-10-i)
                col[i].set_alpha(0.8)
                
                
        if ax is plt:
            ax.xlabel(labels[0])
            ax.ylabel(labels[1])
            ax.xlim(limits[0])
            ax.ylim(limits[1])
        else:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
