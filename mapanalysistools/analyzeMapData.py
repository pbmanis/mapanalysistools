from __future__ import print_function
from __future__ import absolute_import
"""

This code was originally built to process the TTX experiments with maps directly from the data, but
may be otherwise useful...

Add this to your .bash_profile
    export PYTHONPATH="path_to_acq4:${PYTHONPATH}"
For example:
    export PYTHONPATH="/Users/pbmanis/Desktop/acq4:${PYTHONPATH}"
"""


import sys
import sqlite3
import matplotlib
import pyqtgraph.multiprocess as mp

import argparse
import numpy as np
import scipy.signal
import scipy.ndimage

import os.path
from collections import OrderedDict

import re
import math
import dill as pickle
import datetime
import timeit
import ephysanalysis as EP
import montage as MONT
import minis

#from pyqtgraph.metaarray import MetaArray

from mapanalysistools import functions
import mapanalysistools.digital_filters as FILT
from minis import minis_methods
import matplotlib.pyplot as mpl
import matplotlib.colors
import matplotlib
import matplotlib.collections as collections
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from  matplotlib import colors as mcolors
import matplotlib.cm
from mapanalysistools import colormaps
import pylibrary.PlotHelpers as PH

color_sequence = ['k', 'r', 'b']
colormapname = 'parula'
# ANSI terminal colors  - just put in as part of the string to get color terminal output
colors = {'red': '\x1b[31m', 'yellow': '\x1b[33m', 'green': '\x1b[32m', 'magenta': '\x1b[35m',
              'blue': '\x1b[34m', 'cyan': '\x1b[36m' , 'white': '\x1b[0m', 'backgray': '\x1b[100m'}

basedir = "/Users/pbmanis/Desktop/Python/mapAnalysisTools"

re_degree = re.compile('\s*(\d{1,3}d)\s*')
re_duration = re.compile('(\d{1,3}ms)')

# print ('maps: ', colormaps)
# print(dir(colormaps))
def setMapColors(colormapname, reverse=False):
    from mapanalysistools import colormaps
    if colormapname == 'terrain':
        cm_sns = mpl.cm.get_cmap('terrain_r')  # terrain is not bad    #
    elif colormapname == 'gray':
        cm_sns = mpl.cm.get_cmap('gray')  # basic gray scale map
    # elif colormap == 'cubehelix':
    #     cm_sns = seaborn.cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0,
    #         hue=0.8, light=0.85, dark=0.15, reverse=reverse, as_cmap=False)
    # elif colormap == 'snshelix':
    #     cm_sns = seaborn.cubehelix_palette(n_colors=64, start=3, rot=0.5, gamma=1.0, dark=0, light=1.0, reverse=reverse,
    #      as_cmap=True)
    elif colormapname == 'a':
        from colormaps import option_a
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_a', colormaps.option_a.cm_data)
    elif colormapname == 'b':
        import colormaps.option_b
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_b', colormaps.option_b.cm_data)
    elif colormapname == 'c':
        import colormaps.option_c
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_c', colormaps.option_c.cm_data)
    elif colormapname == 'd':
        import colormaps.option_a
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_d', colormaps.option_d.cm_data)
    elif colormapname == 'parula':
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('parula', colormaps.parula.cm_data)
    else:
        print('Unrecongnized color map {0:s}; setting to "parula"'.format(colormapname))
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('parula', colormaps.parula.cm_data)
    return cm_sns    

cm_sns = setMapColors('parula')

# arc collection form https://gist.github.com/syrte/592a062c562cd2a98a83
# retrieved 10/5/2018

def wedges(x, y, w, h=None, theta1=0.0, theta2=360.0,
         c='b', **kwargs):
    """
    Make a scatter plot of Wedges. 
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of ellipses.
    w, h : scalar or array_like, shape (n, )
        Total length (diameter) of horizontal/vertical axis.
        `h` is set to be equal to `w` by default, ie. circle.

    theta1 : float 
        start angle in degrees
    theta2 : float
        end angle in degrees
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    plt.figure()
    x = np.arange(20)
    y = np.arange(20)
    arcs(x, y, 3, h=x, c = x, rot=0., theta1=0., theta2=35.)
    plt.show()
         
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w

    zipped = np.broadcast(x, y, w, h, theta1, theta2)
    patches = [Wedge((x_,y_), w_, t1_, t2_)
               for x_, y_, w_, h_, t1_, t2_ in zipped]

    collection = PatchCollection(patches, **kwargs)

    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)
    return collection


class AnalyzeMap(object):
    
    def __init__(self, rasterize=True):
        self.AR = EP.acq4read.Acq4Read()
        self.SP = EP.SpikeAnalysis.SpikeAnalysis()
        self.RM = EP.RmTauAnalysis.RmTauAnalysis()
        self.MT = MONT.montager.Montager()
        self.last_dataset = None
        self.last_results = None
        self.LPF = 5000.
        self.notch = False
        self.notch_freqs = [60., 120., 180., 240.]
        self.lbr_command = False  # laser blue raw waveform (command)
        self.photodiode = False  # photodiode waveform (recorded)
        self.fix_photodiode_artifact = True
        self.response_window = 0.030  # seconds
        self.direct_window = 0.001
        # set some defaults - these will be overwrittein with readProtocol
        self.twin_base = [0., 0.295]
        self.twin_resp = [[0.300+self.direct_window, 0.300 + self.response_window]]
       # self.taus = [0.5, 2.0]
        self.taus = [0.0002, 0.005]
        self.threshold = 3.0
        self.sign = -1  # negative for EPSC, positive for IPSC
        self.scale_factor = 1 # scale factore for data (convert to pA or mV,,, )
        self.overlay_scale = 0.
        self.shutter_artifact = [0.055]
        self.artifact_suppress = True
        self.stepi = 20.
        self.datatype = 'I'  # 'I' or 'V' for Iclamp or Voltage Clamp
        self.rasterize = rasterize
        self.methodname = 'aj'  # default event detector
        self.colors = {'red': '\x1b[31m', 'yellow': '\x1b[33m', 'green': '\x1b[32m', 'magenta': '\x1b[35m',
              'blue': '\x1b[34m', 'cyan': '\x1b[36m' , 'white': '\x1b[0m', 'backgray': '\x1b[100m'}
        self.MA = minis.minis_methods.MiniAnalyses()  # get a minianalysis instance

    def set_notch(self, notch, freqs=[60]):
        self.notch = notch
        self.notch_freqs = freqs
    
    def set_LPF(self, LPF):
        self.LPF = LPF

    def set_methodname(self, methodname):
        if methodname in ['aj', 'AJ']:
            self.methodname = 'aj'
        elif methodname in ['cb', 'CB']:
            self.methodname = 'cb'
        else:
            raise ValueError("Selected event detector %s is not valid" % methodname)

    def set_taus(self, taus):
        if len(taus) != 2:
            raise ValueError('Analyze Map Data: need two tau values in list!, got: ', taus)
        self.taus = sorted(taus)

    def set_shutter_artifact_time(self, t):
        self.shutter_artifact = t
    
    def set_artifact_suppression(self, suppr=True):
        if not isinstance(suppr, bool):
            raise ValueError('analyzeMapData: artifact suppresion must be True or False')
        self.artifact_suppress = suppr
        

    def readProtocol(self, protocolFilename, records=None, sparsity=None, getPhotodiode=False):
        starttime = timeit.default_timer()
        self.protocol = protocolFilename
        print('Protocol: ', protocolFilename)
        self.AR.setProtocol(protocolFilename)
        if not self.AR.getData():
            print('  >>No data found in protocol: %s' % protocolFilename)
            return None, None, None, None
        #print('Protocol: ', protocolFilename)
        self.datatype = self.AR.mode[0].upper()  # get mode and simplify to I or V
        if self.datatype == 'I':
            self.stepi = 2.0
        else:
            self.stepi = 20.0
        self.stimtimes = self.AR.getBlueLaserTimes()
        if self.stimtimes is not None:
            self.twin_base = [0., self.stimtimes['start'][0] - 0.001]  # remember times are in seconds
            self.twin_resp = []
            for j in range(len(self.stimtimes['start'])):
                self.twin_resp.append([self.stimtimes['start'][j]+self.direct_window, self.stimtimes['start'][j]+self.response_window])
        self.lbr_command = self.AR.getLaserBlueCommand() # just get flag; data in self.AR
        try:
            self.photodiode = self.AR.getPhotodiode()
        except:
            pass
        self.shutter = self.AR.getDeviceData('Laser-Blue-raw', 'Shutter')
        self.AR.getScannerPositions()
        data = np.reshape(self.AR.traces, (self.AR.repetitions, self.AR.traces.shape[0], self.AR.traces.shape[1]))
        endtime = timeit.default_timer()
        print("    Reading protocol {0:s} took {1:6.1f} s".format(protocolFilename.name, endtime-starttime))
        return data, self.AR.time_base, self.AR.sequenceparams, self.AR.scannerinfo

    def show_images(self, show = True):
        r, c = PH.getLayoutDimensions(len(self.images), pref='height')
        f, ax = mpl.subplots(r, c)
        self.figure_handle = f
        f.suptitle(self.celldir.replace('_', '\_'), fontsize=9)
        ax = ax.ravel()
        PH.noaxes(ax)
        for i, img in enumerate(self.imagedata):
#            print (self.imagemetadata[i])
            fna, fne = os.path.split(self.images[i])
            imfig = ax[i].imshow(self.gamma_correction(img, 2.2))
            PH.noaxes(ax[i])
            ax[i].set_title(fne.replace('_', '\_'), fontsize=8)
            imfig.set_cmap(self.cmap)
        if show:
            mpl.show()

    def set_analysis_windows(self):
        pass

    def calculate_charge(self, tb, data, twin_base=[0, 0.1], twin_resp=[[0.101, 0.130]]):
        """
        One-dimensional data...
        """
        # get indices for the integration windows
        tbindx = np.where((tb >= twin_base[0]) & (tb < twin_base[1]))
        trindx = np.where((tb >= twin_resp[0]) & (tb < twin_resp[1]))
        Qr = 1e6*np.sum(data[trindx])/(twin_resp[1]-twin_resp[0]) # response
        Qb = 1e6*np.sum(data[tbindx])/(twin_base[1]-twin_base[0])  # baseline
        return Qr, Qb

    def ZScore(self, tb, data, twin_base=[0, 0.1], twin_resp=[[0.101, 0.130]]):
        # abs(post.mean() - pre.mean()) / pre.std()
        # get indices for the integration windows
        tbindx = np.where((tb >= twin_base[0]) & (tb < twin_base[1]))
        
        trindx = np.where((tb >= twin_resp[0]) & (tb < twin_resp[1]))
        mpost = np.mean(data[trindx]) # response
        mpre = np.mean(data[tbindx])  # baseline
        try:
            zs = np.fabs((mpost-mpre)/np.std(data[tbindx]))
        except:
            zs = 0
        return zs

    def Imax(self, tb, data, twin_base=[0, 0.1], twin_resp=[[0.101, 0.130]], sign=1):

        tbindx = np.where((tb >= twin_base[0]) & (tb < twin_base[1]))
        trindx = np.where((tb >= twin_resp[0]) & (tb < twin_resp[1]))
        mpost = np.max(sign*data[trindx]) # response goes negative... 
        return(mpost)

    def select_events(self, pkt, tstarts, tdurs, rate, mode='reject', thr=5e-12, data=None, first_only=False):
        """
        return indices where the input index is outside (or inside) a set of time windows.
        tstarts is a list of window starts
        twin is the duration of each window
        rate is the data sample rate (in msec...)
        pkt is the list of times to compare against.
        """

        # print('rate: ', rate)
        if mode in ['reject', 'threshold_reject']:
            npk = list(range(len(pkt)))  # assume all
        else:
            npk = []
        for itw, tw in enumerate(tstarts): # and for each stimulus
            first = False
            if isinstance(tdurs, list) or isinstance(tdurs, np.ndarray):  # either use array parallel to tstarts, or
                ttwin = tdurs[itw]
            else:
                ttwin = tdurs  # or use just a single value
            ts = int(tw/rate)
            te = ts + int(ttwin/rate)
            for k, pk in enumerate(pkt):  # over each index
                if mode == 'reject' and npk[k] is None:  # means we have already rejected the n'th one
                    continue
                if mode == 'reject':
                    if pk >= ts and pk <  te:
                        npk[k] = None
                elif (mode == 'threshold_reject') and (data is not None):
                    if (pk >= ts) and (pk <  te) and (np.fabs(data[k]) < thr):
                        print('np.fabs: ', np.fabs(data[k]), thr)
                        npk[k] = None
                elif mode == 'accept':
                  #  print(t0, pkt[k], te)
                    if pk >= ts and pk < te and not first:
                        if k not in npk:
                            npk.append(k)
                    if first_only and not first:
                        first = True
                        break
                        
                else:
                    raise ValueError('analyzeMapData:select_times: mode must be accept, threshold_reject, or reject; got: %s' % mode)
        npks = [n for n in npk if n is not None]  # return the truncated list of indices into pkt
        return npks
        
    def select_by_sign(self, method, npks, data, min_event=5e-12):
        """
        Screen events for correct sign and minimum amplitude.
        Here we use the onsets and smoothed peak to select
        for events satisfying criteria.
        
        Parameters
        ----------
        method : object (mini_analysis object)
            result of the mini analysis. The object must contain 
            at least two lists, one of onsets and one of the smoothed peaks. 
            The lists must be of the same length.
        
        data : array 
            a 1-D array of the data to be screened. This is the entire
            trace.
        
        event_min : float (default 5e-12)
            The smallest size event that will be considered acceptable.
        """
            
        pkt = []
        if len(method.onsets) == 0:
            return(pkt)
        tb = method.timebase  # full time base
        smpks = np.array(method.smpkindex)
        # events[trial]['aveventtb']
        rate = np.mean(np.diff(tb))
        tb_event = method.avgeventtb  # event time base
        tpre = 0.002 # 0.1*np.max(tb0)
        tpost = np.max(method.avgeventtb)-tpre
        ipre = int(tpre/rate)
        ipost = int(tpost/rate)
        #tb = np.arange(-tpre, tpost+rate, rate) + tpre
        pt_fivems = int(0.0005/rate)
        pk_width = int(0.0005/rate/2)

        # from pyqtgraph.Qt import QtGui, QtCore
        # import pyqtgraph as pg
        # pg.setConfigOption('leftButtonPan', False)
        # app = QtGui.QApplication([])
        # win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
        # win.resize(1000,600)
        # win.setWindowTitle('pyqtgraph example: Plotting')
        # # Enable antialiasing for prettier plots
        # pg.setConfigOptions(antialias=True)
        # p0 = win.addPlot(0, 0)
        # p1 = win.addPlot(1, 0)
        # p1.plot(tb, data[:len(tb)]) # whole trace
        # p1s = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 255))
        # p0s = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 255, 255))

        for npk, jevent in enumerate(np.array(method.onsets[npks])):
            jstart = jevent - ipre
            jpeak = method.smpkindex[npk]
            jend = jevent + ipost + 1
            evdata = data[jstart:jend].copy()
            l_expect = jend - jstart
           # print('data shape: ', evdata.shape[0], 'expected: ', l_expect)
            
            if evdata.shape[0] == 0 or evdata.shape[0] < l_expect:
               # print('nodata', evdata.shape[0], l_expect)
                continue
            bl = np.mean(evdata[:pt_fivems])
            evdata -= bl
            # p0.plot(tb_event, evdata)  # plot every event we consider
            # p1s.addPoints(x=[tb[jpeak]], y=[data[jpeak]])
            # next we make a window over which the data will be averaged to test the ampltiude
            left = jpeak - pk_width
            right = jpeak + pk_width
            left = max(0, left)
            right = min(right, len(data))
            if right-left == 0:  # peak and onset cannot be the same
                # p0s.addPoints(x=[tb_event[jpeak-jstart]], y=[evdata[jpeak-jstart]], pen=pg.mkPen('y'), symbolBrush=pg.mkBrush('y'), symbol='o', size=6)
                # print('r - l = 0')
                continue
            if (self.sign < 0 ) and (np.mean(data[left:right]) > self.sign*min_event):  # filter events by amplitude near peak
                # p0s.addPoints(x=[tb_event[jpeak-jstart]], y=[evdata[jpeak-jstart]], pen=pg.mkPen('y'), symbolBrush=pg.mkBrush('y'), symbol='o', size=6)
              #  print('data pos, sign neg', np.mean(data[left:right]))
                continue
            if (self.sign >= 0) and (np.mean(data[left:right]) < self.sign*min_event):
                #p0s.addPoints([tb_event[jpeak-jstart]], [evdata[jpeak-jstart]], pen=pg.mkPen('y'), symbolBrush=pg.mkBrush('y'), symbol='o', size=6)
               # print('data neg, sign pos', np.mean(data[left:right]))
                continue
            # print('dataok: ', jpeak)
            pkt.append(npk)  # build array through acceptance.
        #     p0s.addPoints([tb_event[jpeak-jstart]], [evdata[jpeak-jstart]], pen=pg.mkPen('b'), symbolBrush=pg.mkBrush('b'), symbol='o', size=4)
        # p1s.addPoints(tb[smpks[pkt]], data[smpks[pkt]], pen=pg.mkPen('r'), symbolBrush=pg.mkBrush('r'), symbol='o', size=4)
        # p1.addItem(p1s)
        # p0.addItem(p0s)
        # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #     QtGui.QApplication.instance().exec_()

        return pkt

    def filter_data(self, tb, data):
        filtfunc = scipy.signal.filtfilt
        rate = np.mean(np.diff(tb))  # t is in seconds, so freq is in Hz
        self.rate = rate
        samplefreq = 1.0/rate
        nyquistfreq = samplefreq/1.95
        wn = self.LPF/nyquistfreq 
        b, a = scipy.signal.bessel(2, wn)
        if data.ndim == 3:
            for r in range(data.shape[0]):
                for t in range(data.shape[1]):
                    data[r,t,:] = filtfunc(b, a, data[r, t, :] - np.mean(data[r, t, 0:250]))
                    if self.notch:
                        #print('notching', self.notch_freqs)
                        data[r,t,:] = FILT.NotchFilter(data[r, t, :], notchf=self.notch_freqs, Q=90., QScale=False, samplefreq=samplefreq)
        else:
            data= filtfunc(b, a, data - np.mean(data[0:250]))
            if self.notch:
                #print('notching', self.notch_freqs)
                data = FILT.NotchFilter(data, notchf=self.notch_freqs, Q=90., QScale=False, samplefreq=samplefreq)
            
        return data

    def analyze_protocol(self, data, tb, info,  eventstartthr=None, eventhist=True, testplots=False, 
                dataset=None, data_nostim=None):
        """
        data_nostim is a list of points where the stimulus/response DOES NOT occur, so we can compute the SD
        for the threshold in a consistent manner if there are evoked responses in the trace.
        """

        use_AJ = False
        if self.methodname in ['aj', 'AJ']:
            use_AJ = True 
        aj = minis_methods.AndradeJonas()
        cb = minis_methods.ClementsBekkers()
        data = self.filter_data(tb, data)
        rate = self.rate
        mdata = np.mean(data, axis=0)  # mean across ALL reps
#        rate = rate*1e3  # convert rate to msec

        # make visual maps with simple scores
        nstim = len(self.twin_resp)
        self.nstim = nstim
        # find max position stored in the info dict
        pmax = len(list(info.keys()))
        Qr = np.zeros((nstim, data.shape[1]))
        Qb = np.zeros((nstim, data.shape[1]))
        Zscore = np.zeros((nstim, data.shape[1]))
        I_max = np.zeros((nstim, data.shape[1]))
        pos = np.zeros((data.shape[1], 2))
        infokeys = list(info.keys())
        for ix, t in enumerate(range(data.shape[1])):  # compute for each target
            for s in range(len(self.twin_resp)): # and for each stimulus
                Qr[s, t], Qb[s, t] = self.calculate_charge(tb, mdata[t,:], twin_base=self.twin_base, twin_resp=self.twin_resp[s])
                Zscore[s, t] = self.ZScore(tb, mdata[t,:], twin_base=self.twin_base, twin_resp=self.twin_resp[s])
                I_max[s, t] = self.Imax(tb, data[0,t,:], twin_base=self.twin_base, twin_resp=self.twin_resp[s],
                                sign=self.sign)*self.scale_factor  # just the FIRST pass
            try:
                pos[t,:] = [info[infokeys[ix]]['pos'][0], info[infokeys[ix]]['pos'][1]]
            except:
                print(self.colors['red']+('Failed to establish position for t=%d, ix=%d of max values %d,  protocol: %s' %
                    (t, ix, pmax,  self.protocol))+self.colors['white'])
                raise ValueError()

        nr = 0

        key1=[]
        key2=[]
        for ix in infokeys:
            k1, k2 = ix
            key1.append(k1)
            key2.append(k2)
        self.nreps = len(set(list(key1)))
        self.nspots = len(set(list(key2)))
      #  print('Repetitions: {0:d}   Spots in map: {1:d}'.format(self.nreps, self.nspots))
        events = {}
        eventlist = []  # event histogram across ALL events/trials
        nevents = 0
        if eventhist:
            tmaxev = np.max(tb) # msec
            for jtrial in range(data.shape[0]):  # all trials
                result = []
                crit = []
                scale = []
                tpks = []
                smpks = []
                smpksindex = []
                avgev = []
                avgtb = []
                avgnpts = []
                avg_spont = []
                avg_evoked = []
                fit_tau1 = []
                fit_tau2 = []
                fit_amp = []
                spont_dur = []
                evoked_ev = []  # subsets that pass criteria, onset values stored
                spont_ev = []
                for itarget in range(data.shape[1]):  # all targets
                    if use_AJ:
                        jmax = int(tmaxev/rate)
                        aj.setup(tau1=self.taus[0], tau2=self.taus[1], dt=rate, delay=0.0, template_tmax=rate*(jmax-1),
                                sign=self.sign, eventstartthr=eventstartthr)
                        idata = data.view(np.ndarray)[jtrial, itarget, :]
                        meandata = np.mean(idata[:jmax])
                        aj.deconvolve(idata[:jmax]-meandata, data_nostim=data_nostim, 
                                thresh=self.threshold/5., llambda=10., order=7)  # note threshold scaling... 
                        method = aj
                    else:
                        jmax = int((2*self.taus[0] + 3*self.taus[1])/rate)
                        cb.setup(tau1=self.taus[0], tau2=self.taus[1], dt=rate, delay=0.0, template_tmax=rate*(jmax-1),
                                sign=self.sign, eventstartthr=eventstartthr)
                        idata = data.view(np.ndarray)[jtrial, itarget, :]
                        meandata = np.mean(idata[:jmax])
                        res = cb.cbTemplateMatch(idata-meandata, threshold=self.threshold)
                        result.append(res)
                        crit.append(cb.Crit)
                        scale.append(cb.Scale)
                        method = cb
                        
                    # filter out events at times of stimulus artifacts
                    # build array of artifact times first 
                    art_starts = []
                    art_durs = []
                    art_starts = [0.599, self.shutter_artifact]
                    art_durs = [2, 2*rate]
                    for si, s in enumerate(self.stimtimes['start']):
                        art_starts.append(s)
                        art_starts.append(s+self.stimtimes['duration'][si])
                        art_durs.append(2*rate)
                        art_durs.append(2*rate)
                    npk0 = self.select_events(method.smpkindex, art_starts, art_durs, rate, mode='reject')
                    npk4 = self.select_by_sign(method, npk0, idata, min_event=5e-12)  # events must also be of correct sign and min magnitude
 
                    npk = list(set(npk0).intersection(set(npk4))) # only all peaks that pass all tests
                    if not self.artifact_suppress:
                        npk = npk4  # only suppress shutter artifacts  .. exception
                    nevents += len(np.array(method.onsets)[npk])
                    result.append(np.array(method.onsets)[npk])
                    eventlist.append(tb[np.array(method.onsets)[npk]])
                    tpks.append(np.array(method.peaks)[npk])
                    smpks.append(np.array(method.smoothed_peaks)[npk])
                    smpksindex.append(np.array(method.smpkindex)[npk])
                    spont_dur.append(self.stimtimes['start'][0])  # window until the FIRST stimulus

                    if method.averaged:  # grand average, calculated after deconvolution
                        avgev.append(method.avgevent)
                        avgtb.append(method.avgeventtb)
                        avgnpts.append(method.avgnpts)
                    else:
                        avgev.append([])
                        avgtb.append([])
                        avgnpts.append(0)
                    
                    # define:
                    # spont is not in evoked window, and no sooner than 10 msec before a stimulus,
                    # at least 4*tau[0] after start of trace, and 5*tau[1] before end of trace
                    # evoked is after the stimulus, in a window (usually ~ 5 msec)
                    # data for events are aligned on the peak of the event, and go 4*tau[0] to 5*tau[1]
                    # stimtimes: dict_keys(['start', 'duration', 'amplitude', 'npulses', 'period', 'type'])
                    st_times = np.array(self.stimtimes['start'])
                    ok_events = np.array(method.smpkindex)[npk]
                   # print(ok_events*rate)

                    win_end = 0.015
                    npk_ev = self.select_events(ok_events, st_times, win_end, rate, mode='accept', first_only=True)

                    ev_onsets = np.array(method.onsets)[npk_ev]
                    evoked_ev.append([np.array(method.onsets)[npk_ev], np.array(method.smpkindex)[npk_ev]])
                    avg_evoked_one, avg_evokedtb, allev_evoked = method.average_events(ev_onsets)
                    fit_tau1.append(method.fitted_tau1)  # these are the average fitted values for the i'th trace
                    fit_tau2.append(method.fitted_tau2)
                    fit_amp.append(method.Amplitude)
                    avg_evoked.append(avg_evoked_one)
                    txb = avg_evokedtb  # only need one of these.
                    if not np.isnan(method.fitted_tau1):
                        npk_sp = self.select_events(ok_events, [0.], st_times[0]-(method.fitted_tau1*5), rate, mode='accept')
                        sp_onsets = np.array(method.onsets)[npk_sp]
                        avg_spont_one, avg_sponttb, allev_spont = method.average_events(sp_onsets)
                        avg_spont.append(avg_spont_one)
                        spont_ev.append([np.array(method.onsets)[npk_sp], np.array(method.smpkindex)[npk_sp]])
                    else:
                        spont_ev.append([])
                    
                    if testplots:
                        method.plots(title='%d' % i, events=None)

                events[jtrial] = {'criteria': crit, 'result': result, 'peaktimes': tpks, 'smpks': smpks, 'smpksindex': smpksindex, 
                    'avgevent': avgev, 'avgtb': avgtb, 'avgnpts': avgnpts, 'avgevoked': avg_evoked, 'avgspont': avg_spont, 'aveventtb': txb,
                    'fit_tau1': fit_tau1, 'fit_tau2': fit_tau2, 'fit_amp': fit_amp, 'spont_dur': spont_dur, 'ntraces': data.shape[1],
                    'evoked_ev': evoked_ev, 'spont_ev': spont_ev}

        return{'Qr': Qr, 'Qb': Qb, 'ZScore': Zscore, 'I_max': I_max, 'positions': pos,
               'aj': aj, 'stimtimes': self.stimtimes, 'events': events, 'eventtimes': eventlist, 'dataset': dataset, 
               'sign': self.sign, 'avg_tau1': method.fitted_tau1, 'avg_tau2': method.fitted_tau2, 'avg_amplitude': method.Amplitude}

    def scale_and_rotate(self, poslist, sign=[1., 1.], scaleCorr=1., scale=1e6, autorotate=False, angle=0.):
        """ 
        Angle is in radians
        """
        poslist = [tuple([sign[0]*p[0]*scale*scaleCorr, sign[1]*p[1]*scale*scaleCorr]) for p in poslist]
        posl = np.array(poslist, dtype=[('x', float), ('y', float)]).view(np.recarray)

        newpos = np.array(poslist)

        # get information to do a rotation to horizontal for positions.
        if autorotate:
            iy = np.argsort(posl.y)
            y = posl.y[iy[-3:]]
            x = posl.x[iy[-3:]]
            theta = np.arctan2((y[1]-y[0]), (x[1]-x[0]))
            # perform rotation around 0 using -theta to flatten to top of the array
            c, s = np.cos(-theta), np.sin(-theta)
            rmat = np.matrix([[c, -s], [s, c]]) # rotation matrix
            newpos = np.dot(rmat, newpos.T).T
        if not autorotate and angle != 0.:
            theta = angle
            c, s = np.cos(-theta), np.sin(-theta)
            rmat = np.matrix([[c, -s], [s, c]]) # rotation matrix
            newpos = np.dot(rmat, newpos.T).T
        return newpos

    def reorder(self, a, b):
        """
        make sure that b > a
        if not, swap and return
        """
        if a > b:
            t = b
            b = a
            a = t
        return(a, b)

    def shortName(self, name):
        (h, pr) = os.path.split(name)
        (h, cell) = os.path.split(h)
        (h, sliceno) = os.path.split(h)
        (h, day) = os.path.split(h)
        return(os.path.join(day, sliceno, cell, pr))

    def save_pickled(self, dfile, data):
        now = datetime.datetime.now().isoformat()
        dstruct = {
                    'date': now, 
                    'data': data,
                  }
        print('\nWriting to {:s}'.format(dfile))
        fn = open(dfile, 'wb')
        pickle.dump(dstruct, fn)
        fn.close()

    def read_pickled(self, dfile):
    #    pfname = os.path.join(self.paperpath, pfname) + '.p'
        fn = open(dfile + '.p', 'rb')
        data = pickle.load(fn)
        fn.close()
        return(data)
        
    def plot_timemarker(self, ax):
        """
        Plot a vertical time line marker for the stimuli
        """
        yl = ax.get_ylim()
        for j in range(len(self.stimtimes['start'])):
            t = self.stimtimes['start'][j]
            if isinstance(t, float) and np.diff(yl) > 0: # check that plot is ok to try
                ax.plot([t, t], yl, 'b-', linewidth=0.5, alpha=0.6, rasterized=self.rasterize)

    def plot_events(self, axh, results, colorid=0):
        """
        Plot the events (average) and the histogram of event times
        hist goes into axh
        traces into axt
        """
        plotevents = True
        rotation = 0.
        plotFlag = True
        idn = 0
        self.newvmax = None
        try:
            len(results['eventtimes'])
        except:
            print('There were no results/eventtimes to plot')
            return

        if plotevents and len(results['eventtimes']) > 0:
            nevents = 0
            y=[]
#            print(results['eventtimes'])
            for xn in results['eventtimes']:
                nevents += len(xn)
                y.extend(xn)
            y = np.array(y)
            bins = np.linspace(0., self.AR.tstart, int(self.AR.tstart*1000/2.0)+1)
            axh.hist(y, bins=bins,
                facecolor='k', edgecolor='k', linewidth=0.5, histtype='stepfilled', align='right')
            self.plot_timemarker(axh)
            PH.nice_plot(axh, spines=['left', 'bottom'], position=-0.025, direction='outward', axesoff=False)
            axh.set_xlim(0., self.AR.tstart-0.005)

    def plot_stacked_traces(self, tb, mdata, title, events=None, ax=None):
        if ax is None:
            f, ax = mpl.subplots(1,1)
            self.figure_handle = f
        nevtimes = 0
        for j in range(mdata.shape[0]):
            for i in range(mdata.shape[1]):
                if tb.shape[0] > 0 and mdata[0,i,:].shape[0] > 0:
                    ax.plot(tb, mdata[0, i, :]*self.scale_factor + self.stepi*i, linewidth=0.2,
                            rasterized=self.rasterize, zorder=10)
                if events is not None and j in list(events.keys()):
                    smpki = events[j]['smpksindex'][i]
                    # for k in range(len(smpki)):
                    #     if tb[smpki][k] < 0.6:
                    #         print(f'tb, ev: {i:3d} {k:3d} {tb[smpki][k]:.4f}: {mdata[0,i,smpki][k]*1e12:.1f}')
                    nevtimes += len(smpki)
                    if len(smpki) > 0 and len(tb[smpki]) > 0 and len(mdata[0, i, smpki]) > 0:
                        # The following plot call causes problems if done rasterized. 
                        # See: https://github.com/matplotlib/matplotlib/issues/12003
                        # may be fixed in the future. For now, don't rasterize.
                        ax.plot(tb[smpki], mdata[0, i, smpki]*self.scale_factor + self.stepi*i,
                         'ro', alpha=0.6, markersize=2, markeredgecolor='None', zorder=0, rasterized=False) # self.rasterize)
            
        mpl.suptitle(str(title).replace('_', '\_'), fontsize=8)
        self.plot_timemarker(ax)
        ax.set_xlim(0, self.AR.tstart-0.001)

    def plot_avgevent_traces(self, evtype, mdata=None, events=None, ax=None, scale=1.0, label='pA'):

        if events is None or ax is None:
            return
        nevtimes = 0
        line = {'avgevoked': 'r-', 'avgspont': 'k-'}
        ltitle = {'avgevoked': 'Evoked (%s)'%label, 'avgspont': 'Spont (%s)'%label}
        result_names = {'avgevoked': 'evoked_ev', 'avgspont': 'spont_ev'}
            
        ax.set_ylabel(ltitle[evtype])
        ax.spines['left'].set_color(line[evtype][0])
        ax.yaxis.label.set_color(line[evtype][0])
        ax.tick_params(axis='y', colors=line[evtype][0], labelsize=7)
        ax.tick_params(axis='x', colors=line[evtype][0], labelsize=7)
        ev_min = 5e-12
        sp_min = 5e-12
        if evtype == 'avgevoked':
            eventmin = ev_min
        else:
            eventmin = sp_min
        ave = []
        for trial in events.keys():
           # print('  trail, len(events[trial][evtype]): ', trial, len(events[trial][evtype]))
            tb0 = events[trial]['aveventtb']
            rate = np.mean(np.diff(tb0))
            tpre = 0.002 # 0.1*np.max(tb0)
            tpost = np.max(tb0)
            ipre = int(tpre/rate)
            ipost = int(tpost/rate)
            tb = np.arange(-tpre, tpost+rate, rate) + tpre
            ptfivems = int(0.0005/rate)
            pwidth = int(0.0005/rate/2)
            for itrace, evs in enumerate(events[trial][result_names[evtype]]):  # traces in the evtype list
                if len(evs) == 0:
                    continue
                for jevent in range(len(evs[0])): # evs is 2 element array: [0] are onsets and [1] is peak
                    evdata = mdata[trial, itrace, evs[0][jevent]-ipre:evs[0][jevent]+ipost].copy()  # 0 is onsets
                    if evdata.shape[0] == 0:
                        continue
                    bl = np.mean(evdata[ipre-ptfivems:ipre])
                    evdata -= bl
                    ave.append(evdata)
                    ax.plot(tb[:len(evdata)], scale*evdata, line[evtype], linewidth=0.1, alpha=0.25, rasterized=False)

        nev = len(ave)
        ave = np.array(ave)
        if ave.shape[0] == 0:
            return
        tx = np.broadcast_to(tb, (ave.shape[0], tb.shape[0])).T

        self.MA.set_sign(self.sign)
        avedat = np.mean(ave, axis=0)
        tb = tb[:len(avedat)]
        avebl = np.mean(avedat[:ptfivems])
        avedat = avedat - avebl
        self.MA.fit_average_event(tb, avedat)
        Amplitude = self.MA.fitresult[0]
        tau1 = self.MA.fitresult[1]
        tau2 = self.MA.fitresult[2]
        bfdelay = self.MA.fitresult[3]
        bfit = self.MA.avg_best_fit
        if self.sign == -1:
            amp = np.min(bfit)
        else:
            amp = np.max(bfit)
        txt = f"Amp: {scale*amp:.1f} tau1:{1e3*tau1:.2f} tau2: {1e3*tau2:.2f} (N={ave.shape[0]:d})"
        if evtype == 'avgspont':
            srate = (float(ave.shape[0])/events[0]['spont_dur'][0])/events[0]['ntraces'] # dur should be same for all trials
            txt += f" SR: {srate:.2f} Hz"
        ax.text(0.05, 0.95, txt, fontsize=7, transform=ax.transAxes)
        # ax.plot(tx, scale*ave.T, line[evtype], linewidth=0.1, alpha=0.25, rasterized=False)
        ax.plot(tb, scale*bfit, 'c', linestyle='-', linewidth=0.35, rasterized=self.rasterize)
        ax.plot(tb, scale*avedat, line[evtype], linewidth=0.625, rasterized=self.rasterize)
            #ax.set_label(evtype)

    def plot_average_traces(self, ax, tb, mdata, color='k'):
        """
        Average the traces
        
        Parameters
        ----------
        ax : matplotlib axis object
            axis to plot into
        tb : float array (list or numpy)
            time base for the plot
        mdata : float 2d numpy array
            data to be averaged. Must be trace number x trace
        color : str
            color for trace to be plotter
        
        """
        if mdata is None:
            return
        while mdata.ndim > 1:
            mdata = mdata.mean(axis=0)
#        print('average traces: ', tb.shape, mdata.shape)
        if len(tb) > 0 and len(mdata) > 0:
            ax.plot(tb, mdata*self.scale_factor, color, rasterized=self.rasterize, linewidth=0.6)
        ax.set_xlim(0., self.AR.tstart-0.001)
        return
        # if self.sign < 0:
        #     if self.datatype == 'I':
        #         ax.set_ylim([-20., 20.]) # IPSP negative, CC
        #     else:  # 'V'
        #         ax.set_ylim([-100., 20.])  # EPSC, negative, VC
        # else:
        #     if self.datatype == 'I':
        #         ax.set_ylim([-20., 20.])  # EPSC, positive, current clamp
        #     else: # 'V'
        #         ax.set_ylim([-20., 100.])  # IPSC, positive, VC
                

    def clip_colors(self, cmap, clipcolor):
        cmax = len(cmap)
        colmax = cmap[cmax-1]
        for i in range(cmax):
            if (cmap[i] == colmax).all():
                cmap[i] = clipcolor
        return cmap

    def plot_photodiode(self, ax, tb, pddata, color='k'):
#        print('photodiode: ', tb.shape, np.mean(pddata, axis=0).shape)
        if len(tb) > 0 and len(np.mean(pddata, axis=0)) > 0:
            ax.plot(tb, np.mean(pddata, axis=0), color, rasterized=self.rasterize, linewidth=0.6)
        ax.set_xlim(0., self.AR.tstart-0.001)
        
    def plot_map(self, axp, axcbar, pos, measure, measuretype='I_max', vmaxin=None, 
            imageHandle=None, imagefile=None, angle=0, spotsize=20e-6, cellmarker=False):

        sf = 1.0 # could be 1e-6 ? data in Meters? scale to mm. 
        cmrk = 50e-6*sf # size, microns
        npos = pos.shape[0]
        # if pos.shape[0] > 1:
        #     for n in range(pos.shape[0]):
        #         if pos[n,0] == 0. and pos[n,1] == 0.:
        #             print(colors['red']+'Empty position'+colors['white'])
        #             break
        # else:
        #      n = 0
        npos += 1  # need to count up one more to get all of the points in the data
        pos = pos[:npos,:] *1e3 # clip unused positions   
        
        pz = [np.mean(pos[:,0]), np.mean(pos[:,1])]
        # pos[:,0] = pos[:,0]-pz[0]  # centering... 
        # pos[:,1] = pos[:,1]-pz[1]
            
        if imageHandle is not None and imagefile is not None:
            imageInfo = imageHandle.imagemetadata[0]
            # compute the extent for the image, offsetting it to the map center position
            ext_left = imageInfo['deviceTransform']['pos'][0]*1e3 # - pz[0]
            ext_right = ext_left + imageInfo['region'][2]*imageInfo['deviceTransform']['scale'][0]*1e3
            ext_bottom = imageInfo['deviceTransform']['pos'][1]*1e3 # - pz[1]
            ext_top = ext_bottom + imageInfo['region'][3]*imageInfo['deviceTransform']['scale'][1]*1e3
            ext_left, ext_right = self.reorder(ext_left, ext_right)
            ext_bottom, ext_top = self.reorder(ext_bottom, ext_top)
            # extents are manually adjusted - something about the rotation should be computed in them first...
            # but fix it manually... worry about details later.
            # yellow cross is aligned on the sample cell for this data now
            extents = [ext_bottom, ext_top, ext_left, ext_right]# 
            extents = [ext_left, ext_right, ext_bottom, ext_top]
            img = imageHandle.imagedata[0]
            if angle != 0.:  # note that angle arrives in radians - must convert to degrees for this one.
                img = scipy.ndimage.interpolation.rotate(img, angle*180./np.pi + 90., axes=(1, 0),
                    reshape=True, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
#            axp.imshow(img, extent=extents, aspect='auto', origin='lower')
            axp.imshow(img, aspect='equal', extent=extents, origin='lower', cmap=setMapColors('gray'))

        spotsize = 1e3*spotsize
        spotsizes = spotsize*np.linspace(1.0, 0.2, len(measure[measuretype]))
        pos = self.scale_and_rotate(pos, scale=1.0, angle=angle)
        xlim = [np.min(pos[:,0])-spotsize, np.max(pos[:,0])+spotsize]
        ylim = [np.min(pos[:,1])-spotsize, np.max(pos[:,1])+spotsize]
        # if vmaxin is not None:
        #     vmax = vmaxin  # fixed
        # else:
        vmax = np.max(np.max(measure[measuretype]))
        vmin = np.min(np.min(measure[measuretype]))
        # print('vmax: ', vmin, vmax)
        scaler = PH.NiceScale(0, vmax)
        vmax = scaler.niceMax

        # print('vmax res: ', vmax)
#        print('n measures/spots: ', len(measure[measuretype]), len(spotsizes), len(pos))

        for im in range(len(measure[measuretype])):  # there may be multiple measure[measuretype]s (repeated stimuli of different kinds) in a map
            # note circle size is radius, and is set by the laser spotsize (which is diameter)
            radw = np.ones(pos.shape[0])*spotsizes[im]
            radh = np.ones(pos.shape[0])*spotsizes[im]
            cmx = matplotlib.cm.ScalarMappable(norm=None, cmap=cm_sns)
            spotcolors = cmx.to_rgba(np.clip(measure[measuretype][im]/vmax, 0., 1.))
         #   colors = self.clip_colors(colors, [1., 1., 1., 1.])
            if self.nreps == 1:
                ec = collections.EllipseCollection(radw, radh, np.zeros_like(radw), offsets=pos, units='xy', transOffset=axp.transData,
                        facecolor=spotcolors, edgecolor='k', linewidth=0.2, alpha=1.0)
                axp.add_collection(ec)
            else:  # make arcs within the circle, each for a repeat
                ic = 0
                npos = pos.shape[0]
                dtheta = 360./self.nreps
                ri = 0
                rs = int(npos/self.nreps)
                for nr in range(self.nreps):
                    ec = wedges(pos[ri:(ri+rs),0], pos[ri:(ri+rs),1], radw[ri:(ri+rs)]/2.0,
                        theta1=nr*dtheta, theta2=(nr+1)*dtheta,
                        color=spotcolors[ri:ri+rs])
                    axp.add_collection(ec)
                    ri += rs
        if cellmarker:
            print('Cell marker')
            axp.plot([-cmrk, cmrk], [0., 0.], '-', color='r') # cell centered coorinates
            axp.plot([0., 0.], [-cmrk, cmrk], '-', color='r') # cell centered coorinates

        tickspace = scaler.tickSpacing
        try:
            ntick = 1 + int(vmax/tickspace)
        except:
            ntick = 3
        ticks = np.linspace(0, vmax, num=ntick, endpoint=True)
        if axcbar is not None:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
            c2 = matplotlib.colorbar.ColorbarBase(axcbar, cmap=cm_sns, ticks=ticks, norm=norm)
            c2.ax.tick_params(axis='y', direction='out')
       # axp.scatter(pos[:,0], pos[:,1], s=2, marker='.', color='k', zorder=4)
        axr = 250.

        axp.set_xlim(xlim)
        axp.set_ylim(ylim)
        if imageHandle is not None and imagefile is not None:
            axp.set_aspect('equal')
        axp.set_aspect('equal')
        axp.set_title(measuretype)
        if vmaxin is None:
            return vmax
        else:
            return vmaxin



    def fix_pd_artifact(self, data):
        """
        Hand tuned to minimize the transient and DC step from the PD signal - still leaves a bit of a fast
        transient (obviously the filter shape is not quite correct)
        The algorithm is simple:
            1. average all the PD data to reduce nose
            2. calculate a high-pass filtered version of the PD signal to emulate the capacitative cross-talk
            2b. Low pass filter the signal, same as the ephys data
            3. add (or subtract) a small amount of the PD signal to emulate the DC part of the cross-talk
                Done by fitting the transient to get a scale.
        
        Problems:
        1. averaging all traces in CC can produce artifacts across all traces when spikes occur
        2. averaging all traces in VC can reduce response amplitudes when there are lots of responses... 
        Attempts to use the PD signal or shutter signal (filtered or not) didn't work.
        
        """
        testplot = False

        avgd = data.copy()
        while avgd.ndim > 1:
            avgd = np.mean(avgd, axis=0)
        meanpddata = self.AR.Photodiode.mean(axis=0)  # get the average PD signal that was recorded
        shutter = self.AR.getBlueLaserShutter()
        # if meanpddata is not None:
        #     Util = EP.Utility.Utility()
        #     # filter the PD data - low pass to match data; high pass for apparent oupling
        #     crosstalk = Util.SignalFilter_HPFBessel(meanpddata, 2100., self.AR.Photodiode_sample_rate[0], NPole=1, bidir=False)
        #     crosstalk = Util.SignalFilter_LPFBessel(crosstalk, self.LPF, self.AR.Photodiode_sample_rate[0], NPole=1, bidir=False)
        #     crosstalk -= np.mean(meanpddata[0:int(0.010*self.AR.Photodiode_sample_rate[0])])
        #     crosstalk = np.hstack((np.zeros(1), crosstalk[:-1]))
        # else:
        #     return data, avgd
        protocol = self.protocol.name
        if protocol.find('_VC_10Hz') > 0:
            template_file = 'template_data_map_10Hz.pkl'
            ptype = '10Hz'
        elif protocol.find('_Single') > 0:
            template_file = 'template_data_map_10Hz.pkl'
            ptype = 'single'
        else:
            return(data, avgd)
        
        with open(template_file, 'rb') as fh:
            d = pickle.load(fh)
        ct_SR = np.mean(np.diff(d['t']))
        
        # or if from photodiode:
        # ct_SR = 1./self.AR.Photodiode_sample_rate[0]
        crosstalk = d['I'] - np.mean(d['I'][0:int(0.020/ct_SR)])  # remove baseline
        # crosstalk = self.filter_data(d['t'], crosstalk)
        avgdf  = avgd - np.mean(avgd[0:int(0.020/ct_SR)])
        #meanpddata = crosstalk
        # if self.shutter is not None:
        #     crossshutter = 0* 0.365e-21*Util.SignalFilter_HPFBessel(self.shutter['data'][0], 1900., self.AR.Photodiode_sample_rate[0], NPole=2, bidir=False)
        #     crosstalk += crossshutter

        maxi = np.argmin(np.fabs(self.tb - 0.6))
        ifitx = []
        art_times = np.array(self.stimtimes['start'])
        # artifact are:
        # 0.030 - 0.050: Camera
        # 0.050: Shutter
        # 0.055 : Probably shutter actual opening
        # 0.0390, 0.0410: Camera
        # 0.600 : shutter closing
        if ptype == '10Hz':
            other_arts = np.array([0.030, shutter['start'], 0.055, 0.390, 0.410, shutter['start']+shutter['duration']])
        else:
            other_arts = np.array([0.010, shutter['start'], 0.055, 0.305, 0.320, shutter['start']+shutter['duration']])
            
        art_times = np.append(art_times, other_arts)  # unknown (shutter is at 50 msec)
        art_durs = np.array(self.stimtimes['duration'])
        other_artdurs = 0.002*np.ones_like(other_arts)
        art_durs = np.append(art_durs, other_artdurs)  # shutter - do 2 msec
        
        for i in range(len(art_times)):
            strt_time_indx = int(art_times[i]/ct_SR)
            idur = int(art_durs[i]/ct_SR)
            send_time_indx = strt_time_indx + idur+int(0.001/ct_SR) # end pulse plus 1 msec
            # avglaser = np.mean(self.AR.LaserBlue_pCell, axis=0) # FILT.SignalFilter_LPFButter(np.mean(self.AR.LaserBlue_pCell, axis=0), 10000., self.AR.sample_rate[0], NPole=8)
            fitx = crosstalk[strt_time_indx:send_time_indx]# -np.mean(crosstalk)
            ifitx.extend([f[0]+strt_time_indx for f in np.argwhere((fitx > 0.5e-12) | (fitx < -0.5e-12))])
        wmax = np.max(np.fabs(crosstalk[ifitx]))
        weights = np.sqrt(np.fabs(crosstalk[ifitx])/wmax)
        scf, intcept = np.polyfit(crosstalk[ifitx], avgdf[ifitx], 1, w=weights)
        avglaserd = meanpddata # np.mean(self.AR.LaserBlue_pCell, axis=0)

#        print(scf, intcept)
        lbr = np.zeros_like(crosstalk)
        lbr[ifitx] = scf*crosstalk[ifitx]
        datar = np.zeros_like(data)
        for i in range(data.shape[0]):
            datar[i,:] = data[i,:] - lbr
        if testplot:
            """
            Note: This cannot be used if we are running in multiprocessing mode - will throw an error
            """
            from pyqtgraph.Qt import QtGui, QtCore
            import pyqtgraph as pg
            pg.setConfigOption('leftButtonPan', False)
            app = QtGui.QApplication([])
            win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
            win.resize(1000,600)
            win.setWindowTitle('pyqtgraph example: Plotting')
            # Enable antialiasing for prettier plots
            pg.setConfigOptions(antialias=True)
            p1 = win.addPlot(0, 0)
            p2 = win.addPlot(1, 0)
            p3 = win.addPlot(2, 0)
            p1r = win.addPlot(0,1)
            lx = np.linspace(np.min(crosstalk[ifitx]), np.max(crosstalk[ifitx]), 50)
            sp = pg.ScatterPlotItem(crosstalk[ifitx], avgdf[ifitx])  # plot regression over points
            ly = scf*lx + intcept
            p1r.addItem(sp)
            p1r.plot(lx, ly, pen=pg.mkPen('r', width=0.75))
            for i in range(10):
                p1.plot(self.tb, data[0,i,:]+2e-11*i, pen=pg.mkPen('r'))
                p1.plot(self.tb, datar[0,i,:]+2e-11*i, pen=pg.mkPen('g'))
            p1.plot(self.tb, lbr, pen=pg.mkPen('c'))
            p2.plot(self.tb, crosstalk, pen=pg.mkPen('m'))
            p2.plot(self.tb, lbr, pen=pg.mkPen('c'))
            p2.setXLink(p1)
            p3.setXLink(p1)
            p3.plot(self.tb, avgdf, pen=pg.mkPen('w', width=1.0))
            meandata = np.mean(datar[0], axis=0)
            meandata -= np.mean(meandata[0:int(0.020/ct_SR)])
            p3.plot(self.tb, meandata, pen=pg.mkPen('y'))
            p2.setXLink(p1)
            p3.setXLink(p1)
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtGui.QApplication.instance().exec_()
            exit()
#        print('dmax: ', np.max(np.max(datar)))
#        print('dmin: ', np.min(np.min(datar)))
        return datar, avgd

    # def fit_transient(self, p, x, y=None, pole1=1, pole2=1, idt=0, rate=None):
    #     scf = p[0]
    #     hpf_freq = p[1]*1e3  # to keep numbers scaled near 1, we use kHz and msec
    #     lpf_freq = p[2]*1e3
    #     tshift = p[3]*1e-3
    #     Util = EP.Utility.Utility()
    #     crosstalk = scf*x
    #     # filter the PD data - low pass to match data; high pass for apparent coupling
    #     crosstalk = Util.SignalFilter_HPFBessel(crosstalk, hpf_freq, rate, NPole=pole1, bidir=False)
    #     crosstalk = Util.SignalFilter_LPFBessel(crosstalk, lpf_freq, rate, NPole=pole2, bidir=False)
    #     if idt > 0:
    #         crosstalk = np.hstack((np.zeros(idt), crosstalk[:-idt]))
    #     # tx = np.arange(0., len(crosstalk)/rate, 1./rate)
    #     # # try time shifting
    #     # newx = tx + tshift
    #     # crosstalk = np.interp(newx, tx, crosstalk, left=crosstalk[0], right=crosstalk[-1])
    #
    #     if y is None:
    #         return crosstalk
    #     else:
    #         return np.sum(np.fabs(y-crosstalk))
    #

#     def fix_pd_artifact_2(self, data):
#         """
#         Hand tuned to minimize the transient and DC step from the PD signal - still leaves a bit of a fast
#         transient (obviously the filter shape is not quite correct)
#         The algorithm is simple:
#             1. average all the PD data to reduce nose
#             2. calculate a high-pass filtered version of the PD signal to emulate the capacitative cross-talk
#             2b. Low pass filter the signal, same as the ephys data
#             3. add (or subtract) a small amount of the PD signal to emulate the DC part of the cross-talk
#                 Done by fitting the transient to get a scale.
#
#         Problems:
#         1. averaging all traces in CC can produce artifacts across all traces when spikes occur
#         2. averaging all traces in VC can reduce response amplitudes when there are lots of responses...
#         Attempts to use the PD signal or shutter signal (filtered or not) didn't work.
#
#         """
#         testplot = True
#
#         avgd = data.copy()
#         while avgd.ndim > 1:
#             avgd = np.mean(avgd, axis=0)
#         meanpddata = self.AR.Photodiode.mean(axis=0)  # get the average PD signal that was recorded
#         if meanpddata is None:
#             return data, avgd
#         rate = self.AR.Photodiode_sample_rate[0]
#         meanpddata = meanpddata-np.mean(meanpddata[0:int(0.01*rate)])
#         strt_time_indx = int(np.array(self.stimtimes['start'][0])*rate)
#         send_time_indx = strt_time_indx + int(0.002*rate) # 5 msec later
#
#         fitx = meanpddata[strt_time_indx:send_time_indx]
#         ifitx = [f[0]+strt_time_indx for f in np.argwhere((fitx > 1e-4) | (fitx < -1e-4))]
#         pole1 = 1
#         pole2 = 4
#         idt = 1
#         scf = 1
#         tshift = 0.
#         pars = [scf, 0.3, 2.5, tshift]
#         relscale = 1./np.max(meanpddata[ifitx])/np.max(avgd[ifitx])
#         print(f'relscale: {relscale:.3e}')
#         bounds = [(-1e4, 1e4), (0.1, 2.), (2., 1e-3*rate*0.45), (-1, 1)]
#         fpar = scipy.optimize.minimize(self.fit_transient, pars, method='TNC', bounds=bounds,
#                 args=(meanpddata[ifitx], relscale*avgd[ifitx], pole1, pole2, idt, rate),
#                 options={'eps': 1e-6, 'maxiter': 10000})
#         print(fpar)
#         crosstalk = (1./relscale)*self.fit_transient(fpar.x, meanpddata, y=None, pole1=pole1, pole2=pole2, idt=idt, rate=rate)
#
#         datar = np.zeros_like(data)
#         for i in range(data.shape[0]):
#             datar[i,:] = data[i,:] - crosstalk
#         if testplot:
#             from pyqtgraph.Qt import QtGui, QtCore
#             import pyqtgraph as pg
#             app = QtGui.QApplication([])
#             win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
#             win.resize(1000,600)
#             win.setWindowTitle('pyqtgraph example: Plotting')
#             # Enable antialiasing for prettier plots
#             pg.setConfigOptions(antialias=True)
#             p1 = win.addPlot(0, 0)
#             p2 = win.addPlot(1, 0)
#             p3 = win.addPlot(2, 0)
#             p1r = win.addPlot(0,1)
#             p2r = win.addPlot(1,1)
#             # print(ifitx)
#             # print(meanpddata[ifitx])
#             # print(avgd[ifitx])
#             p2r.plot(self.tb[ifitx], meanpddata[ifitx]*fpar.x[0], pg.mkPen('b'))
#             p2r.plot(self.tb[ifitx], avgd[ifitx], pg.mkPen('w'))
#             sp = pg.ScatterPlotItem(crosstalk[ifitx], avgd[ifitx])  # plot regression over points
#             p1r.addItem(sp)
#             # lx = np.linspace(np.min(crosstalk[ifitx]), np.max(crosstalk[ifitx]), 50)
#             # ly = scf*lx + intcept
#             # p1r.plot(lx, ly, pen=pg.mkPen('r', width=0.75))
#             for i in range(10):
#                 p1.plot(self.tb, data[0,i,:]+2e-11*i, pen=pg.mkPen('r'))
#                 p1.plot(self.tb, datar[0,i,:]+2e-11*i, pen=pg.mkPen('g'))
#             p1.plot(self.tb, crosstalk, pen=pg.mkPen('c'))
#             p2.plot(self.tb, crosstalk, pen=pg.mkPen('m'))
#             p2.setXLink(p1)
#             p3.setXLink(p1)
#             p3.plot(self.tb, avgd, pen=pg.mkPen('r', width=1.0))
#             p3.plot(self.tb, np.mean(datar[0], axis=0), pen=pg.mkPen('y'))
#             p2.setXLink(p1)
#             p3.setXLink(p1)
#             if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#                 QtGui.QApplication.instance().exec_()
#             exit()
# #        print('dmax: ', np.max(np.max(datar)))
# #        print('dmin: ', np.min(np.min(datar)))
#         return datar, avgd

    def analyze_one_map(self, dataset, plotevents=False):
       # print('ANALYZE ONE MAP')
        self.data, self.tb, pars, info=self.readProtocol(dataset, sparsity=None)
        if self.data is None:   # check that we were able to retrieve data
            self.P = None
            return None
        self.last_dataset = dataset
        if self.fix_photodiode_artifact:
            self.data_clean, self.avgdata = self.fix_pd_artifact(self.data)
        else:
            self.data_clean = self.data
        stimtimes = []
#        print(self.twin_resp)
        data_nostim = []
#        indxs = []
        lastd = 0
        for i, tr in enumerate(self.twin_resp):  # get window for response
            notokd = np.where((self.tb >= tr[0]) & (self.tb < tr[1]))[0]
            data_nostim.append(list(range(lastd, notokd[0])))
#            indxs.append([lastd, notokd[0]])
            lastd = notokd[-1]
        # fill end space... 
        endindx = np.where(self.tb >= self.AR.tstart)[0][0]
        data_nostim.append(list(range(lastd, endindx)))
#        indxs.append([lastd, endindx])
        data_nostim = list(np.hstack(np.array(data_nostim)))

        results = self.analyze_protocol(self.data_clean, self.tb, info, eventhist=True, dataset=dataset, data_nostim=data_nostim)
        self.last_results = results
        return results

    def display_one_map(self, dataset, imagefile=None, rotation=0.0, measuretype=None, 
            plotevents=True):
        if dataset != self.last_dataset:
            results = self.analyze_one_map(dataset)
        else:
            results = self.last_results
        if results is None:
            return
        if '_IC_' in str(dataset.name) or 'CC' in str(dataset.name):
            scf = 1e3
            label = 'mV'  # mV
        elif '_VC' in str(dataset.name):
            scf = 1e12 # pA, vc
            label = 'pA'
        else:
            scf = 1.0
            label = 'AU'
        # f, ax = mpl.subplots(2,2)
#         ax = ax.ravel()
#         for k, s in enumerate(['I_max', 'ZScore', 'Qr', 'Qb']):
#             for i in results[s]:
#                 ax[k].scatter(np.arange(len(i)), i, s=5)
#                 ax[k].set_title(s)
#             if s in ['Qb', 'Qr']:
#                 ax[k].set_ylim(-0.20, np.max(i))
#         mpl.show()
        
        # build a figure
        l_c1 = 0.1  # column 1 position
        l_c2 = 0.50 # column 2 position
        trw = 0.32  # trace x width
        trh = 0.10  # trace height
        imgw = 0.25 # image box width
        imgh = 0.25
        trs = imgh - trh  # 2nd trace position (offset from top of image box)
        y = 0.08 + np.arange(0., 0.7, imgw+0.05)  # y positions 
        self.mapfromid = {0: ['A', 'B', 'C'], 1: ['D', 'E', 'F'], 2: ['G', 'H', 'I']}        
        self.plotspecs = OrderedDict([('A', {'pos': [0.07, 0.3, 0.62, 0.3]}),
                                 ('A1',{'pos': [0.37, 0.012, 0.62, 0.3]}), # scale bar
                                 ('B', {'pos': [0.07, 0.3, 0.475, 0.125]}),
                                 ('C1', {'pos': [0.07, 0.3, 0.31, 0.125]}),
                                 ('C2', {'pos': [0.07, 0.3, 0.16, 0.125]}),
                                 ('D', {'pos': [0.07, 0.3, 0.05, 0.075]}),
                                 ('E', {'pos': [0.47, 0.45, 0.05, 0.85]}),
                            #     ('F', {'pos': [0.47, 0.45, 0.10, 0.30]}),
                                 ])  # a1 is cal bar

        self.P = PH.Plotter(self.plotspecs, label=False, figsize=(10., 8.))

        self.plot_events(self.P.axdict['B'], results)  # PSTH
        if imagefile is not None:
            self.MT.get_image(imagefile)
            self.MT.load_images()
            # print (self.MT.imagemetadata)
            # self.MT.show_images()
            # exit(1)
       # self.plot_average_traces(self.P.axdict['C'], self.tb, self.data_clean)
        
        ident = 0
        if ident == 0:
            cbar = self.P.axdict['A1']
        else:
            cbar = None
        idm = self.mapfromid[ident]

        if self.AR.spotsize == None:
            self.AR.spotsize=50e-6
        self.newvmax = np.max(results[measuretype])
        if self.overlay_scale > 0.:
            self.newvmax = self.overlay_scale
        self.newvmax = self.plot_map(self.P.axdict['A'], cbar, results['positions'], measure=results, measuretype=measuretype, 
            vmaxin=self.newvmax, imageHandle=self.MT, imagefile=imagefile, angle=rotation, spotsize=self.AR.spotsize)
        self.plot_stacked_traces(self.tb, self.data_clean, dataset, events=results['events'], ax=self.P.axdict['E'])  # stacked on right
        
        self.plot_avgevent_traces(evtype='avgevoked', mdata=self.data_clean, ax=self.P.axdict['C1'], 
                events=results['events'], scale=scf, label=label)
        self.plot_avgevent_traces(evtype='avgspont', mdata=self.data_clean, ax=self.P.axdict['C2'], 
                events=results['events'], scale=scf, label=label)
        
        if self.photodiode:
            self.plot_photodiode(self.P.axdict['D'], self.AR.Photodiode_time_base[0], self.AR.Photodiode)
       # mpl.show()
        return True # indicated that we indeed plotted traces.
        
 
if __name__ == '__main__':
    # these must be done here to avoid conflict when we import the class, versus
    # calling directly for testing etc.
    matplotlib.use('Agg')
    rcParams = matplotlib.rcParams
    rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['text.latex.unicode'] = True
    #rcParams['font.family'] = 'sans-serif'
    rcParams['font.weight'] = 'regular'                  # you can omit this, it's the default
    #rcParams['font.sans-serif'] = ['Arial']

    datadir = '/Volumes/PBM_004/data/MRK/Pyramidal'
    parser = argparse.ArgumentParser(description='mini synaptic event analysis')
    parser.add_argument('datadict', type=str,
                        help='data dictionary')
    parser.add_argument('-s', '--scale', type=float, default=0., dest='scale',
                        help='set maximum scale for overlay plot (default=0 -> auto)')
    parser.add_argument('-i', '--IV', action='store_true', dest='do_iv',
                        help='just do iv')
    parser.add_argument('-o', '--one', type=str, default='', dest='do_one',
                        help='just do one')
    parser.add_argument('-m', '--map', type=str, default='', dest='do_map',
                        help='just do one map')
    parser.add_argument('-c', '--check', action='store_true',
                        help='Check for files; no analysis')
    # parser.add_argument('-m', '--mode', type=str, default='aj', dest='mode',
    #                     choices=['aj', 'cb'],
    #                     help='just do one')
    parser.add_argument('-v', '--view', action='store_false',
                        help='Turn off pdf for single run')

    args = parser.parse_args()

    filename = os.path.join(datadir, args.datadict)
    if not os.path.isfile(filename):
        print('File not found: %s' % filename)
        exit(1)

    DP = EP.DataPlan.DataPlan(os.path.join(datadir, args.datadict))  # create a dataplan
    plan = DP.datasets
    print('plan dict: ', plan.keys())
        #print('plan: ', plan)
    if args.do_one != '':
        cellid = int(args.do_one)
    else:
        raise ValueError('no cell id found for %s' % args.do_one)
    cell = DP.excel_as_df[DP.excel_as_df['CellID'] == cellid].index[0]

    print('cellid: ', cellid)
    print('cell: ', cell)

    print('cell: ', plan[cell]['Cell'])
    datapath = os.path.join(datadir, str(plan[cell]['Date']).strip(), str(plan[cell]['Slice']).strip(), str(plan[cell]['Cell']).strip())
   # print( args)
    
    if args.do_iv:
        
        EPIV = EP.IVSummary.IVSummary(os.path.join(datapath, str(plan[cell]['IV']).strip()))
        EPIV.compute_iv()
        print('cell: ', cell, plan[cell]['Cell'])
        DP.post_result('CellID', cellid, 'RMP', EPIV.RM.analysis_summary['RMP'])
        DP.post_result('CellID', cellid, 'Rin', EPIV.RM.analysis_summary['Rin'])
        DP.post_result('CellID', cellid, 'taum', EPIV.RM.analysis_summary['taum'])
        now = datetime.datetime.now()
        DP.post_result('CellID', cellid, 'IV Date', str(now.strftime("%Y-%m-%d %H:%M")))
        DP.update_xlsx(os.path.join(datadir, args.datadict), 'Dataplan')
        exit(1)
    
    if args.do_map:
        getimage = True
        plotevents = True
        dhImage = None
        rotation = 0
        AM = AnalyzeMap()
        AM.sign = plan[cell]['Sign']
        AM.overlay_scale = args.scale
        AM.display_one_map(os.path.join(datapath, str(plan[cell]['Map']).strip()),
            imagefile=os.path.join(datapath, plan[cell]['Image'])+'.tif', rotation=rotation, measuretype='ZScore')
        mpl.show()