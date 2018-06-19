from __future__ import print_function

"""

This code was build to process the TTX experimetns with maps directly from the data, but
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

#from pyqtgraph.metaarray import MetaArray

import functions
from minis import minis_methods
import matplotlib.pyplot as mpl
import matplotlib.colors
import matplotlib
import matplotlib.collections as collections
from  matplotlib import colors as mcolors
import matplotlib.cm
import colormaps.parula
import pylibrary.PlotHelpers as PH
#import seaborn
#cm_sns = mpl.cm.get_cmap('terrain')  # terrain is not bad
#cm_sns = mpl.cm.get_cmap('parula')  # terrain is not bad
#cm_sns = mpl.cm.get_cmap('jet')  # jet is terrible
color_sequence = ['k', 'r', 'b']
colormap = 'snshelix'

basedir = "/Users/pbmanis/Desktop/Python/mapAnalysisTools"

# this was copied from the cell3 image .index file 
imageInfo = OrderedDict([(u'region', [0, 0, 1376, 1024]),
                    (u'transform',
                        {'scale': (1.6349999896192458e-06, -1.6349999896192458e-06, 1.0),
                        'angle': -0.0,
                        'pos': (0.054586220532655716, 0.002669319976121187, -0.0013764000032097101),
                        'axis': (0, 0, 1)}),
                    (u'binning', [1, 1]),
                    (u'__timestamp__', 1485552541.933),
                    (u'__object_type__', 'ImageFile'),
                    (u'fps', 1.683427746690181),
                    (u'time', 1485552541.5737224),
                    (u'objective', '4x 0.1na ACHROPLAN'),
                    (u'deviceTransform',
                        {'scale': (1.6349999896192458e-06, -1.6349999896192458e-06, 1.0),
                        'angle': -0.0,
                        'pos': (0.054586220532655716, 0.002669319976121187, -0.0013764000032097101),
                        'axis': (0, 0, 1)}),
                    (u'pixelSize', [1.6354024410247803e-06, 1.6349367797374725e-06]),
                    (u'exposure', 0.5),
                    (u'id', 71617),
                    (u'triggerMode', 'Normal'),
                    (u'frameTransform',
                        {'scale': (1.0, 1.0, 1.0),
                        'angle': 0.0,
                        'pos': (0.0, 0.0, 0.0),
                        'axis': (0, 0, 1)})
                    ])

re_degree = re.compile('\s*(\d{1,3}d)\s*')
re_duration = re.compile('(\d{1,3}ms)')

def setMapColors(colormap, reverse=False):
    if colormap == 'terrain':
        cm_sns = mpl.cm.get_cmap('terrain_r')  # terrain is not bad    #
    # elif colormap == 'cubehelix':
    #     cm_sns = seaborn.cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0,
    #         hue=0.8, light=0.85, dark=0.15, reverse=reverse, as_cmap=False)
    # elif colormap == 'snshelix':
    #     cm_sns = seaborn.cubehelix_palette(n_colors=64, start=3, rot=0.5, gamma=1.0, dark=0, light=1.0, reverse=reverse,
    #      as_cmap=True)
    elif colormap == 'a':
        from colormaps import option_a
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_a', option_a.cm_data)
    elif colormap == 'b':
        import colormaps.option_b
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_b', colormaps.option_b.cm_data)
    elif colormap == 'c':
        import colormaps.option_c
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_c', colormaps.option_c.cm_data)
    elif colormap == 'd':
        import colormaps.option_a
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_d', colormaps.option_d.cm_data)
    elif colormap == 'parula':
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('parula', colormaps.parula.cm_data)
    else:
        print('Unrecongnized color map {0:s}; setting to "parula"'.format(colormap))
        import colormaps.parula
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('parula', colormaps.parula.cm_data)
    return cm_sns    


class AnalyzeMap(object):
    
    def __init__(self):
        self.AR = EP.acq4read.Acq4Read()
        self.SP = EP.SpikeAnalysis.SpikeAnalysis()
        self.RM = EP.RmTauAnalysis.RmTauAnalysis()
        self.MT = MONT.montager.Montager()
        self.twin_base = [0., 0.295]
        self.twin_resp = [0.301, 0.4]
        
    def readProtocol(self, protocolFilename, records=None, sparsity=None, getPhotodiode=False):
        starttime = timeit.default_timer()
        self.AR.setProtocol(protocolFilename)
    #    a.setProtocol('/Volumes/Pegasus/ManisLab_Data3/Kasten, Michael/2017.11.20_000/slice_000/cell_000/CCIV_4nA_max_000')
        self.AR.getData()
        self.AR.getScannerPositions()
        #mpl.plot(AR.scannerpositions[:,0], AR.scannerpositions[:,1], 'ro')
        #mpl.show()
        data = np.reshape(self.AR.traces, (self.AR.repetitions, self.AR.traces.shape[0], self.AR.traces.shape[1]))
        endtime = timeit.default_timer()
        print("Time to read data: %f s" % (endtime-starttime))
        return data, self.AR.time_base, self.AR.sequenceparams, self.AR.scannerinfo

    def show_images(self, show=True):
        r, c = PH.getLayoutDimensions(len(self.images), pref='height')
        f, ax = mpl.subplots(r, c)
        f.suptitle(self.celldir, fontsize=9)
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

    def calculate_charge(self, tb, data):
        """
        One-dimensional data...
        """

        # get indices for the integration windows
        tbindx = np.where((tb >= self.twin_base[0]) & (tb < self.twin_base[1]))
        trindx = np.where((tb >= self.twin_resp[0]) & (tb < self.twin_resp[1]))
        Qr = 1e6*np.sum(data[trindx])/(self.twin_resp[1]-self.twin_resp[0]) # response
        Qb = 1e6*np.sum(data[tbindx])/(self.twin_base[1]-self.twin_base[0])  # baseline
        return Qr, Qb


    def ZScore(self, tb, data, tbase=[0., 0.1], ):
        # abs(post.mean() - pre.mean()) / pre.std()
        # get indices for the integration windows
        tbindx = np.where((tb >= self.twin_base[0]) & (tb < self.twin_base[1]))
        trindx = np.where((tb >= self.twin_resp[0]) & (tb < self.twin_resp[1]))
        mpost = np.mean(data[trindx]) # response
        mpre = np.mean(data[tbindx])  # baseline
        return(np.fabs((mpost-mpre)/np.std(data[tbindx])))

    def Imax(self, tb, data, sign=1):

        tbindx = np.where((tb >= self.twin_base[0]) & (tb < self.twin_base[1]))
        trindx = np.where((tb >= self.twin_resp[0]) & (tb < self.twin_resp[1]))
        mpost = np.max(sign*data[trindx]) # response goes negative... 
        return(mpost)

    def analyze_protocol(self, data, tb, info, taus, LPF=5000., sign=1, threshold=2.0, eventstartthr=None, eventhist=True, testplots=False):
        use_AJ = True
        aj = minis_methods.AndradeJonas()
        cb = minis_methods.ClementsBekkers()
        cutoff = 5000. # LPF at Hz
        filtfunc = scipy.signal.filtfilt
        rate = np.mean(np.diff(tb))  # t is in seconds, so freq is in Hz
        samplefreq = 1.0/rate
        nyquistfreq = samplefreq/1.95
        wn = cutoff/nyquistfreq 
        b, a = scipy.signal.bessel(2, wn)
        for r in range(data.shape[0]):
            for t in range(data.shape[1]):
                data[r,t,:] = filtfunc(b, a, data[r, t, :] - np.mean(data[r, t, 0:250]))
        mdata = np.mean(data, axis=0)  # mean across ALL reps
        rate = rate*1e3  # convert rate to msec
        # make visual maps
        Qr = np.zeros(data.shape[1])
        Qb = np.zeros(data.shape[1])
        Zscore = np.zeros(data.shape[1])
        I_max = np.zeros(data.shape[1])
        pos = np.zeros((data.shape[1], 2))
        for t in range(data.shape[1]):  # compute for each target
            Qr[t], Qb[t] = self.calculate_charge(tb, mdata[t,:])
            Zscore[t] = self.ZScore(tb, mdata[t,:])
            I_max[t] = self.Imax(tb, data[0,t,:], sign=-1)*1e12  # just the FIRST pass
            try:
                pos[t,:] = [info[(0,t)]['pos'][0], info[(0,t)]['pos'][1]]
            except:
                pass
    
        events = {}
        eventlist = []  # event histogram across ALL events/trials
        nevents = 0
        if eventhist:
            v = [-1.0, 0., taus[0], taus[1]]
            x = np.linspace(0., taus[1]*5, int(50./rate))
            cbtemplate = functions.pspFunc(v, x, risePower=2.0).view(np.ndarray)
            tmaxev = 600. # msec
            jmax = int(tmaxev/rate)
            for j in range(data.shape[0]):  # all trials
                result = []
                crit = []
                scale = []
                tpks = []
                smpks = []
                smpksindex = []
                avgev = []
                avgtb = []
                avgnpts = []
                for i in range(data.shape[1]):  # all targets
                    if use_AJ:
                        idata = 1e12*data.view(np.ndarray)[j, i, :]
                        aj.setup(tau1=taus[0], tau2=taus[1], dt=rate, delay=0.0, template_tmax=rate*(jmax-1),
                                sign=sign, eventstartthr=eventstartthr)
                        print('AJ Sign: ', aj.sign)
                        meandata = np.mean(idata[:jmax])
                        aj.deconvolve(idata[:jmax]-meandata, 
                                thresh=threshold, llambda=10., order=7)
                        # if len(aj.onsets) == 0:  # no events, so skip
                        #     continue
                        nevents += len(aj.onsets)
                        result.append(aj.onsets)
                        eventlist.append(tb[aj.onsets])
                        tpks.append(aj.peaks)
                        smpks.append(aj.smoothed_peaks)
                        smpksindex.append(aj.smpkindex)
                        if aj.averaged:
                            avgev.append(aj.avgevent)
                            avgtb.append(aj.avgeventtb)
                            avgnpts.append(aj.avgnpts)
                        else:
                            avgev.append([])
                            avgtb.append([])
                            avgnpts.append(0)
                    
                        if testplots:
                            aj.plots(title='%d' % i, events=None)
                      #  print ('eventlist: ', eventlist)
                    else:
                        res = cb.cbTemplateMatch(1e12*data.view(np.ndarray)[j, i, :jmax], 
                                template=cbtemplate, threshold=2.5, sign=1)
                        result.append(res)
                        crit.append(cb.Crit)
                        scale.append(cb.Scale)
                        for r in res: # aj.onsets:
                        #    print r
                            if r[0] > 0.5:
                                eventlist.append(r[0]*rate)
                        if testplots:
    #                        print ('testplots')
                            mpl.plot(tb, 1e12*data.view(np.ndarray)[j, i, :], 'k-', linewidth=0.5)
                            for k in range(len(res)):
                                mpl.plot(tb[res[k][0]], 1e12*data.view(np.ndarray)[j, i, res[k][0]], 'ro')
                                mpl.plot(tb[res[k][0]]+np.arange(len(cb.template))*rate/1000.,
                                 cb.sign*cb.template*np.max(res['scale'][k]), 'b-')
                            mpl.show()
                events[j] = {'criteria': crit, 'result': result, 'peaktimes': tpks, 'smpks': smpks, 'smpksindex': smpksindex,
                    'avgevent': avgev, 'avgtb': avgtb, 'avgnpts': avgnpts}
        print('analyze protocol returns, nevents = %d' % nevents)
        return{'Qr': Qr, 'Qb': Qb, 'ZScore': Zscore, 'I_max': I_max, 'positions': pos, 'aj': aj, 'events': events, 'eventtimes': eventlist}

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

    def analyze_file(self, filename, dhImage=None, plotFlag=False, 
            taus=[0.33, 4.0], LPF=5000., sign=1, threshold=2.0):
        protodata = {}
        nmax = 1000
        data, tb, pars, info = self.readProtocol(filename, sparsity=None)
        print('read from raw file')
        results = self.analyze_protocol(data, tb, info, taus=taus, LPF=LPF, sign=sign, threshold=threshold, eventhist=True)
        plot_all_traces(tb, data, title=filename)
    
        return(results)

    def plot_events(self, axh, axt, results, colorid=0):
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

        self.plot_traces(axt, self.tb, np.mean(self.data, axis=0), color=color_sequence[colorid])
        if plotevents and len(results['eventtimes']) > 0:
            y=[]
            for x in range(len(results['eventtimes'])):
                for xn in results['eventtimes'][x]:
                    y.append(xn)
            axh.hist(y, 300, range=[0., 0.6], normed=1, facecolor='k')
            axh.set_xlim([0., 0.6])
            #axh.set_ylim([0., 50.])
                   # mpl.show()

    def plot_all_traces(self, tb, mdata, title):
        f, ax = mpl.subplots(1,1)
        stepi = 20.
        for i in range(mdata.shape[1]):
            ax.plot(tb, mdata[0, i,:]*1e12 + stepi*i, linewidth=0.2)
        mpl.suptitle(title, fontsize=9)
        ax.set_xlim(0, 0.599)

    def plot_traces(self, ax, tb, mdata, color='k'):
        ax.plot(tb, np.mean(mdata, axis=0)*1e12, color)
        ax.set_xlim([0., 0.6])
     #   ax.set_ylim([-100., 20.])

    def plot_map(self, axp, axcbar, pos, measure, vmaxin=None, 
            imageHandle=None, angle=0, spotsize=20e-6, cellmarker=False):

        sf = 1.0 # could be 1e-6 ? data in Meters? scale to mm. 
        cmrk = 50e-6*sf # size, microns
        for n in range(pos.shape[0]):
            if pos[n,0] == 0. and pos[n,1] == 0:
                break
        pos = pos[:n,:] *1e3 # clip unused positions   
        pz = [np.mean(pos[:,0]), np.mean(pos[:,1])]
  #      pos[:,0] = pos[:,0]-pz[0]
   #     pos[:,1] = pos[:,1]-pz[1]
            
        if imageHandle is not None:
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
            axp.imshow(img, aspect='equal', extent=extents, origin='lower')
        # mpl.show()
        # exit(1)
        vmin = 0.
        # adjust vmax's to use rational values. 
        if vmaxin is not None:
            vmax = vmaxin
        else:
            vmax = np.max(measure)
        scaler = PH.NiceScale(0, vmax)
        vmax = scaler.niceMax
        print('vmax: ', vmax)
        spotsize = 1e3*spotsize
        pos = self.scale_and_rotate(pos, scale=1.0, angle=angle)
        print(pos.shape)
        xlim = [np.min(pos[:,0])-spotsize, np.max(pos[:,0])+spotsize]
        ylim = [np.min(pos[:,1])-spotsize, np.max(pos[:,1])+spotsize]
        # note circle size is radius, and is set by the laser spotsize (which is diameter)
        ec = []
        spotsize = spotsize
        radw = np.ones(pos.shape[0])*spotsize
        radh = np.ones(pos.shape[0])*spotsize
        colors = []
        maxv = np.max(measure)
        cmx = matplotlib.cm.ScalarMappable(norm=None, cmap=cm_sns)
        colors = cmx.to_rgba(measure/vmax)
            #colors.append(mcolors.to_rgba(cm_sns(measure/vmax), alpha=0.6))
        f, axe = mpl.subplots(1,1)
        print('pos: ', pos)
        print('spotsize: ', spotsize)
        ec = collections.EllipseCollection(radw, radh, np.zeros_like(radw), offsets=pos, units='xy', transOffset=axp.transData,
                    facecolor=colors, edgecolor='k', alpha=0.75)
        # print(ec)
        axp.add_collection(ec)
        #pm = PH.circles(pos[:,0], pos[:,1], spotsize/2., c=np.array(measure),
         #                   vmax=vmax, vmin=vmin,
         #                cmap=cm_sns, ax=axp, edgecolors='k', linewidths=0.5, alpha=0.9)
        if cellmarker:
            axp.plot([-cmrk, cmrk], [0., 0.], '-', color='r') # cell centered coorinates
            axp.plot([0., 0.], [-cmrk, cmrk], '-', color='r') # cell centered coorinates

        tickspace = scaler.tickSpacing
        ntick = 1 + int(vmax/tickspace)
        ticks = np.linspace(0, vmax, num=ntick, endpoint=True)
        if axcbar is not None:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
            c2 = matplotlib.colorbar.ColorbarBase(axcbar, cmap=cm_sns, ticks=ticks, norm=norm)
            c2.ax.tick_params(axis='y', direction='out')
        axp.scatter(pos[:,0], pos[:,1], s=2, marker='.', color='k', zorder=4)
        axr = 250.
        print (xlim, ylim)
        # axp.set_xlim(xlim)
        # axp.set_ylim(ylim)
        axp.set_aspect('equal')

       # axp.set_facecolor([0.6, 0.6, 0.9, 0.3])

        if vmaxin is None:
            return vmax
        else:
            return vmaxin

    def display_one_map(self, dataset, imagefile=None, rotation = 0.0, measuretype = 'ZScore'):
        self.data, self.tb, pars, info = self.readProtocol(dataset, sparsity=None)
        print('read from raw file')
        # if writepickle:  # save the data off... moving sequences to nparrays seemed to solve a pickle problem...
        #     pdata[protocolfilename] = {'data': data, 'tb': tb, 'info': info ,
        #         'sequence1': np.array(pars['sequence1']['d']),
        #         'sequence2': np.array(pars['sequence2']['index']),
        #     }
        #     continue
        results = self.analyze_protocol(self.data, self.tb, info, taus=[0.4, 5], sign=1, eventhist=plotevents)

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
        self.plotspecs = OrderedDict([('A', {'pos': [l_c1, imgw, y[2], imgh]}), 
                                 ('B', {'pos': [l_c2, trw, y[2]+trs, trh]}),
                                 ('C', {'pos': [l_c2, trw, y[2], trh]}),
                                 ('D', {'pos': [l_c1, imgw, y[1], imgh]}), 
                                 ('E', {'pos': [l_c2, trw, y[1]+trs, trh]}), 
                                 ('F', {'pos': [l_c2, trw, y[1], trh]}),
                                 ('G', {'pos': [l_c1, imgw, y[0], imgh]}), 
                                 ('H', {'pos': [l_c2, trw, y[0]+trs, trh]}), 
                                 ('I', {'pos': [l_c2, trw, y[0], trh]}),
                                 ('A1', {'pos': [l_c1+imgw+0.01, 0.012, y[2], imgh]})])

        self.P = PH.Plotter(self.plotspecs, label=False, figsize=(8., 6.))

        self.plot_events(self.P.axdict['B'], self.P.axdict['C'], results)
        
        if imagefile is not None:
            self.MT.get_image(imagefile)
            self.MT.load_images()
            # print (self.MT.imagemetadata)
            # self.MT.show_images()
            # exit(1)
            
        ident = 0
        if ident == 0:
            cbar = self.P.axdict['A1']
        else:
            cbar = None
        idm = self.mapfromid[ident]

        if self.AR.spotsize == None:
            self.AR.spotsize=50e-6
        self.newvmax = np.max(results[measuretype])
        print('newmax: ', self.newvmax)
        
        self.newvmax = self.plot_map(self.P.axdict['A'], cbar, results['positions'], results[measuretype], 
            vmaxin=self.newvmax, imageHandle=self.MT, angle=rotation, spotsize=self.AR.spotsize)
        self.plot_all_traces(self.tb, self.data, dataset)
    # if writepickle:
    #     save_pickled(cell+'.p', pdata)

        # if not writepickle:
        mpl.show()
    
    
    # def run_analysis(self, dataplan, dhImage=None, rotation=0.):
    #
    #     # writepickle = False
    #     # if 'writep' in sys.argv[1:]:
    #     #     writepickle = True
    #     plotFlag = True
    #     protodata = {}
    #     nmax = 1000
    #
    #     imgw = 0.25 # image box width
    #     imgh = 0.25
    #     l_c1 = 0.1  # column 1 position
    #     l_c2 = 0.50 # column 2 position
    #     trw = 0.32  # trace x width
    #     trh = 0.10  # trace height
    #     trs = imgh - trh  # 2nd trace position (offset from top of image box)
    #     y = 0.08 + np.arange(0., 0.7, imgw+0.05)  # y positions
    #
    #     plotspecs = OrderedDict([('A', {'pos': [l_c1, imgw, y[2], imgh]}),
    #                              ('B', {'pos': [l_c2, trw, y[2]+trs, trh]}),
    #                              ('C', {'pos': [l_c2, trw, y[2], trh]}),
    #                              ('D', {'pos': [l_c1, imgw, y[1], imgh]}),
    #                              ('E', {'pos': [l_c2, trw, y[1]+trs, trh]}),
    #                              ('F', {'pos': [l_c2, trw, y[1], trh]}),
    #                              ('G', {'pos': [l_c1, imgw, y[0], imgh]}),
    #                              ('H', {'pos': [l_c2, trw, y[0]+trs, trh]}),
    #                              ('I', {'pos': [l_c2, trw, y[0], trh]}),
    #                              ('A1', {'pos': [l_c1+imgw+0.01, 0.012, y[2], imgh]})])
    #
    #     P = PH.Plotter(plotspecs, label=False, figsize=(8., 6.))
    #
    #     mapfromid = {0: ['A', 'B', 'C'], 1: ['D', 'E', 'F'], 2: ['G', 'H', 'I']}
    #     # set up low-pass filter
    #     self.newvmax = None
    #
    #     for ic, cell in enumerate(dataplan.datasets.keys()):
    #         if cell not in [celln]:
    #             continue
    #         imagefile = None
    #         if cell == 'cell3_low_power_image':
    #             imagefile = os.path.join(datadir, dataplan.datasets[cell])
    #             dhImage = DM.getDirHandle(protocolFilename, create=False)
    #         if writepickle:
    #             print('Will write pickled file')
    #             pdata = OrderedDict()
    #         dataset = dataplan.datasets[cell]
    #         for ident, d in enumerate(dataset['runs']):
    #             if d == '':
    #                 continue
    #             print ('Cell: ', cell)
    #             protocolfilename = os.path.join(dataplan.datadir, d)
    #              # check to see if pickled file exists first - it is faster to read
    #             if os.path.isfile(cell + '.p') and not writepickle:
    #                 d = read_pickled(cell)
    #                 d = d['data']
    #                 dx = d[protocolfilename]
    #                 data = dx['data']
    #                 tb = dx['tb']
    #                 info = dx['info']
    #                 pars = {'sequence1': dx['sequence1'], 'sequence2': dx['sequence2']}
    #                 print('read from .p file')
    #             else:
    #                 data, tb, pars, info = readProtocol(protocolfilename, sparsity=None)
    #                 print('read from raw file')
    #             if writepickle:  # save the data off... moving sequences to nparrays seemed to solve a pickle problem...
    #                 pdata[protocolfilename] = {'data': data, 'tb': tb, 'info': info ,
    #                     'sequence1': np.array(pars['sequence1']['d']),
    #                     'sequence2': np.array(pars['sequence2']['index']),
    #                 }
    #                 continue
    #             results = analyze_protocol(data, tb, info, taus=[0.4, 5], sign=dataset['sign'], eventhist=plotevents)
    #             if ident not in mapfromid.keys():
    #                 print('ident: %d not in keys: ' % ident, mapfromid.keys())
    #                 continue
    #             idm = mapfromid[ident]
    #             if ident == 0:
    #                 cbar = P.axdict['A1']
    #             else:
    #                 cbar = None
    #             #if dhImage is not None:
    #             if self.AR.spotsize == None:
    #                 self.AR.spotsize=50e-6
    #             self.newvmax = self.plot_map(P.axdict[idm[0]], cbar, results['positions'], results['I_max'],
    #                 vmaxin=self.newvmax, dh=dhImage, angle=rotation, spotsize=self.AR.spotsize*1e6)
    # #            print('calling plottraces')
    #             self.plot_traces(P.axdict[idm[1]], tb, np.mean(data, axis=0), color=color_sequence[ident])
    #             # for i in range(len(results['events']['result'])):
    #             #         A.axarr[0,0].plot(tb[0:len(results['events']['criteria'][i])], results['events']['criteria'][i])
    #             #         A.axarr[1,0].plot(tb, 1e12*data[j, i, :])
    #             #print('results[eventtimes]: ', results['eventtimes'])
    #             if plotevents and len(results['eventtimes']) > 0:
    #                 axh = P.axdict[idm[2]]
    #                 y=[]
    #                 for x in range(len(results['eventtimes'])):
    #                     for xn in results['eventtimes'][x]:
    #                         y.append(xn)
    #                 axh.hist(y, 300, range=[0., 0.6], normed=1, facecolor='k')
    #                 axh.set_xlim([0., 0.6])
    #                # axh.set_ylim([0., 50.])
    #                # mpl.show()
    #             # if plotsummary and len(results['eventtimes']) > 0:
    #             #     PN = PH.regular_grid(1 , 1, order='columns', figsize=(6., 6), showgrid=False,
    #             #             verticalspacing=0.08, horizontalspacing=0.08,
    #             #             margins={'leftmargin': 0.07, 'rightmargin': 0.20, 'topmargin': 0.03, 'bottommargin': 0.1},
    #             #             labelposition=(-0.12, 0.95))
    #             #
    #             plot_all_traces(tb, data, protocolfilename)
    #         if writepickle:
    #             save_pickled(cell+'.p', pdata)
    #
    #     if not writepickle:
    #         mpl.show()


if __name__ == '__main__':
    datadir = '/Users/pbmanis/Documents/data/MRK_Pyramidal'
    parser = argparse.ArgumentParser(description='mini synaptic event analysis')
    parser.add_argument('datadict', type=str,
                        help='data dictionary')
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
    
    plan = EP.DataPlan.DataPlan(os.path.join(datadir, args.datadict)).datasets

    #print('plan: ', plan)
    if args.do_one != '':
        cell = int(args.do_one)
    else:
        cell = 1
    print('cell: ', cell)
    print('plan dict: ', plan.keys())
    print('cell: ', plan[cell]['Cell'])
    datapath = os.path.join(datadir, str(plan[cell]['Date']).strip(), str(plan[cell]['Slice']).strip(), str(plan[cell]['Cell']).strip())
    print( args)
    
    if args.do_iv:
        
        EPIV = EP.IVSummary.IVSummary(os.path.join(datapath, str(plan[cell]['IV']).strip()))
        EPIV.compute_iv()
        exit(1)
    
    if args.do_map:
        cm_sns = setMapColors(colormap, reverse=True)
        getimage = True
        plotevents = True
        dhImage = None
        rotation = 0
        # if 'getimage' in sys.argv[1:]:
        #     getimage = True
        #     cell = 'cell3_low_power_image'
        #     imagefile = os.path.join(dataplan.datadir, dataplan.datasets[cell])
        #     dhImage = DM.getDirHandle(imagefile, create=False)
        #     #print dhImage.indexFile()
        #     print (dhImage.parent().info())
        #     exit(1)
        #     # print imageInfo['deviceTransform']['scale']
        #     # print imageInfo['deviceTransform']['pos']
        #     rotation =  0.865*np.pi
        
        print('map: ', plan[cell]['Map'])
        print('plan cell: ', plan[cell])
        AM = AnalyzeMap()
        AM.display_one_map(os.path.join(datapath, str(plan[cell]['Map']).strip()),
            imagefile=os.path.join(datapath, plan[cell]['Image'])+'.tif', rotation=rotation, measuretype='I_max')
        mpl.show()