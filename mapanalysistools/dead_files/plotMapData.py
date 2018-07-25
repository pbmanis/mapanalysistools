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
import numpy as np
import scipy.signal
import scipy.ndimage
import os.path
from collections import OrderedDict
import re
import math
import pickle
import datetime
from numba import jit

from acq4.util.metaarray import MetaArray
from acq4.analysis.dataModels import PatchEPhys
import acq4.util.functions as functions
from acq4.util import DataManager
import matplotlib.pyplot as mpl
import matplotlib.colors
import matplotlib
import colormaps.parula
import pylibrary.PlotHelpers as PH
import seaborn
#cm_sns = mpl.cm.get_cmap('terrain')  # terrain is not bad
#cm_sns = mpl.cm.get_cmap('parula')  # terrain is not bad
#cm_sns = mpl.cm.get_cmap('jet')  # jet is terrible
color_sequence = ['k', 'r', 'b']
colormap = 'snshelix'


DM = DataManager

basedir = "/Users/pbmanis/Documents/Lab/Manuscripts and Abstracts in Progress/NCAMMapping/Manis_FinalAnalysis/"
datadir = '/Volumes/Pegasus/PBM_DATA3/Xuying/Zhang_Xuying/TTX-Controls'


datasets = OrderedDict([('cell1', ['2017.01.27_000/slice_000/cell_000/Map_VGAT_Vclamp_WC_000' , 
                                   '2017.01.27_000/slice_000/cell_000/Map_VGAT_Vclamp_WC_002',
                                   '2017.01.27_000/slice_000/cell_000/Map_VGAT_Vclamp_WC_003']),
            
            ('cell2', ['' , '2017.01.27_000/slice_000/cell_001/Map_VGAT_Vclamp_WC_002',
            '']),
            
            ('cell3', ['2017.01.27_000/slice_001/cell_000/Map_VGAT_Vclamp_WC_001' , 
                       '2017.01.27_000/slice_001/cell_000/Map_VGAT_Vclamp_WC_005',
                       '2017.01.27_000/slice_001/cell_000/Map_VGAT_Vclamp_WC_006']),
                      ('cell3_low_power_image', '2017.01.27_000/slice_001/cell_000/image_006.tif'),
            
            ('cell4', ['2017.01.27_000/slice_002/cell_000/Map_VGAT_Vclamp_WC_000' , 
                       '2017.01.27_000/slice_002/cell_000/Map_VGAT_Vclamp_WC_003',
            '']),
        ])

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
        cm_sns = mpl.cm.get_cmap('terrain_r')  # terrain is not bad
    elif colormap == 'cubehelix':
        cm_sns = seaborn.cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0,
            hue=0.8, light=0.85, dark=0.15, reverse=reverse, as_cmap=False)
    elif colormap == 'snshelix':
        cm_sns = seaborn.cubehelix_palette(n_colors=64, start=3, rot=0.5, gamma=1.0, dark=0, light=1.0, reverse=reverse,
         as_cmap=True)
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
        import colormaps.parula
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('parula', colormaps.parula.cm_data)
    else:
        print('Unrecongnized color map {0:s}; setting to "parula"'.format(colormap))
        import colormaps.parula
        cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('parula', colormaps.parula.cm_data)
    return cm_sns    


def readProtocol(protocolFilename, records=None, sparsity=None):
    dh = DM.getDirHandle(protocolFilename, create=False)
    if records is None:
#       print 'protocolfile: ', protocolFilename
#        print 'info: ', dh.info()
        try:
            records = range(0, len(dh.info()['sequenceParams'][('Scanner', 'targets')]))
        except:
            raise StandardError("File not readable or not found: %s" % protocolFilename)
            exit()
    else:
        records = sorted(records)
    print ('Processing Protocol: %s' % protocolFilename)
    (rest, mapnumber) = os.path.split(protocolFilename)
#    protocol = dh.name()
    PatchEPhys.cell_summary(dh)
    dirs = dh.subDirs()
    pars = {}
    if records is not None:
        pars['select'] = records
    Clamps = PatchEPhys.GetClamps()
    modes = []
    clampDevices = PatchEPhys.getClampDeviceNames(dh)
    # must handle multiple data formats, even in one experiment...
    if clampDevices is not None:
        data_mode = dh.info()['devices'][clampDevices[0]]['mode']  # get mode from top of protocol information
    else:  # try to set a data mode indirectly
        if 'devices' not in dh.info().keys():
            devices = 'not parsed'
        else:
            devices = dh.info()['devices'].keys()  # try to get clamp devices from another location
        for kc in PatchEPhys.knownClampNames():
            if kc in devices:
                clampDevices = [kc]
        try:
            data_mode = dh.info()['devices'][clampDevices[0]]['mode']
        except:
            data_mode = 'Unknown'
    if data_mode not in modes:
        modes.append(data_mode)
    sequence = PatchEPhys.listSequenceParams(dh)
    pars['sequence1'] = {}
    pars['sequence2'] = {}
    reps = sequence[('protocol', 'repetitions')]
    if sparsity is None:
        targets = range(len(sequence[('Scanner', 'targets')]))
    else:
        targets = range(0, len(sequence[('Scanner', 'targets')]), sparsity)
    pars['sequence1']['index'] = reps
    pars['sequence2']['index'] = targets
    try:
        del pars['select']
    except KeyError:
        pass
    ci = Clamps.getClampData(dh, pars)
    info = {}
    rep = 0
    tar = 0
    for i, directory_name in enumerate(dirs):  # dirs has the names of the runs within the protocol, is a LIST
#        if sparsity is not None and i % sparsity != 0:
#            continue
        data_dir_handle = dh[directory_name]  # get the directory within the protocol
        pd_file_handle = PatchEPhys.getNamedDeviceFile(data_dir_handle, 'Photodiode')
        pd_data = pd_file_handle.read()
        
        # if i == 0: # wait until we know the length of the traces
        #     data = np.zeros((len(reps), len(targets), len(Clamps.traces[i])))
        #     print Clamps.traces.shape
        # d = Clamps.traces[i]
        # data[rep, tar, :] = d
        info[(rep, tar)] = {'directory': directory_name, 'rep': rep, 'pos': data_dir_handle.info()['Scanner']['position']}
        tar = tar + 1
        if tar > len(targets):
            tar = 0
            rep = rep + 1
        DM.cleanup()  # close all opened files
    DM.cleanup()
    data = np.reshape(Clamps.traces, (len(reps), len(targets), Clamps.traces.shape[1]))
    return data, Clamps.time_base, pars, info

def calculate_charge(tb, data):
    """
    One-dimensional data...
    """
    tbase = [0., 0.1]
    tresp = [0.105, 0.3]
    # get indices for the integration windows
    tbindx = np.where((tb >= tbase[0]) & (tb < tbase[1]))
    trindx = np.where((tb >= tresp[0]) & (tb < tresp[1]))
    Qr = 1e6*np.sum(data[trindx])/(tresp[1]-tresp[0]) # response
    Qb = 1e6*np.sum(data[tbindx])/(tbase[1]-tbase[0])  # baseline
    return Qr, Qb

def ZScore(tb, data):
    # abs(post.mean() - pre.mean()) / pre.std()
    tbase = [0., 0.1]
    tresp = [0.105, 0.3]
    # get indices for the integration windows
    tbindx = np.where((tb >= tbase[0]) & (tb < tbase[1]))
    trindx = np.where((tb >= tresp[0]) & (tb < tresp[1]))
    mpost = np.mean(data[trindx]) # response
    mpre = np.mean(data[tbindx])  # baseline
    return(np.fabs((mpost-mpre)/np.std(data[tbindx])))

def Imax(tb, data, sign=1):
    tbase = [0., 0.1]
    tresp = [0.105, 0.3]
    tbindx = np.where((tb >= tbase[0]) & (tb < tbase[1]))
    trindx = np.where((tb >= tresp[0]) & (tb < tresp[1]))
    mpost = np.max(sign*data[trindx]) # response goes negative... 
    return(mpost)
    
@jit(nopython=True, cache=True)
def clementsBekkers(data, template):
    """Implements Clements-bekkers algorithm: slides template across data,
    returns array of points indicating goodness of fit.
    Biophysical Journal, 73: 220-229, 1997.
    """
    
    ## Strip out meta-data for faster computation
    D = data # data.view(np.ndarray)
    T = template # template.view(np.ndarray)
    
    ## Prepare a bunch of arrays we'll need later
    N = len(T)
    sumT = T.sum()
    sumT2 = (T**2.0).sum()
    D2 = D**2.0
    NDATA = len(data)
    crit = np.zeros(NDATA)

    sumD = np.zeros((NDATA-N))
    sumD2 = np.zeros((NDATA-N))
    sumDTprod = np.zeros((NDATA-N))
    sumD[0] = D[:N].sum()
    sumD2[0] = D2[:N].sum()
    # sumD = rollingSum(D[:N], N)
    # sumD2 = rollingSum(D2[:N], N)
    for i in range(NDATA-N):
        if i > 0:
            sumD[i] = sumD[i-1] + D[i+N] - D[i]
            sumD2[i] = sumD2[i-1] + D2[i+N] - D2[i]
        sumDTprod[i] = (D[i:N+i]*T).sum()
    S = (sumDTprod - sumD*sumT/N)/(sumT2 - sumT*sumT/N)
    C = (sumD - S*sumT)/N
    SSE = sumD2 + (S*S*sumT2) + (N*C*C) - 2.0*(S*sumDTprod + C*sumD - (S*C*sumT))
    crit = S/np.sqrt(SSE/(N-1))
    DC = S / crit
    return DC, S, crit

@jit(nopython=True, cache=True)
def rollingSum(data, n):
    d1 = data.copy()
    d1[1:] = np.cumsum(d1[1:]) # d1[1:] + d1[:-1]  # integrate
    d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
    d2[0] = d1[n-1]  # copy first point
    d2[1:] = d1[n:] - d1[:-n]  # subtract
    return d2
    
@jit(nopython=True, cache=True)
def clementsBekkers2(data, template):
    """Implements Clements-bekkers algorithm: slides template across data,
    returns array of points indicating goodness of fit.
    Biophysical Journal, 73: 220-229, 1997.
    
    Campagonla's version...
    """
    
    ## Strip out meta-data for faster computation
    D = data # data.view(np.ndarray)
    T = template # template.view(np.ndarray)
    NDATA = len(D)
    ## Prepare a bunch of arrays we'll need later
    N = len(T)
    sumT = T.sum()
    sumT2 = (T**2.).sum()
    sumD = rollingSum(D, N)
    sumD2 = rollingSum(D**2., N)
    sumTD = scipy.signal.correlate(D, T, mode='valid')
    
    ## compute scale factor, offset at each location:
    scale = (sumTD - sumT * sumD /N) / (sumT2 - sumT**2. /N)
    offset = (sumD - scale * sumT) /N
    
    ## compute SSE at every location
    SSE = sumD2 + scale**2.0 * sumT2 + N * offset**2. - 2. * (scale*sumTD + offset*sumD - scale*offset*sumT)
    ## finally, compute error and detection criterion
    error = np.sqrt(SSE / (N-1))
    DC = scale / error
    return DC, scale, offset
    
@jit(cache=True)  # nopython chokes on the np.argwhere call
def cbTemplateMatch(data, template, threshold=3.0):
    dc, scale, crit = clementsBekkers(data, template)
    mask = crit > threshold
    diff = mask[1:] - mask[:-1]
    times = np.argwhere(diff==1)[:, 0]  ## every time we start OR stop a spike
    
    ## in the unlikely event that the very first or last point is matched, remove it
    if abs(crit[0]) > threshold:
        times = times[1:]
    if abs(crit[-1]) > threshold:
        times = times[:-1]
    
    nEvents = len(times) / 2
    result = np.empty(nEvents, dtype=[('peak', int), ('dc', float), ('scale', float), ('offset', float)])
    for i in range(nEvents):
        i1 = times[i*2]
        i2 = times[(i*2)+1]
        d = crit[i1:i2]
        p = np.argmax(d)
        result[i][0] = p+i1
        result[i][1] = d[p]
        result[i][2] = scale[p+i1]
        result[i][3] = crit[p+i1]
    return result, crit

def analyze_protocol(data, tb, info, eventhist=True):
    cutoff = 5000. # LPF at Hz
    filtfunc = scipy.signal.filtfilt
    rate = np.mean(np.diff(tb))  # t is in seconds, so freq is in Hz
    samplefreq = 1.0/rate
    nyquistfreq = samplefreq/1.95
    print ('nyquist, rate: ', nyquistfreq, rate)
    wn = cutoff/nyquistfreq 
    print( 'wn: ', wn )# 
    b, a = scipy.signal.bessel(2, wn)

    for r in range(data.shape[0]):
        for t in range(data.shape[1]):
            data[r,t,:] = filtfunc(b, a, data[r, t, :] - np.mean(data[r, t, 0:250]))
    mdata = np.mean(data, axis=0)  # mean across ALL reps

    # make visual maps
    Qr = np.zeros(data.shape[1])
    Qb = np.zeros(data.shape[1])
    Zscore = np.zeros(data.shape[1])
    I_max = np.zeros(data.shape[1])
    pos = np.zeros((data.shape[1], 2))
    for t in range(data.shape[1]):  # compute for each target
        Qr[t], Qb[t] = calculate_charge(tb, mdata[t,:])
        Zscore[t] = ZScore(tb, mdata[t,:])
        I_max[t] = Imax(tb, data[0,t,:], sign=-1)*1e12  # just the FIRST pass
        pos[t,:] = [info[(0,t)]['pos'][0], info[(0,t)]['pos'][1]]
    
    events = {}
    eventlist = []  # event histogram across ALL events/trials
    if eventhist:
        # try CB fitting:
        print('    ...identifying individual events')
    # v = [amplitude, x offset, rise tau, decay tau]
        v = [-1.0, 0., 1e-3, 25e-3]
        x = np.linspace(0., 0.02, int(0.05/rate))
        template = functions.pspFunc(v, x, risePower=2.0)
        for j in range(data.shape[0]):  # all trials
            result = []
            crit = []
            for i in range(data.shape[1]):  # all targets
                res, criteria = cbTemplateMatch(1e12*data.view(np.ndarray)[j, i, :], template.view(np.ndarray), threshold=4.0)
                result.append(res)
                crit.append(criteria)
                for r in res:
                    if r[0] > 0.5:
                        eventlist.append(r[0]*rate)
            events[j] = {'criteria': crit, 'result': result}
    return{'Qr': Qr, 'Qb': Qb, 'ZScore': Zscore, 'I_max': I_max, 'positions': pos, 'events': events, 'eventtimes': eventlist}

def scale_and_rotate(poslist, sign=[1., 1.], scaleCorr=1., scale=1e6, autorotate=False, angle=0.):
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

def reorder(a, b):
    """
    make sure that b > a
    if not, swap and return
    """
    if a > b:
        t = b
        b = a
        a = t
    return(a, b)

def plot_map(axp, axcbar, pos, measure, vmaxin=None, dh=None, angle=0):
    
    pz = [55673., 1700.]
    if dh is not None:
        img = dh.read()
        print( dir(dh))
        exit(1)
        # compute the extent for the image, offsetting it to the map center position
        ext_left = imageInfo['deviceTransform']['pos'][0]*1e6 - pz[0]
        ext_right = ext_left + imageInfo['region'][2]*imageInfo['deviceTransform']['scale'][0]*1e6
        ext_bottom = imageInfo['deviceTransform']['pos'][1]*1e6 - pz[1]
        ext_top = ext_bottom + imageInfo['region'][3]*imageInfo['deviceTransform']['scale'][1]*1e6
        ext_left, ext_right = reorder(ext_left, ext_right)
        ext_bottom, ext_top = reorder(ext_bottom, ext_top)
        # extents are manually adjusted - something about the rotation should be computed in them first...
        # but fix it manually... worry about details later.
        # yellow cross is aligned on the sample cell for this data now
        extents = [ext_bottom-125, ext_top-125, ext_left-160, ext_right-160] # [ext_left, ext_right, ext_bottom, ext_top]
        if angle != 0.:  # note that angle arrives in radians - must convert to degrees for this one.
            img = scipy.ndimage.interpolation.rotate(img, angle*180./np.pi + 90., axes=(1, 0),
                reshape=True, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        axp.imshow(img, extent=extents, aspect='auto', origin='lower')
    vmin = 0.
    # adjust vmax's to use rational values. 
    if vmaxin is not None:
        vmax = vmaxin
    else:
        vmax = np.max(measure)
    scaler = PH.NiceScale(0, vmax)
    vmax = scaler.niceMax

    pos[:,0] = (pos[:,0]*1e6-pz[0])
    pos[:,1] = (pos[:,1]*1e6-pz[1])
    pos = scale_and_rotate(pos, scale=1.0, angle=angle)
    # note circle size is radius
    pm = PH.circles(pos[:,0], pos[:,1], 15., c=np.array(measure),
                        vmax=vmax, vmin=vmin,
                     cmap=cm_sns, ax=axp, edgecolors='k', linewidths=0.2, alpha=0.9)
    axp.plot([-20., 20.], [0., 0.], '-', color='r') # cell centered coorinates
    axp.plot([0., 0.], [-20., 20.], '-', color='r') # cell centered coorinates

    tickspace = scaler.tickSpacing
    ntick = 1 + int(vmax/tickspace)
    ticks = np.linspace(0, vmax, num=ntick, endpoint=True)
    if axcbar is not None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        c2 = matplotlib.colorbar.ColorbarBase(axcbar, cmap=cm_sns, ticks=ticks, norm=norm)
        c2.ax.tick_params(axis='y', direction='out')
    axr = 250.
    axp.set_xlim([-axr, axr])
    axp.set_ylim([-axr, axr])
    axp.set_aspect('equal', 'datalim')

    if vmaxin is None:
        return vmax
    else:
        return vmaxin

def plot_traces(ax, tb, mdata, color='k'):
    ax.plot(tb, np.mean(mdata, axis=0)*1e12, color)
    ax.set_xlim([0., 0.300])
    ax.set_ylim([-100., 20.])

def shortName(name):
    (h, pr) = os.path.split(name)
    (h, cell) = os.path.split(h)
    (h, sliceno) = os.path.split(h)
    (h, day) = os.path.split(h)
    return(os.path.join(day, sliceno, cell, pr))

def save_pickled(dfile, data):
    now = datetime.datetime.now().isoformat()
    dstruct = {
                'date': now, 
                'data': data,
              }
    print('\nWriting to {:s}'.format(dfile))
    fn = open(dfile, 'wb')
    pickle.dump(dstruct, fn)
    fn.close()

def read_pickled(dfile):
#    pfname = os.path.join(self.paperpath, pfname) + '.p'
    fn = open(dfile + '.p', 'rb')
    data = pickle.load(fn)
    fn.close()
    return(data)

def run_analysis(celln, dhImage, rotation):
    if celln not in datasets.keys():
        raise ValueError('Cell %s not in list' % celln)
    writepickle = False
    if 'writep' in sys.argv[1:]:
        writepickle=True
    plotFlag = True
    protodata = {}
    nmax = 1000

    imgw = 0.25 # image box width
    imgh = 0.25
    l_c1 = 0.1  # column 1 position
    l_c2 = 0.50 # column 2 position
    trw = 0.32  # trace x width
    trh = 0.10  # trace height
    trs = imgh - trh  # 2nd trace position (offset from top of image box)
    y = 0.08 + np.arange(0., 0.7, imgw+0.05)  # y positions 
    plotspecs = OrderedDict([('A', {'pos': [l_c1, imgw, y[2], imgh]}), 
                             ('B', {'pos': [l_c2, trw, y[2]+trs, trh]}),
                             ('C', {'pos': [l_c2, trw, y[2], trh]}),
                             ('D', {'pos': [l_c1, imgw, y[1], imgh]}), 
                             ('E', {'pos': [l_c2, trw, y[1]+trs, trh]}), 
                             ('F', {'pos': [l_c2, trw, y[1], trh]}),
                             ('G', {'pos': [l_c1, imgw, y[0], imgh]}), 
                             ('H', {'pos': [l_c2, trw, y[0]+trs, trh]}), 
                             ('I', {'pos': [l_c2, trw, y[0], trh]}),
                             ('A1', {'pos': [l_c1+imgw+0.01, 0.012, y[2], imgh]})])
    P = PH.Plotter(plotspecs, label=False, figsize=(8., 6.))
    #PH.show_figure_grid(P.figure_handle)
    mapfromid = {0: ['A', 'B', 'C'], 1: ['D', 'E', 'F'], 2: ['G', 'H', 'I']}
    # set up low-pass filter
    newvmax = None 
    for ic, cell in enumerate(datasets.keys()):
        if cell not in [celln]:
            continue
        imagefile = None
        if cell == 'cell3_low_power_image':
            imagefile = os.path.join(datadir, datasets[cell])
            dhImage = DM.getDirHandle(protocolFilename, create=False)
            print (dir(dhImage))
            exit(1)
        if writepickle:
            print('Will write pickled file')
            pdata = OrderedDict()
        dataset = datasets[cell]
        for id, d in enumerate(dataset):
            if d == '':
                continue
            print ('Cell: ', cell)
            protocolfilename = os.path.join(datadir, d)
             # check to see if pickled file exists first - it is faster to read
            if os.path.isfile(cell + '.p') and not writepickle:
                d = read_pickled(cell)
                d = d['data']
                dx = d[protocolfilename]
                data = dx['data']
                tb = dx['tb']
                info = dx['info']
                pars = {'sequence1': dx['sequence1'], 'sequence2': dx['sequence2']}
                print('read from .p file')
            else:
                data, tb, pars, info = readProtocol(protocolfilename, sparsity=None)
                print('read from raw file')
            if writepickle:  # save the data off... moving sequences to nparrays seemed to solve a pickle problem...
                pdata[protocolfilename] = {'data': data, 'tb': tb, 'info': info ,
                    'sequence1': np.array(pars['sequence1']['d']),
                    'sequence2': np.array(pars['sequence2']['index']),
                }
                continue
            results = analyze_protocol(data, tb, info, eventhist=plotevents)

            idm = mapfromid[id]
            print ('id, idm: ', id, idm)
            if id == 0:
                cbar = P.axdict['A1']
            else:
                cbar = None
            if dhImage is not None:
                newvmax = plot_map(P.axdict[idm[0]], cbar, results['positions'], results['I_max'], vmaxin=newvmax, dh=dhImage, angle=rotation)
            plot_traces(P.axdict[idm[1]], tb, np.mean(data, axis=0), color=color_sequence[id])
            # for i in range(len(results['events']['result'])):
            #         A.axarr[0,0].plot(tb[0:len(results['events']['criteria'][i])], results['events']['criteria'][i])
            #         A.axarr[1,0].plot(tb, 1e12*data[j, i, :])
            if plotevents:
                axh = P.axdict[idm[2]]
                axh.hist(results['eventtimes'], 250, range=[0., 0.5], normed=1, facecolor='k')
                axh.set_xlim([0., 0.3])
                axh.set_ylim([0., 50.])

        if writepickle:
            save_pickled(cell+'.p', pdata)

    if not writepickle:
        mpl.show()

if __name__ == '__main__':
    cm_sns = setMapColors(colormap, reverse=True)
    getimage = True
    plotevents = True
    dhImage = None
    rotation = 0
    if 'getimage' in sys.argv[1:]:
        getimage = True
        cell = 'cell3_low_power_image'
        imagefile = os.path.join(datadir, datasets[cell])
        dhImage = DM.getDirHandle(imagefile, create=False)
        #print dhImage.indexFile()
        print (dhImage.parent().info())
        exit(1)
        # print imageInfo['deviceTransform']['scale']
        # print imageInfo['deviceTransform']['pos']
        rotation =  0.865*np.pi

    
    celln = sys.argv[1]
    run_analysis(celln, dhImage, rotation)
