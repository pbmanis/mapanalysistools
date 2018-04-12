mapanalysistools
================

This is a small repository that provides some tools for analysis of laser-scanning photostimulation maps. 

getTable: a program that reads the datasummary table (see ephysanalysis), or can look for a protocol directory,
displays the waveforms, and can use the Andreade-Jonas deconvolution algorithm to identify events. 
Events are marked on the traces. Also displayed are the average events, fits, and template, and
a histogram of event times across all trials.

analyzeMapData: a program that is similar to getTable, without the GUI. Generates a matplotlib
display of the map grid (color coded by amplitude), average waveforms, and the event histogram.

plotMapData is the routine that was used in the Zhang et al 2017 paper for the TTX plots (similar to analyzeMapData).

Dependencies
------------

To read acq4 files (videos, scanning protocols, images):  ephysanalysis

pyqtgraph
matplotlib

pylibrary (https://github.com/pbmanis/pylibrary)
minis (https://github.com/pbmanis/mini_analysis)

