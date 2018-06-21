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

To read acq4 files (videos, scanning protocols, images):  ephysanalysis (https://github.com/pbmanis/ephysanalysis)
To analyze events, minis (provides Andrade-Jonas and Clements Bekkers): minis (https://github.com/pbmanis/mini_analysis)
pylibrary (plotting, findspikes) (https://github.com/pbmanis/pylibrary)

pyqtgraph
matplotlib
seaborn
xlrd
pandas
numpy
re (regular expressions)

Usage:
usage: analyzeMapData.py [-h] [-i] [-o DO_ONE] [-m DO_MAP] [-c] [-v] datadict

mini synaptic event analysis

positional arguments:
  datadict              data dictionary

optional arguments:
  -h, --help            show this help message and exit
  -i, --IV              just do iv
  -o DO_ONE, --one DO_ONE
                        just do one
  -m DO_MAP, --map DO_MAP
                        just do one map
  -c, --check           Check for files; no analysis
  -v, --view            Turn off pdf for single run
  
  

