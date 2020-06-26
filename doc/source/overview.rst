.. Contrast documentation master file, created by
   sphinx-quickstart on Wed Jan  2 15:26:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Overview
========

Layout of files
---------------

Contrast directory
^^^^^^^^^^^^^^^^^^^

The contents of the Contrast directory are listed below:

==================   ============
Name                 Description
==================   ============
doc/                 Sphinx Documentation directory for generating documentation
experiments/         Location for individual experiments
model/               Model and Library code
==================   ============


Experiments directory
^^^^^^^^^^^^^^^^^^^^^

The Contrast/experiments directory is where the all the experiments live.  

The contents of the Contrast/experiments directory is listed below:

Commands for running all experiments:

==================   ============
Name                 Description
==================   ============
herzog/              Manassi, M., B. Sayim and M. Herzog (2012). Experiment 1.
kahneman/            Flom, M., F. Weymouth and D. Kahneman (1963). Experiment 1.
pachai/              Pachai, M., A. Doerig and M. Herzog (2016).
pelli/               Pelli, D. and K. Tillman (2008). Experiment in Figure 5.
==================   ============

Sample Experiment directory: Herzog
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section describes the files in an individual experiment directory.  In this case, the Herzog experiment but the same format will apply to the other experiments as well.

==================   ============
Name                 Description
==================   ============
data/                Place where data from the runs is stored
herzog.log           Log file for the run of the experiment
herzog.py            Code for running the herzog experiment
images/              Directory where heatmaps are stored
report/              Location for final figures from running the herzog experiment
report.py            File to generate the report figures, called automatically from herzog.py
stimuli/             Stimuli for the herzog experiment. Created by ./herzog.py -genstim
==================   ============

Code for generating the Contrast Jacobian
-----------------------------------------





