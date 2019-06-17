.. Framework documentation master file, created by
   sphinx-quickstart on Wed Jan  2 15:26:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Overview
========

Layout of files
---------------

Framework directory
^^^^^^^^^^^^^^^^^^^

The Framework directory is the basic library of functionality independent of specific experiments.

The contents of the Framework directory are listed below:

==================   ============
Name                 Description
==================   ============
doc/                 Sphinx Documentation directory 
experiments/         Location for individual experiments
framework.py         Main Framework
library.py           Library file
stimulus.py          Stimulus file
==================   ============


Experiments directory
^^^^^^^^^^^^^^^^^^^^^

The Framework/experiments directory is where the all the experiments live.  There are two ways of running the system: 1) Run all experiments together, or 2) run experiments individually.

The contents of the Framework/experiments directory is listed below:

Commands for running all experiments:

==================   ============
Name                 Description
==================   ============
resetall             Deletes hidden .params files in each experiment directory from previous run
runall               Runs all the experiments
reportall            Generates reports for all experiments based on previous run
==================   ============

Sample Experiment directory: Bex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section describes the files in an individual experiment directory.  In this case, the Bex experiment but the same format will apply to the other experiments as well.

==================   ============
Name                 Description
==================   ============
.params              Hidden file containing the parameters of the last run for this experiment.
bex.cfg              Symbolic link to the main experiments.cfg file in the directory above.
bex.psyexp           Corresponding rough beginning of a Psychopy experiments that uses conditions.xlsx for the experiment conditions.
bex.py               The main experiment file. 
conditions.xlsx      Conditions for the experiment.
data/                Where the data from a run will appear
images/              The generated stimuli for the experiment
report/              Where the final figures of the experiment will appear
==================   ============


Experiments.cfg config file
---------------------------

The contents of the 'experiments.cfg' config file are used to define all parameters in the experiments.  The config file is divided into sections such as '[DEFAULT]', which is common to all experiments, and '[experiment_name]' sections, which are defined for each specific experiment.  The '[experiment_name]' sections will inherit all the parameters from the '[DEFAULT]' section and allow overriding parameters already defined in '[DEFAULT]'.

.. code-block:: html
   :linenos:
      
    [DEFAULT]
    experiment_params: 	     	['offsets']
    model_params: 		['screen_pixel_size', 'new_response_max', 'new_response_min',
                                 'viewing_distance','image_height','image_width']
    stimulus_params:		[]
    image_height: 		600   # needs to be big enough to handle the largest image
    image_width:		600
    viewing_distance: 		22.0			# inches
    screen_pixel_size: 		0.282 			# mm
    offsets: 			[0,5,10]		# degrees of visual angle
      
    [bex]
    stimulus_params:		['target_orientations', 'flank_distances','flank_orientations',
                                 'gaps']
    flank_distances:		[-1,0.5,0.92,1.62,2.58,3.9]	# degrees of visual angle outer_edge
                                                                   target to outer_edge flanker
    flank_orientations: 	[45,135,225,315]	# degrees from noon orientation
    target_orientations: 	[0,90,180,270] 		# degrees from noon orientation
    offsets: 			[10]			# degrees of visual angle from fixation
    gaps:			[0,1]			# 0=no gap present, 1=gap present
    line_width:			0.40	# degrees of visual angle
    gap_width:			0.4	# degrees of visual angle
    target_diameter:		2.0	# degrees of visual angle


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



