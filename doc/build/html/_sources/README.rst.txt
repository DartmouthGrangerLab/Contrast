Contrast 
========


Contrast is an implementation of the generalized contrast function described in:

.. topic:: Reference Paper
   
	   | Contrast-dependent crowding
	   |    (2020, in submission)
	   |
	   | A.M.Rodriguez, R.Granger
	   |     Dartmouth College
	   

.. topic:: Abstract

	   Visual clutter affects our ability to see: objects that
           would be identifiable on their own, may become
           unrecognizable when presented close together (“crowding”) –
           but the psychophysical characteristics of crowding have
           resisted simplification. Image properties initially thought
           to produce crowding have paradoxically yielded unexpected
           results, e.g., adding flanking objects can ameliorate
           crowding (Manassi, Sayim et al. 2012, Herzog, Sayim et
           al. 2015, Pachai, Doerig et al. 2016) The resulting theory
           revisions have been sufficiently complex and specialized as
           to make it difficult to discern what principles may
           underlie the observed phenomena. A generalized formulation
           of simple visual contrast energy is presented, arising from
           straightforward analyses of center and surround neurons in
           the early visual stream. Extant contrast measures, such as
           RMS contrast, are easily shown to fall out as reduced
           special cases. The new generalized contrast energy metric
           surprisingly predicts the principal findings of a broad
           range of crowding studies.  These early crowding phenomena
           may thus be said to arise predominantly from contrast, or
           are, at least, severely confounded by contrast
           effects. (These findings may be distinct from accounts of
           other, likely downstream, “configural” or “semantic”
           instances of crowding, suggesting at least two separate
           forms of crowding that may resist unification.) The new
           fundamental contrast energy formulation provides a
           candidate explanatory framework that addresses multiple
           psychophysical phenomena beyond crowding.


Install
-------

This program is written in Python 3.7.3.   https://www.python.org/

Ensure that the following Python packages are installed:
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

* numpy       numpy (1.18.1)
* scipy       scipy (1.4.1)
* Pillow      Pillow (5.4.1)  https://pillow.readthedocs.io/en/latest/installation.html
* matplotlib  matplotlib (3.1.2)
* pandas      pandas (0.25.3)
* PsychoPy    Psychopy (2020.1.2)
  
Install the latest version of these libraries::

  $ pip3 install numpy scipy Pillow matplotlib pandas psychopy


Running the Herzog experiment:
-------

To run Herzog experiment from the paper::
  
  $ cd experiments/herzog
  $ ./herzog.py

Results will be in the report directory::

  $ ls report
  contrast-Herzog-2012-Figure-1a.pdf  
  contrast-Herzog-2012-Figure-1b.pdf   
  contrast-Herzog-2012-Figure-1c.pdf   
  contrast-Herzog-2012-Figure-1d.pdf   
  "decision-'herzog'-Figure-1a.pdf"     
  "decision-'herzog'-Figure-1b.pdf"     
  "decision-'herzog'-Figure-1c.pdf"
  "decision-'herzog'-Figure-1d.pdf"
  Herzog-2012-Figure-1a.pdf
  Herzog-2012-Figure-1b.pdf
  Herzog-2012-Figure-1c.pdf
  Herzog-2012-Figure-1d.pdf
  $ 

All the parameters are listed at the top of each experiment file. For the Herzog
experiment the parameters are in herzog.py:
 
.. code-block:: python
		
    #foreground_color = [-1,-1,-1]
    background_color = [-1,-1,-1]
    eccentricity = 3.88
    herzog_params = Params(
	name             = 'herzog',
	expName          = 'herzog',
	viewing_distance = 75.0,
	monitor          = 'testMonitor',
	logfile          = 'herzog.log',
	window_color     = [-1,-1,-1],
	exp_info         = {u'session': u'001', u'participant': u'default'},
	cwd              = os.getcwd(),
	target_identifier= ('num_flank',0),
	stair      = Params(startVal=0, stepSizes=1, stepType='lin',
		      nReversals=0, nTrials=4, nUp=1, nDown=1,
		      minVal=0, maxVal=7, autoLog=True,
		      originPath=-1, name='staircase_trials'),
	levels     = [
			Params(exp_num=[1],
			       num_flank=[0,1,2,4,8],
			       jitter=[0],
			       flank_target_height_ratio=[0.5]),
			Params(exp_num=[2],
			       num_flank=[0,1,2,4,8],
			       jitter=[0],
			       flank_target_height_ratio=[1]),
			Params(exp_num=[3],
			       num_flank=[0,1,2,4,8],
			       jitter=[0],
			       flank_target_height_ratio=[2]),
			Params(exp_num=[4],
			       num_flank=[0,1,2,4,8],
			       jitter=[1],
			       flank_target_height_ratio=[0.5])],
	experiment = Params(eccentricity= eccentricity,
		      nTrialReps= 2,
		      nStaircaseTrials= 8),
	stimulus   = Params(eccentricity= eccentricity,
			    jitters=np.array([-0.1,  0.26, -0.87,  0.24,
					      0.86, -0.34, 0.5 , -0.51])*0.5*(40/60.0),
			    flank_distance=23.33/60.0,
			    target_orientation= 0,
			    line_height= 40/60.0,
			    line_width= 4/60.0,
			    vertical_gap= 4/60.0,
			    offset= 0.0,
			    filename= ['num_flank','jitter','offset','flank_target_height_ratio','target_orientation'],
			    offset_level= 16.66/60.0,
			    offsets= np.array([16.66, 19.04, 21.42, 23.8,
					       26.18, 28.56, 30.94, 33.32])),
	model      = Params(eccentricities= [eccentricity], # in deg
			    view_size= (600,600), # in pixels
			    view_pos= (eccentricity,0), # center in degrees of visual angle
			    est_max= 0.1,
			    upper_limit= 0.85,                   
			    lower_limit= 0.0))



To recreate the stimuli for Herzog experiment (note: various windows will appear while the stimuli are being generated)::
  
  $ cd experiments/herzog
  $ ./herzog.py -genstim
  
Running the Kahneman experiment:
-------

To run Herzog experiment from the paper::
  
  $ cd experiments/herzog
  $ ./herzog.py

Results will be in the report directory::

  $ ls report
  contrast-Kahneman-2012-Figure-1.pdf  
  "decision-'kahneman'-Figure-1.pdf"
  Kahneman-2012-Figure-1.pdf
  $ 

All the parameters are listed at the top of each experiment file. For the Kahneman
experiment the parameters are in kahneman.py:
 
.. code-block:: python
		
    foreground_color = [-1,-1,-1]
    background_color = [0.1,0.1,0.1]
    eccentricity = 0.0
    kahneman_params = Params(
	name             = 'kahneman',
	expName          = 'kahneman',
	exp_num          = 1,
	viewing_distance = 2300.0,
	monitor          = 'testMonitor',
	logfile          = 'kahneman.log',
	window_color     = background_color,
	cwd              = os.getcwd(),
	exp_info         = {u'session': u'001', u'participant': u'default'},
	target_identifier= ('flank_distance',-1/60.0),
	levels = Params(
	    flank_distance = np.array([-1., 0.06, 0.12, 0.18, 0.24, 0.6, 1.2, 1.8, 2.4, 3., 5.4 ])/60,
	    offset = [0],
	    target_orientation = [0,90,180,270]), # degrees from noon orientation
	experiment = Params(eccentricity= eccentricity,
		      nTrialReps= 1),
	stimulus   = Params(
	    eccentricity = eccentricity,
	    target_size = 0.0548,
	    gap_size = 0.01124,
	    line_width= 0.014),
	model      = Params(eccentricities= [5], # in deg
			    view_size= (1000,1000), # in pixels
			    view_pos= (eccentricity,0), # center in degrees of visual angle
			    est_max= 0.032,
			    upper_limit= 0.85,                   
			    lower_limit= 0.0))


To recreate the stimuli for Kahneman experiment (note: various windows will appear while the stimuli are being generated)::
  
  $ cd experiments/kahneman
  $ ./kahneman.py -genstim
  


Running the Pachai experiment:
-------

To run Pachai experiment from the paper::
  
  $ cd experiments/pachai
  $ ./pachai.py

Results will be in the report directory::

  $ ls report
  contrast-1-Pachai-Figure-1.pdf      
  contrast-5-Pachai-Figure-1.pdf     
  "decision-1-'pachai'-Figure-a.pdf"  
  "decision-5-'pachai'-Figure-a.pdf"
  Pachai-1-Figure-1.pdf
  Pachai-5-Figure-1.pdf
  "plot-'pachai'-barplot.pdf"
  $ 

All the parameters are listed at the top of each experiment file. For the Pachai
experiment the parameters are in pachai.py:
 
.. code-block:: python

    eccentricity = 10.0
    #background_color = [-1,-1,-1]
    background_color = [0,0,0]

    pachai_params = Params(
	name             = 'pachai',
	expName          = 'pachai',
	exp_num          = 1,
	viewing_distance = 58,
	monitor          = 'testMonitor',
	logfile          = 'pachai.log',
	exp_info         = {u'session': u'001', u'participant': u'default'},
	cwd              = os.getcwd(),
	window_color     = background_color,
	target_identifier= ('flank_distance',-1),
	levels = Params(flank_distance= [-1,0.5,0.9,1.62,2.58,3.9], # degrees of visual angle
			flank_orientation= [45,135,225,315], # degrees from noon orientation
			target_orientation= [0,90,180,270], # degrees from noon orientation
			gap= [0,1],
			num_flank= [1,5]),
	experiment = Params(eccentricity= eccentricity,
			    nTrialReps= 1),
	stimulus   = Params(line_width= 0.4,
			    gap_width= 1.2,
			    target_diameter= 2.0,
			    flank_height= 10.0),
	model      = Params(eccentricities= [eccentricity], # in deg
			    view_size= (600,600), # in pixels
			    view_pos= (eccentricity,0), # center in degrees of visual angle
			    upper_limit= 0.85,                   
			    lower_limit= 0.0))


To recreate the stimuli for Pachai experiment (note: various windows will appear while the stimuli are being generated)::
  
  $ cd experiments/pachai
  $ ./pachai.py -genstim
  

Running the Pelli experiment:
-------

To run Pelli experiment from the paper::
  
  $ cd experiments/pelli
  $ ./pelli.py

Results will be in the report directory::

  $ ls report
  "contrast-'LargeLetters'-Figure-1.pdf"  
  "contrast-'SmallLetters'-Figure-1.pdf"  
  "decision-'LargeLetters'-Figure-1.pdf"  
  "decision-'SmallLetters'-Figure-1.pdf" 
  "'LargeLetters'-Figure-1.pdf"
  "'SmallLetters'-Figure-1.pdf"
  $ 

All the parameters are listed at the top of each experiment file. For the Herzog
experiment the parameters are in herzog.py:
 
.. code-block:: python

    background_color = [1,1,1]
    eccentricity = [5,10,15,20]
    pelli_params = Params(
	name             = 'pelli',
	expName          = 'pelli',
	exp_num          = 1,
	viewing_distance = 22*2.54, 
	monitor          = 'testMonitor',
	logfile          = 'pelli.log',
	exp_info         = {u'session': u'001', u'participant': u'default'},
	cwd              = os.getcwd(),
	window_color     = background_color,
	target_identifier= ('flank_distance',-1.0),
	levels = Params(
	    flank_distance = np.array([-1.0,0.05,0.1,0.15,0.2,0.3,0.4,0.6]),
	    offset = eccentricity),
	experiment = Params(
	    nTrialReps= 2),
	stimulus   = Params(
	    name = '',
	    target_size = 1.0,
	    gap_size = 0.4,
	    stim = '02',
	    line_width= 0.4),
	model      = Params(eccentricities= eccentricity, # in deg
			    view_size= (500,500), # in pixels
			    view_pos= (0,0), # center in degrees of visual angle
			    upper_limit= 0.85,                   
			    lower_limit= 0.0))

Since we are using two different decision function in the Pelli case, one for 'Small Letters' and one for the 'LargeLetters', the target_contrast and est_max for those values are defined in the following code fragment at the bottom of the pelli.py file:

.. code-block:: python
		
    if name == 'SmallLetters':
	pelli_params['model']['target_contrast'] = 0.005
	pelli_params['model']['est_max'] = 0.01
    else:
	pelli_params['model']['target_contrast'] = 0.012
	pelli_params['model']['est_max'] = 0.025	

To recreate the stimuli for Pelli experiment (note: various windows will appear while the stimuli are being generated)::
  
  $ cd experiments/pelli
  $ ./pelli.py -genstim
  



