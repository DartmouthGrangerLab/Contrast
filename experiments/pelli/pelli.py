#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TestDriver
"""
import numpy as np  # whole numpy lib is available, prepend 'np.'
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.getcwd(),'..','..')))

from psychopy import logging,visual,core
logging.console.setLevel(logging.CRITICAL)  # TURN OFF WARNINGS TO THE CONSOLE
from psychopy.tools.coordinatetools import pol2cart

from Contrast.model.newlibrary import Params,StimulusComponent,Experiment,TrialRoutine,StaircaseTrialRoutine,Routine

dirname = os.getcwd()

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


class PelliRoutine(Routine):
    def __init__(self,params,components=[],timeout=10):
        super(PelliRoutine, self).__init__(components=components,timeout=timeout)
        self.exp_name = params['name']

    def get_filename(self,trialparams={},loopstate={}):
        paramstr = '_'.join(['{:02}'.format(y) for x,y in list(trialparams.items())])
        return self.exp_name+'_'+paramstr+'.png'        

    def run(self,runmodel=None,genstim=False,trialparams={},params={},loopstate={}):
        offset = trialparams['offset']
        params['model']['view_pos'] = (offset,0)
        if not genstim:
            runmodel.eccentricity = offset
        #print('offset:'+str(offset))
        return super(PelliRoutine, self).run(runmodel=runmodel,genstim=genstim,trialparams=trialparams,params=params,loopstate=loopstate)


class PelliComponent(StimulusComponent):
    def __init__(self, params={}, start=0, stop=1000000):
        super(PelliComponent, self).__init__(params=params,start=start, stop=stop)

        self.line_width = self.params['line_width']
        self.gap_width= self.params['gap_size']
        self.target_diameter= self.params['target_size']
        #self.flank_height= self.params['flank_height']
        self.eccentricity = 10.0
        self.current_depth = 1.0
        self.items = []

    def letter(self,letter='a',stim='02',item='01'):
        return visual.ImageStim(self.win, image=dirname+os.sep+'templates/stim-'+stim+'-item-'+item+'.png', mask=None, units='deg', pos=(self.eccentricity, 0.0), size=None, ori=0.0, color=(1.0, 1.0, 1.0), colorSpace='rgb', contrast=1.0, opacity=1.0, depth=0, interpolate=False, flipHoriz=False, flipVert=False, texRes=128, name=None, autoLog=None, maskParams=None)
    
    def create(self,win):
        self.win = win
        self.current_depth = 0

        self.flanks = []
        stim = self.params['stim']
        flank_left = self.letter('a',stim=stim,item='01')
        flank_right = self.letter('a',stim=stim,item='01')
        self.flanks.extend([flank_left,flank_right])
        self.items.extend(self.flanks)
        self.target = self.letter('r',stim=stim,item='02')
        self.items.extend([self.target])

    def update(self,trialparams={},loopstate={}):
        flank_distance = trialparams['flank_distance']
        #flank_orientation = trialparams['flank_orientation']
        offset = trialparams['offset']
        target_orientation = 0

        flank_left,flank_right = self.flanks
        flank_width = flank_left.size[0]
        if flank_distance > 0:
            flank_left.setPos((offset-(flank_distance+flank_width),0))
            flank_right.setPos((offset+(flank_distance+flank_width),0))
            flank_left.setOpacity(1)
            flank_right.setOpacity(1)
        else:
            flank_left.setOpacity(0)
            flank_right.setOpacity(0)
            
        self.target.setPos((offset,0))

from report import PelliReport

# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    stim = ['02','03']
    for i,name in enumerate(['SmallLetters','LargeLetters']):
        pelli_params['name'] = name
        pelli_params['expName'] = name
        pelli_params['stimulus']['stim'] = stim[i]
        trial_conditions = pelli_params['levels']
        stimulus_params = pelli_params['stimulus']
        nReps = pelli_params['experiment']['nTrialReps']
        if name == 'SmallLetters':
            pelli_params['model']['target_contrast'] = 0.005
            pelli_params['model']['est_max'] = 0.01
        else:
            pelli_params['model']['target_contrast'] = 0.012
            pelli_params['model']['est_max'] = 0.025
        experiment = Experiment(pelli_params,PelliReport,
                       [TrialRoutine(trial_conditions,nReps,
                         #[PelliRoutine(pelli_params,
                         [PelliRoutine(pelli_params,
                           [PelliComponent(stimulus_params)])])])
        experiment.run()
    core.quit()
