#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TestDriver
"""
import numpy as np  # whole numpy lib is available, prepend 'np.'
import os, glob, os.path
import sys
sys.path.append(os.path.realpath(os.path.join(os.getcwd(),'..','..')))

from psychopy import logging,visual,core
logging.console.setLevel(logging.CRITICAL)  # TURN OFF WARNINGS TO THE CONSOLE

from Contrast.model.newlibrary import Params,StimulusComponent,Experiment,TrialRoutine,StaircaseTrialRoutine,Routine

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
                        est_max= 0.08,
                        upper_limit= 0.85,                   
                        lower_limit= 0.0))

class HerzogRoutine(Routine):
    def __init__(self,params,components=[],timeout=10):
        super(HerzogRoutine, self).__init__(components=components,timeout=timeout)

        self.exp_name = params['name']
        self.offsets = params['stimulus']['offsets']
        nStaircaseTrials = params['experiment']['nStaircaseTrials']
        nTrialReps = params['experiment']['nTrialReps']
        self.target_orientations=np.tile(np.array((-1,1,)*int(nStaircaseTrials/2)),(nTrialReps,1))
        self.params = params
        for ii in range(nTrialReps):
            np.random.shuffle(self.target_orientations[ii])

    def update_levels(self,levels):
        min = self.params['stair']['minVal']
        max = self.params['stair']['maxVal']                                        
        stepSizes = self.params['stair']['stepSizes']
        nTrials = self.params['stair']['nTrials']
        lvls = np.arange(min,max+1,stepSizes)
        #levels.update(level=lvls[0:nTrials].tolist())
        
        nStaircaseTrials = 500  # in order to generate all possible stimuli, we set this to a large number here so we have enough target_orientations to cover all possible stimuli
        nTrialReps = 2
        self.target_orientations=np.tile(np.array((-1,1,)*int(nStaircaseTrials/2)),(nTrialReps,1))
        
        levels.update(level=lvls.tolist())
        levels.update(target_orientation=[1,-1])

    def get_answer(self,trialparams={},loopstate={}):
        target_orientation = trialparams['target_orientation']

        if target_orientation == -1:
            correct_answer = 'left'
            incorrect_answer = 'right'
        else:
            correct_answer = 'right'
            incorrect_answer = 'left'

        return correct_answer,incorrect_answer

    def get_filename(self,trialparams={},loopstate={}):
        #logging.exp('filename:'+str(trialparams))
        target_orientation = trialparams['target_orientation']
        level = trialparams['level']
        paramstr = '_'.join(['{:02}'.format(y) for x,y in list(trialparams.items())])
        return self.exp_name+'_'+paramstr+'.png'        

    def update_params(self,trialparams={},loopstate={}):
        target_orientation  = self.target_orientations[loopstate['trials.thisRepN'],loopstate['staircase.thisTrialN']]
        trialparams.update(target_orientation=target_orientation)
    
class HerzogComponent(StimulusComponent):
    def __init__(self, params={}, exp_num=1, eccentricity=3.88, flank_target_height_ratio=1.0, target_orientation=1, num_flank=4, flank_distance=1, start=0, stop=1000000):
        super(HerzogComponent, self).__init__(params=params,start=start, stop=stop)

        #import pdb; pdb.set_trace()

        self.eccentricity = self.params['eccentricity']
        self.flank_target_height_ratio = flank_target_height_ratio
        self.exp_num = exp_num
        self.line_width = self.params['line_width']
        self.line_height = self.params['line_height']
        self.offset = self.params['offset']
        self.offsets = self.params['offsets']
        self.flank_distance = self.params['flank_distance']
        self.vertical_gap = self.params['vertical_gap']
        self.offset_level = self.params['offset_level']
        self.jitters = self.params['jitters']
        self.nojitters = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                  
        self.current_depth = 1.0

    def rect(self,width=None,height=None):
        if width is None:
            width = self.line_width
        if height is None:
            height = self.line_height
        self.current_depth = self.current_depth - 1.0
        return visual.Rect(pos=(0,0),size=[width,height],interpolate=False,
                           lineColor=[1,1,1],units='deg',depth=self.current_depth,
                           lineWidth=1,
                           fillColor=[1,1,1],win=self.win)

    def create(self,win):
        self.win = win
        self.top_guide = self.rect()
        self.bottom_guide = self.rect()
        self.left_bar = self.rect()
        self.right_bar = self.rect()

        self.items.extend([self.top_guide,self.bottom_guide,self.left_bar,self.right_bar])
        self.flanks = []
        for n in range(1,len(self.jitters)+1):
            left_flank = self.rect(self.line_width,2*self.flank_target_height_ratio*self.line_height)
            right_flank = self.rect(self.line_width,2*self.flank_target_height_ratio*self.line_height)
            self.flanks.extend([(left_flank,right_flank)])
            self.items.extend([left_flank,right_flank])

    def update(self,trialparams={},loopstate={}):
        target_orientation = trialparams['target_orientation']
        level = trialparams['level']
        num_flank = trialparams['num_flank']
        flank_target_height_ratio = trialparams['flank_target_height_ratio']
        jitter = bool(trialparams['jitter'])
        
        if jitter:
            jitters = self.jitters
        else:
            jitters = self.nojitters

        self.offset = self.offsets[int(level)]
        self.top_guide.setPos((self.eccentricity, 100/60.0+self.line_height/2))
        self.bottom_guide.setPos((self.eccentricity, -100/60.0-self.line_height/2))
        self.left_bar.setPos((self.eccentricity-(self.offset/60.0)/2, -target_orientation*(self.vertical_gap/2+self.line_height/2)), log=False)
        self.right_bar.setPos((self.eccentricity+(self.offset/60.0)/2, target_orientation*(self.vertical_gap/2+self.line_height/2)), log=False)
        for n,flank_pair in enumerate(self.flanks):
            left_flank,right_flank = flank_pair
            left_flank.setPos((self.eccentricity-(self.offset_level/2+self.flank_distance*(n+1)), jitters[n]))
            left_flank.setSize((self.line_width,2*flank_target_height_ratio*self.line_height))
            left_flank.setOpacity(int(num_flank>=n+1))
            right_flank.setPos((self.eccentricity+(self.offset_level/2+self.flank_distance*(n+1)), jitters[n]))
            right_flank.setSize((self.line_width,2*flank_target_height_ratio*self.line_height))
            right_flank.setOpacity(int(num_flank>=n+1))

def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))[::-1]
   
def reportmain(reportclass=None,filenames=None,only_most_recent=False):
    if filenames is None:
        filenames = sorted_ls('data')
        filenames = ['data'+ os.sep + fname for fname in filenames]

    r = reportclass(filenames=filenames,only_most_recent=only_most_recent)
    r.generate()

from Contrast.model.plotting import plot_figure
from report import HerzogReport
    
# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    trial_conditions = herzog_params['levels']
    stair_params = herzog_params['stair']
    stimulus_params = herzog_params['stimulus']
    nReps = herzog_params['experiment']['nTrialReps']
    level_values = herzog_params['stimulus']['offsets']
    for i,trial_condition in enumerate(trial_conditions):
        herzog_params['levels'] = trial_condition
        herzog_params['expName'] = herzog_params['name']+':'+str(i+1)
        experiment = Experiment(herzog_params,HerzogReport,
                      [TrialRoutine(trial_condition,nReps,
                        [StaircaseTrialRoutine(stair_params,level_values,
                          [HerzogRoutine(herzog_params,
                            [HerzogComponent(stimulus_params)])])])])
    
        experiment.run(display_report=False)

    experiment.displayreport()
    core.quit()
