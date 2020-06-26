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
    
class KahnemanComponent(StimulusComponent):
    def __init__(self, params={}, start=0, stop=1000000):
        super(KahnemanComponent, self).__init__(params=params,start=start, stop=stop)

        self.line_width = self.params['line_width']
        self.gap_width= self.params['gap_size']
        self.target_diameter= self.params['target_size']
        self.eccentricity = self.params['eccentricity']
        self.current_depth = 1.0
        self.items = []

    def get_target_gap_pos(self,flank_orientation=0,flank_distance=0,num_flank=0,target_orientation=0,gap=0,offset=0):
        pos = pol2cart(-target_orientation+90, 0.5*self.target_diameter-0.5*self.line_width, units='deg')
        pos = (pos[0]+self.eccentricity,pos[1])
        return pos
    
    def rect(self,width=0.4,height=2.0,color=[1,1,1],ori=0.0):
        #self.current_depth = self.current_depth - 1.0
        return visual.Rect(pos=(0,0),size=[width,height],ori=ori,
                           lineColor=color,units='deg',depth=self.current_depth,
                           fillColor=color,win=self.win)

    def circle(self,width=1.0,height=1.0,color=[1,1,1]):
        #self.current_depth = self.current_depth - 1
        return visual.Polygon(
            win=self.win, units='deg', 
            edges=256, size=[width, height],
            ori=0, pos=[self.eccentricity,0],
            lineWidth=1, lineColor=color, lineColorSpace='rgb',
            fillColor=color, fillColorSpace='rgb',
            opacity=1.0, depth=self.current_depth, interpolate=True)

    def create(self,win):
        self.win = win
        self.current_depth = 0

        self.flanks = []
        flank_left = self.rect(width=self.line_width,height=self.target_diameter,color=foreground_color)
        flank_right = self.rect(width=self.line_width,height=self.target_diameter,color=foreground_color)
        flank_top = self.rect(width=self.line_width,height=self.target_diameter,color=foreground_color,ori=90.0)
        flank_bottom = self.rect(width=self.line_width,height=self.target_diameter,color=foreground_color,ori=90.0)
        self.flanks.extend([flank_left,flank_right,flank_top,flank_bottom])
        self.items.extend(self.flanks)

        self.target_outer = self.circle(width=self.target_diameter,height=self.target_diameter,color=foreground_color)
        self.target_inner = self.circle(width=self.target_diameter-2*self.line_width,height=self.target_diameter-2*self.line_width,color=background_color)
        self.target_gap = self.rect(color=background_color)
        self.items.extend([self.target_outer,self.target_inner,self.target_gap])

    def update(self,trialparams={},loopstate={}):
        flank_distance = trialparams['flank_distance']
        target_orientation = trialparams['target_orientation']

        #offset = 1.0
        offset = self.target_diameter/2.0+self.line_width
        flank_left,flank_right,flank_top,flank_bottom = self.flanks
        if flank_distance > 0:
            flank_left.setPos((self.eccentricity-flank_distance-offset,0))
            flank_right.setPos((self.eccentricity+flank_distance+offset,0))
            flank_top.setPos((self.eccentricity,-flank_distance-offset))
            flank_bottom.setPos((self.eccentricity,flank_distance+offset))
            flank_left.setOpacity(1)
            flank_right.setOpacity(1)
            flank_top.setOpacity(1)
            flank_bottom.setOpacity(1)
        else:
            flank_left.setOpacity(0)
            flank_right.setOpacity(0)
            flank_top.setOpacity(0)
            flank_bottom.setOpacity(0)
            
        self.target_gap.setPos(self.get_target_gap_pos(**trialparams))
        self.target_gap.setSize((self.gap_width, 1.2*self.line_width))
        self.target_gap.setOri(target_orientation)

import os, glob, os.path

def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))[::-1]
   
def reportmain(reportclass=None,filenames=None,only_most_recent=False):
    if filenames is None:
        filenames = sorted_ls('data')
        filenames = ['data'+ os.sep + fname for fname in filenames]

    r = reportclass(filenames=filenames,only_most_recent=only_most_recent)
    r.generate()

from report import KahnemanReport
    
# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    display_report = False
    trial_conditions = kahneman_params['levels']
    stimulus_params = kahneman_params['stimulus']
    nReps = kahneman_params['experiment']['nTrialReps']
    experiment = Experiment(kahneman_params,KahnemanReport,
                   [TrialRoutine(trial_conditions,nReps,
                     #[KahnemanRoutine(kahneman_params,
                     [Routine(
                       [KahnemanComponent(stimulus_params)])])])
    
    experiment.run()
    core.quit()
    
