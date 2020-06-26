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
                        gap_width= 0.4*2,
                        target_diameter= 2.0,
                        flank_height= 10.0),
    model      = Params(eccentricities= [eccentricity], # in deg
                        view_size= (600,600), # in pixels
                        view_pos= (eccentricity,0), # center in degrees of visual angle
                        upper_limit= 0.85,                   
                        lower_limit= 0.0))
    
class PachaiComponent(StimulusComponent):
    def __init__(self, params={}, start=0, stop=1000000):
        super(PachaiComponent, self).__init__(params=params,start=start, stop=stop)

        self.line_width = self.params['line_width']
        self.gap_width= self.params['gap_width']
        self.target_diameter= self.params['target_diameter']
        self.flank_height= self.params['flank_height']
        self.window_color = background_color
        self.eccentricity = eccentricity
        self.current_depth = 1.0
        self.items = []

    def get_flank_outer_size(self,flank_num,flank_orientation,flank_distance,num_flank,target_orientation,gap):
        if flank_num == 1:
            size = (2*(flank_distance+0.5*self.target_diameter),2*(flank_distance+0.5*self.target_diameter))
        else:
            size = (2*((flank_num-1)*2*self.line_width+flank_distance+0.5*self.target_diameter),2*((flank_num-1)*2*self.line_width+flank_distance+0.5*self.target_diameter))
        return size

    def get_flank_inner_size(self,flank_num,flank_orientation,flank_distance,num_flank,target_orientation,gap):
        if flank_num == 1:
            size = (2*(flank_distance+0.5*self.target_diameter-self.line_width),2*(flank_distance+0.5*self.target_diameter-self.line_width))
        else:
            size = (2*((flank_num-1)*2*self.line_width+flank_distance-self.line_width+0.5*self.target_diameter),2*((flank_num-1)*2*self.line_width+flank_distance-self.line_width+0.5*self.target_diameter))
        return size
     
    def get_target_gap_pos_triangle(self,flank_orientation,flank_distance,num_flank,target_orientation,gap):
        pos = pol2cart(-target_orientation+90, 0.5*self.target_diameter-self.line_width, units='deg')
        pos = (pos[0]+self.eccentricity,pos[1])
        return pos
    
    def get_target_gap_pos_rect(self,flank_orientation,flank_distance,num_flank,target_orientation,gap):
        pos = pol2cart(-target_orientation+90, 0.5*self.target_diameter-0.25*self.line_width, units='deg')
        pos = (pos[0]+self.eccentricity,pos[1])
        return pos
    
    def get_target_gap_pos_rect2(self,flank_orientation,flank_distance,num_flank,target_orientation,gap):
        pos = pol2cart(-target_orientation+90, 0.5*self.target_diameter-0.5*self.line_width, units='deg')
        pos = (pos[0]+self.eccentricity,pos[1])
        return pos
    
    def get_flank_gap_pos(self,flank_orientation,flank_distance,num_flank,target_orientation,gap):
        pos = pol2cart(-flank_orientation+90, 0.5*self.target_diameter+0.5*self.flank_height, units='deg')
        pos = (pos[0]+self.eccentricity,pos[1])
        return pos

    def rect(self,width=1.0,height=1.0,color=[1,1,1]):
        self.current_depth = self.current_depth - 1.0
        return visual.Rect(pos=(0,0),size=[width,height],
                           lineColor=color,units='deg',depth=self.current_depth,
                           fillColor=color,win=self.win)

    def circle(self,width=1.0,height=1.0,color=[1,1,1]):
        self.current_depth = self.current_depth - 1
        return visual.Polygon(
            win=self.win, units='deg', 
            edges=256, size=[width, height],
            ori=0, pos=[self.eccentricity,0],
            lineWidth=1, lineColor=color, lineColorSpace='rgb',
            fillColor=color, fillColorSpace='rgb',
            opacity=1.0, depth=self.current_depth, interpolate=True)

    def triangle(self,width=1.0,height=1.0,color=[1,1,1]):
        self.current_depth = self.current_depth - 1
        return visual.Polygon(
            win=self.win, units='deg', 
            edges=3, size=[width, height],
            ori=0, pos=[self.eccentricity,0],
            lineWidth=1, lineColor=color, lineColorSpace='rgb',
            fillColor=color, fillColorSpace='rgb',
            opacity=1.0, depth=self.current_depth, interpolate=True)
    
    def create(self,win):
        self.win = win
        self.current_depth = 0

        self.flanks = []
        for n in range(5,0,-1):
            flank_outer = self.circle(color=[1,1,1])
            flank_inner = self.circle(color=self.window_color)
            self.flanks.extend([(flank_outer,flank_inner)])
            self.items.extend([flank_outer,flank_inner])
        self.target_outer = self.circle(width=self.target_diameter,height=self.target_diameter,color=[1,1,1])
        self.target_inner = self.circle(width=self.target_diameter-2*self.line_width,height=self.target_diameter-2*self.line_width,color=self.window_color)
        self.target_gap_rect = self.rect(color=self.window_color) 
        self.target_gap_triangle = self.triangle(color=self.window_color)
            
        self.flank_gap = self.rect(color=self.window_color)
        self.items.extend([self.target_outer,self.target_inner,self.target_gap_rect,self.target_gap_triangle,self.flank_gap])
            
    def update(self,trialparams={},loopstate={}):
        num_flank = trialparams['num_flank']
        gap = trialparams['gap']
        flank_distance = trialparams['flank_distance']
        flank_orientation = trialparams['flank_orientation']
        target_orientation = trialparams['target_orientation']

        for n,flank_pair in enumerate(self.flanks):
            n = 5-n
            flank_outer,flank_inner = flank_pair
            flank_outer.setSize(self.get_flank_outer_size(n,**trialparams))
            flank_outer.setOpacity(int(num_flank>=n and flank_distance > -1.0))
            flank_inner.setSize(self.get_flank_inner_size(n,**trialparams))
            flank_inner.setOpacity(int(num_flank>=n and flank_distance > -1.0))

        self.target_gap_rect.setPos(self.get_target_gap_pos_rect(**trialparams))
        self.target_gap_rect.setSize((0.75*self.gap_width, 0.5*self.line_width))
        self.target_gap_rect.setOri(target_orientation)

        self.target_gap_triangle.setPos(self.get_target_gap_pos_triangle(**trialparams))
        self.target_gap_triangle.setSize((self.gap_width, 3.0*self.line_width))
        self.target_gap_triangle.setOri(target_orientation+180)

        self.flank_gap.setPos(self.get_flank_gap_pos(**trialparams))
        self.flank_gap.setSize((self.gap_width, self.flank_height))
        self.flank_gap.setOri(flank_orientation)
        self.flank_gap.setOpacity(gap)

from report import PachaiReport

# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    trial_conditions = pachai_params['levels']
    stimulus_params = pachai_params['stimulus']
    nReps = pachai_params['experiment']['nTrialReps']
    for i,num_flank in enumerate([1,5]):
        if num_flank == 1:
            pachai_params['expName'] = 'pachai:1'
            pachai_params['model']['est_max'] = 0.03
            #pachai_params['model']['est_max'] = 0.031622
            pachai_params['levels']['num_flank'] = [1]
        else:
            pachai_params['expName'] = 'pachai:5'
            pachai_params['model']['est_max'] = 0.08
            #pachai_params['model']['est_max'] = 0.077658
            pachai_params['levels']['num_flank'] = [5]
        experiment = Experiment(pachai_params,PachaiReport,
                       [TrialRoutine(trial_conditions,nReps,
                         [Routine(
                           [PachaiComponent(stimulus_params)])])])

        experiment.run(display_report=False)
    core.quit()
