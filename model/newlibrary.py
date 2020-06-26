#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TestDriver
Must run with Python 3.7 or greater
"""
from psychopy import visual, core, data, event, logging, clock, monitors
from psychopy.tools.monitorunittools import deg2pix, pix2cm
import numpy as np  
import os  
import sys  
import random
import itertools
import pandas as pd
from threading import Timer
import _thread
from psychopy.hardware import keyboard
logging.console.setLevel(logging.CRITICAL)  # TURN OFF WARNINGS TO THE CONSOLE
import argparse
import traceback
import glob
try:
    from PIL import Image
except ImportError:
    import Image
    
np.seterr(divide='ignore')
from Contrast.model.model import Model

def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

class Response(object):
    def __init__(self, key='', rt=0, correct=False, prob=0.0, contrast=0.0,level=None):
        self.key = key
        self.rt = rt
        self.correct = correct
        self.prob = prob
        self.contrast = contrast
        self.level = None

class Params(dict):
    #def __init__(self, experiment=[], model=[], stimulus=[], levels=[],**extra):
    def __init__(self, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        result = ''
        for key,val in self.items():
            if type(val) == type(self):
                result = result + str(key) + ' Params:\n' + str(val) 
            else:
                result = result + '   ' + str(key) + ': ' + str(repr(val)) + '\n'      
        return result

    def flatten(self,prefix=''):
        result = []
        for name,value in self.items():
            if type(value) == type(self):
                result.extend(value.flatten(prefix=name+'.'))
            else:
                value = str(repr(value))
                item_tuple = (prefix+name,value)
                result.append(item_tuple)
        return result

class StimulusComponent(object):
    """Documentation for model

    More documentation

    :Parameters:
        visible : **True** or False
            documentation on parameter
        newPos : **None** or [x,y]
            documentation on parameter
    """

    def __init__(self, start=0, stop=1000000, name=None,params={}):
        self.name = name
        self.items = []
        self.params = params
        self.saveTrials = False
        self.win = None
        self.startTime = start
        self.stopTime = stop

    def update(self,trialparams={},loopstate={}):
        pass

    def create(self,win):
        self.win = win

    def run(self,genstim=False,trialparams={},loopstate={}):
        #self.params.update(**params)
        if self.win is not None:
            self.startThread = Timer(self.startTime, self.start, (trialparams,loopstate))
            self.stopThread = Timer(self.stopTime, self.stop, ())
            self.startThread.start()
            self.stopThread.start()

    def end(self):
        self.startThread.cancel()
        self.stopThread.cancel()

    def start(self,trialparams={},loopstate={}):
        try:
            self.update(trialparams,loopstate)
            for item in self.items:
                if hasattr(item,"setAutoDraw"):
                    item.setAutoDraw(True)
                    
        except Exception as e:
            import threading
            from psychopy import core

            logging.error(traceback.format_exc())
            mainthread = None
            current_thread = threading.currentThread()
            for th in threading.enumerate():
                #logging.error('threads:'+' '.join([x for x in threading.enumerate()]))
                if th.name == 'MainThread':
                    mainthread = th
                elif th.name != current_thread.name:
                    #logging.error('stopping stimulus with name='+str(self.name))
                    th.cancel()

            _thread.interrupt_main()
            _thread.exit()

    def stop(self):
        for item in self.items:
            if hasattr(item,"setAutoDraw"):
                item.setAutoDraw(False)
        
        
class ImageComponent(StimulusComponent):
    def __init__(self, image=None, start=0, stop=1000000, mask=None, units='deg', pos=(0.0, 0.0), size=None, ori=0.0, color=(1.0, 1.0, 1.0), colorSpace='rgb', contrast=1.0, opacity=1.0, depth=0, interpolate=False, flipHoriz=False, flipVert=False, texRes=128, name=None, autoLog=None, maskParams=None):
        # Initialize components for Routine "Instructions"
        params = {'image':image, 'mask':mask, 'units':units, 'pos':pos, 'size':size, 'ori':ori, 'color':color, 'colorSpace':colorSpace, 'contrast':contrast, 'opacity':opacity, 'depth':depth, 'interpolate':interpolate, 'flipHoriz':flipHoriz, 'flipVert':flipVert, 'texRes':texRes, 'name':name, 'autoLog':autoLog, 'maskParams':maskParams}
        super(ImageComponent, self).__init__(start=start, stop=stop, **params)   

    def create(self,win):
        self.items.append(visual.ImageStim(win=win, **self.params))
        
class SoundComponent(StimulusComponent):
    def __init__(self):
        super(SoundComponent, self).__init__()   

class PolygonComponent(StimulusComponent):
    def __init__(self, edges=3, radius=0.5, start=0, stop=1000000, units='deg', lineWidth=1.5, lineColor='white', lineColorSpace='rgb', fillColor=None, fillColorSpace='rgb', vertices=((-0.5, 0), (0, 0.5), (0.5, 0)), closeShape=True, pos=(0, 0), size=1, ori=0.0, opacity=1.0, contrast=1.0, depth=0, interpolate=True, name=None, autoLog=None, autoDraw=False):
        params = {'edges':edges, 'radius':radius, 'units':units, 'lineWidth':lineWidth, 'lineColor':lineColor, 'lineColorSpace':lineColorSpace, 'fillColor':fillColor, 'fillColorSpace':fillColorSpace, 'vertices':vertices, 'closeShape':closeShape, 'pos':pos, 'size':size, 'ori':ori, 'opacity':opacity, 'contrast':contrast, 'depth':depth, 'interpolate':interpolate, 'name':name, 'autoLog':autoLog, 'autoDraw':autoDraw}
        super(PolygonComponent, self).__init__(start=start, stop=stop, **params)   

    def create(self,win):
        self.items.append(visual.Polygon(win=win, **self.params))

class RectComponent(StimulusComponent):
    def __init__(self, width=0.5, height=0.5, start=0, stop=1000000, autoLog=None, units='deg', lineWidth=1.5, lineColor='white', lineColorSpace='rgb', fillColor=None, fillColorSpace='rgb', pos=(0, 0), ori=0.0, opacity=1.0, contrast=1.0, depth=0, interpolate=True, name=None, autoDraw=False):
        params = {'width':width, 'height':height, 'autoLog':autoLog, 'units':units, 'lineWidth':lineWidth, 'lineColor':lineColor, 'lineColorSpace':lineColorSpace, 'fillColor':fillColor, 'fillColorSpace':fillColorSpace, 'pos':pos, 'size':[width,height], 'ori':ori, 'opacity':opacity, 'contrast':contrast, 'depth':depth, 'interpolate':interpolate, 'name':name, 'autoDraw':autoDraw}
        super(RectComponent, self).__init__(start=start, stop=stop, **params)

    def create(self,win):
        self.items.append(visual.Rect(win=win, **self.params))

class Routine(object):
    """Contains only StimulusComponents
       Represents a SCREEN of different stimuli
       Is associated with a filename for saving the screen
       Is associated with a keyboard response

    More documentation

    :Parameters:
        visible : **True** or False
            documentation on parameter
        newPos : **None** or [x,y]
            documentation on parameter
    """

    def __init__(self,components=[],timeout=10):
        self.components = components
        self.win = None
        self.timeout = timeout
        self.saveTrials = False
        self.viewport = None
        
    def addComponent(self,component):
        self.components.append(component)

    def update_levels(self,levels):
        pass
            
    def create(self,win,exp_handler=None):
        self.win = win
        self.exp_handler = exp_handler
        for component in self.components:
            if self.win is not None:
                component.create(win)

    def get_answer(self,trialparams={},loopstate={}):
        correct_answer = 'left'
        incorrect_answer = 'right'
        return correct_answer,incorrect_answer

    def get_filename(self,trialparams={},loopstate={}):
        #filename = 'stim-'+'-'.join(['{:02}'.format(y) for x,y in trialparams.items()])+'.png'
        filename = 'stim'
        for x,y in trialparams.items():
            if type(y) == np.float64 or type(y) == float:
                filename = filename + '-' + '{:5.5f}'.format(y)
            else:
                filename = filename + '-' + '{:02}'.format(y)
        #filename = 'stim-00-01-0.0-02-00'+'.png'
        return filename+'.png'

    def update_params(self,trialparams={},loopstate={}):
        pass
    
    #@profile
    def run(self,runmodel=None,genstim=False,trialparams={},params={},loopstate={}):
        self.update_params(trialparams,loopstate)
        resp = Response()
        filename = ''
        # If we have a window then either we are generating stimuli or
        #   running a human subject experiment
        if self.win is not None:
            for component in self.components:
                if genstim:                  
                    filename = self.get_filename(trialparams,loopstate)
                    component.start(trialparams,loopstate)
                else:
                    component.run(trialparams,loopstate)

            left,top = params['model']['view_pos']
            width,height = params['model']['view_size']

            ###################
            #diff_x,diff_y = params['diff_size']
            diff_x,diff_y = 1.0,1.0  # TODO: remove all this diff_size code
            width_window,height_window = self.win.size
            center_left = int(deg2pix(left,self.win.monitor))
            center_top = int(deg2pix(top,self.win.monitor))
            upper_left = center_left - int(0.5*width)
            upper_top = center_top + int(0.5*height)
            if width % 2 == 0:  # width is even number of pixels
                lower_right = upper_left + width
                lower_bottom = upper_top - height
            else:
                lower_right = upper_left + (width+1)
                lower_bottom = upper_top - (height+1)
            # convert to normalized coords
            upper_left_norm = upper_left / (0.5*width_window)
            upper_top_norm = upper_top / (0.5*height_window)
            lower_right_norm = lower_right / (0.5*width_window)
            lower_bottom_norm = lower_bottom / (0.5*height_window)
            rect = [upper_left_norm,upper_top_norm,lower_right_norm*diff_x,lower_bottom_norm*diff_y]
            extra = 2 # extra space to draw lines around rect
            if not self.viewport:
                self.viewport = visual.Rect(win=self.win,width=width+extra, height=height+extra, autoLog=None, units='pixels', lineWidth=1, lineColor='yellow', lineColorSpace='rgb', fillColor=None, fillColorSpace='rgb', pos=(center_left, center_top), ori=0.0, opacity=0.5, contrast=0.8, depth=1000, interpolate=False, name=None, autoDraw=True)
            else:
                self.viewport.setPos((center_left, center_top))

            self.win.flip()

            if genstim:
                model_stimulus = self.win._getFrame(rect=None,buffer='front')
                model_stimulus = np.array(model_stimulus)
                center_height = int(height_window/2)+center_top
                center_width = int(width_window/2)+center_left
                top_height = int(center_height - height/2)
                bottom_height = int(center_height + height/2)
                top_width = int(center_width - width/2)
                bottom_width = int(center_width + width/2)
                model_stimulus = model_stimulus[top_height:bottom_height,top_width:bottom_width,:]
                model_stimulus = Image.fromarray(model_stimulus)
                if np.shape(model_stimulus)[1] != width or np.shape(model_stimulus)[0] != height:
                    print(np.shape(model_stimulus))
                    import pdb; pdb.set_trace()
                    # TODO: Make it so this exception gets caught
                    logging.error('problem with model window')
                    raise Exception('Error in calculating shape of model window.')

                if 'model' in os.path.join(params['cwd'],'stimuli'+os.sep+filename):
                    import pdb; pdb.set_trace()

                logging.info('saving: '+filename)
                model_stimulus.save(os.path.join(params['cwd'],'stimuli'+os.sep+filename),"PNG")
                keys = []
            else:
                keys = event.waitKeys(maxWait=self.timeout, keyList=None)
                resp = Response(key=keys[0])
                for component in self.components:
                    component.end()
        else:
            filename = self.get_filename(trialparams,loopstate)
            # if we don't have a target_contrast in the parameters
            #  then we need to load an image and get its target_contrast
            if 'target_contrast' not in params['model']:
                target_key,target_val = params['target_identifier']
                target_condition = Params(trialparams)
                target_condition.update({target_key:target_val})
                target_filename = self.get_filename(target_condition,loopstate)

                # load stim from filename 
                im = Image.open(os.path.join(params['cwd'],'stimuli'+os.sep+target_filename)).convert('L')
                im_array = np.frombuffer(im.tobytes(), dtype=np.uint8)
                width, height = im.size
                im_array = im_array.reshape((height,width))
                contrast = runmodel.process(data=im_array)
                runmodel.target_contrast = contrast
                logging.exp('Generating target_contrast from '+target_filename)
                logging.exp('Using target_contrast = '+str(contrast))
                params['model']['target_contrast'] = contrast
                runmodel.update_decision_params()
                params['model']['decision_K'] = runmodel.decision_K
                params['model']['decision_sigma'] = runmodel.decision_sigma
                logging.exp('***Using decision_K = '+str(runmodel.decision_K))
                logging.exp('***Using decision_sigma = '+str(runmodel.decision_sigma))

            # load stim from filename define StimulusNotFound and raise it
            im = Image.open(os.path.join(params['cwd'],'stimuli'+os.sep+filename)).convert('L')
            im_array = np.frombuffer(im.tobytes(), dtype=np.uint8)
            width, height = im.size
            im_array = im_array.reshape((height,width))
            
            print('processing:', filename)
            save_conv_filename = None
            if params['saveimages']:
                #save_conv_filename = 'images/'+filename[:-4]+'-convolved.png'
                save_conv_filename = filename[:-4]
            correct_answer,incorrect_answer = self.get_answer(trialparams,loopstate)

            # if we are saving additional plots we also want to pass the
            #  corresponding target data to the runmodel.runsponse
            im_array_target = None
            if save_conv_filename:
                target_key,target_val = params['target_identifier']
                target_condition = Params(trialparams)
                target_condition.update({target_key:target_val})
                target_filename = self.get_filename(target_condition,loopstate)

                # load stim from filename 
                im_target = Image.open(os.path.join(params['cwd'],'stimuli'+os.sep+target_filename)).convert('L')
                im_array_target = np.frombuffer(im_target.tobytes(), dtype=np.uint8)
                width_target, height_target = im_target.size
                im_array_target = im_array_target.reshape((height_target,width_target))
                
            # TODO: if we are generating all the plots then read the target data
            #       from the file using the 'target_identifier' from params
            #       then pass into runmodel.response
            key,prob,contrast = runmodel.response(data=im_array, correct_answer=correct_answer, incorrect_answer=incorrect_answer,save_conv_filename=save_conv_filename,target_data=im_array_target)
            
            correct = False
            if key == correct_answer:
                correct = True
            resp = Response(key=key, rt=0, correct=correct, prob=prob, contrast=contrast)
            #keys = ['left']

        if resp.key == 'escape':
            core.quit()
            
        return resp

    def end(self):
        for component in self.components:
            component.end()


class ExperimentConditions(object):
    """ExperimentConditions

    More documentation

    :Parameters:
        visible : **True** or False
            documentation on parameter
        newPos : **None** or [x,y]
            documentation on parameter
    """

    def __init__(self,levels=[]):
        vals = list(itertools.product(*[levels[x] for x in levels.keys()]))
        self.trialList = [dict(zip(levels.keys(),val)) for val in vals]
        self.trials_df = pd.DataFrame(vals,columns=levels.keys())

class TrialRoutine(object):
    """Can contain a sequence of either Routines StairCaseTrialRoutines or TrialRoutine

    More documentation

    :Parameters:
        visible : **True** or False
            documentation on parameter
        newPos : **None** or [x,y]
            documentation on parameter
    """

    def __init__(self,conditions=[],nReps=1,subroutines=[],saveTrials=True):
        # exp.nTrialReps = 1
        # set up handler to look after randomization of conditions, etc.
        self.subroutines = subroutines
        self.saveTrials = saveTrials
        self.conditions = ExperimentConditions(conditions)
        self.levels = conditions
        self.exp_handler = None
        self.trials = data.TrialHandler(nReps=nReps,
                                        method='random', 
                                        originPath=-1,
                                        trialList=self.conditions.trialList,
                                        seed=None, name='trials')

    def update_levels(self,levels):
        levels.update(**self.levels)
        for routine in self.subroutines:
            routine.update_levels(levels)

    def create(self,win,exp_handler=None):
        self.win = win
        if not (win and exp_handler):
            self.trials = data.TrialHandler(nReps=1,
                                            method='sequential', 
                                            originPath=-1,
                                            trialList=self.conditions.trialList,
                                            seed=None, name='trials')
        if exp_handler:
            self.exp_handler = exp_handler
            self.exp_handler.addLoop(self.trials)
        for routine in self.subroutines:
            routine.create(win,exp_handler=exp_handler)

    def run(self,runmodel=None,genstim=False,trialparams={},params={},loopstate={}):
        for n,newtrialparams in enumerate(self.trials):
            logging.exp('Trial'+str(n))
            #print('Trial'+str(n))
            fullparams = Params(**newtrialparams)
            fullparams.update(**trialparams)
            loopstate.update({self.trials.name+'.thisRepN':self.trials.thisRepN})
            for subroutine in self.subroutines:
                #print(n,params)
                resp = subroutine.run(runmodel=runmodel,genstim=genstim,trialparams=fullparams,params=params,loopstate=loopstate)

            if resp is not None:
                self.trials.addData('key_resp.keys', resp.key)
                self.trials.addData('key_resp.corr', int(resp.correct))
                self.trials.addData('contrast', resp.contrast)
                self.trials.addData('decision_prob', resp.prob)
                if resp.level:
                    self.trials.addData('offset', resp.level)

                for name,value in fullparams.flatten():
                    self.trials.addData(name, value)

                for name,value in params.flatten():
                    self.trials.addData(name, value)

                if self.exp_handler:
                    logging.info('saving trial')
                    self.exp_handler.nextEntry()

    def end(self):
        for component in self.subroutines:
            component.end()

class StaircaseTrialRoutine(object):
    """ Can contain a sequence of either Routines StairCaseTrialRoutines or TrialRoutine

    More documentation

    :Parameters:
        visible : **True** or False
            documentation on parameter
        newPos : **None** or [x,y]
            documentation on parameter
    """

    def __init__(self,stair_params=[],level_values=None,subroutines=[],saveTrials=False):
        self.subroutines = subroutines
        self.saveTrials = saveTrials
        self.stair_params = stair_params
        self.trials = None
        self.level_values = level_values # array of values for the levels

    def update_levels(self,levels):
        for routine in self.subroutines:
            routine.update_levels(levels)

    def create(self,win,exp_handler=None): 
        self.win = win
        self.exp_handler = exp_handler
        #self.exp_handler.addLoop(self.trials)
        for routine in self.subroutines:
            routine.create(win,exp_handler=exp_handler)
                
    def run(self,runmodel=None,genstim=False,trialparams={},params={},loopstate={}):
        use_staircase = not genstim
        if use_staircase:
            self.trials = data.StairHandler(**self.stair_params)
            self.exp_handler.addLoop(self.trials)
        else:
            levels = dict()
            for routine in self.subroutines:
                routine.update_levels(levels)
            levels.update(levels)
            conditions = ExperimentConditions(levels)
            self.trials = data.TrialHandler(nReps=1,
                                            method='sequential', 
                                            originPath=-1,
                                            trialList=conditions.trialList,
                                            seed=None, name='trials')


        for n,level in enumerate(self.trials):
            fullparams = Params(trialparams)
            if use_staircase:
                # in the case of a normal run, level will be a single number
                fullparams.update(level=level)
            else:
                # in the case of genstim, level will be a dict of values
                fullparams.update(level)
            loopstate.update({'staircase.thisTrialN':n})
            for subroutine in self.subroutines:
                #logging.debug('stair:'+str(n)+' ' +str(level)+' '+str(trialparams))
                resp = subroutine.run(runmodel=runmodel,genstim=genstim,trialparams=fullparams,params=params,loopstate=loopstate)

            logging.exp('Staircase'+str(n))
            #print('Staircase'+str(n))

            # do not let staircase extend beyond the max
            if use_staircase and self.trials.thisTrialN == self.stair_params['maxVal']:
                self.trials.finished = True   
            
            if use_staircase and resp is not None:
                self.trials.addResponse(int(resp.correct))
                
            # TODO: add a flag to Trial and Staircase whether to save or not
            saveToDataFile = True
            if saveToDataFile and use_staircase and resp is not None:
                self.trials.addResponse(int(resp.correct))
                self.trials.addOtherData('key_resp.keys', resp.key)
                self.trials.addOtherData('key_resp.corr', int(resp.correct))
                self.trials.addOtherData('contrast', resp.contrast)
                self.trials.addOtherData('decision_prob', resp.prob)

                self.trials.addOtherData('offset', params['stimulus']['offsets'][level])

                for name,value in fullparams.flatten():
                    self.trials.addOtherData(name, str(value))

                for name,value in params.flatten():
                    self.trials.addOtherData(name, str(value))

                logging.info('saving staircase')
                self.exp_handler.nextEntry()
        if use_staircase: # store final level of staircase
            level_value = fullparams['level']
            if self.level_values is not None:
                resp.level = self.level_values[level_value]
            else:
                resp.level = fullparams['level']  

        return resp
                

    def end(self):
        for component in self.subroutines:
            component.end()

            
class Experiment(object):
    """Contains settings and routines
     call like this: ./herzog.py -model test=1,test2=\'0\' 
    More documentation

    :Parameters:
        visible : **True** or False
            documentation on parameter
        newPos : **None** or [x,y]
            documentation on parameter
    """

    def update_params(self,args,params):
        parser = argparse.ArgumentParser(description='Run experiment.')
        parser.add_argument('-viewing_distance', default=argparse.SUPPRESS, type=int,                            help='Viewing distance in cm.')
        #parser.add_argument('-viewing_distance', default=29.5*2.54, type=int,
        #                    help='Viewing distance in cm.')
        parser.add_argument('-genstim', default=False, action='store_true',
                            help='Saves all possible stimuli.')
        parser.add_argument('-runsubject', default=False, action='store_true',
                            help='Run a human subject.')
        parser.add_argument('-saveimages', default=False, action='store_true')
        parser.add_argument('-model', nargs=1, default=argparse.SUPPRESS, action='append') 
        updated_args = parser.parse_args(args[1:])
        if 'model' in updated_args.__dict__:
            new_model_params = updated_args.__dict__.pop('model',None)
            new_vals = new_model_params[0][0].split(',')
            new_model_params_dict = dict([(x[0],eval(x[1])) for x in [tuple(new_val.split('=')) for new_val in new_vals]])
            params['model'].update(new_model_params_dict)
        params.update(**updated_args.__dict__)
        logging.exp('Command line:'+' '.join(args))
        return params
    
    def __init__(self,params=None,reportobj=None,subroutines=[]):
        logging.LogFile(params['logfile'],level=logging.INFO,filemode='w')
        #logging.LogFile(level=logging.ERROR)

        logging.info('Using Python '+sys.version)

        self.args = sys.argv
        self.params = self.update_params(self.args,params)
        self.win = None
        self.model = None
        self.routines = subroutines
        self.thisExp = None
        self.name = params['name']
        self.cwd = params['cwd']
        self.reportobj = reportobj

        if params['monitor'] not in monitors.getAllMonitors():
            logging.error('monitor not found: '+params.monitor)
            logging.info('available monitors: '+' '.join(monitors.getAllMonitors()))
            logging.info('Define monitor in monitor center.')
            logging.info('To launch monitor center, type:\n'
                         'python -m psychopy.monitors.MonitorCenter')
            core.quit()
        else:
            self.monitor = monitors.Monitor('', width=None,
                                        distance=self.params['viewing_distance'],
                                        gamma=None, notes=None, useBits=None,
                                        verbose=True, currentCalib=None, autoLog=True)
            self.monitor.setSizePix((1920,1280))
            self.monitor.setWidth(54.15) 
            
            logging.exp('using monitor: '+self.params['monitor']+
                        ' with viewing_distance='+
                        str(self.params['viewing_distance'])+' cm\n'+
                        ' resolution (pixel/deg): '+str(deg2pix(1, self.monitor)))
            # TODO: change screen_pixel_size back to calculation from monitor
            # TODO: change the monitor definition width so that screen_pixel_size=0.282
            if 'screen_pixel_size' not in self.params['model']:
                self.params['model']['screen_pixel_size'] = pix2cm(1, self.monitor)*10.0

            #self.params['model']['screen_pixel_size'] = 0.282
            self.params['model']['viewing_distance'] = self.params['viewing_distance']/2.54
                 
        self.expInfo = params['exp_info']
        self.expInfo['date'] = data.getDateStr()

        # Ensure that relative paths start from the same directory as this script
        _thisDir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(_thisDir)
        # Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
        runsubject = self.params['runsubject']
        if runsubject:
            self.filename = self.cwd + os.sep + u'data'+os.sep+'%s_%s_%s' % (self.expInfo['participant'],params['name'], self.expInfo['date'])
        else:
            self.filename = self.cwd + os.sep + u'data'+os.sep+'model-' + params['expName']

       
    def run(self, runsubject=False, monitor=None, distance=None, store_runtime_info=False,display_report=True):
        try:
            # TODO: do this only if this a model run
            print('Removing old model data files.')
            logging.info('Removing old model data files from data directory.')
            filelist = glob.glob(os.path.join(self.cwd, 'data', 'model-'+ self.params['expName']+'.csv'))
            for f in filelist:
                print('Removing: '+ str(f))
                os.remove(f)   

            runtime_info = None
            runsubject = self.params['runsubject']
            if runsubject or self.params['genstim']:
                self.win = visual.Window(
                    size=self.monitor.getSizePix(), fullscr=False, screen=0, 
                    winType='pyglet', allowGUI=True, allowStencil=False,
                    #monitor=self.monitor, color='black', colorSpace='rgb',
                    # TODO: add color parameter for newlibrary: mainwindow
                    monitor=self.monitor, color= self.params['window_color'],
                    colorSpace='rgb', blendMode='avg', useFBO=True, autoLog=False)

                if store_runtime_info:
                    from psychopy import info # this library is slow to load, so we load here only since it is now needed
                    runtime_info = info.RunTimeInfo(author=None, version=None,
                                                    win=self.win, refreshTest='grating',
                                                    userProcsDetailed=False, verbose=True)

            else:
                self.model = Model(name=self.params['expName'],cwd=self.params['cwd'],saveimages=self.params['saveimages'],params=self.params,**self.params['model'])
                self.params['model']['decision_sigma'] = self.model.decision_sigma
                self.params['model']['decision_K'] = self.model.decision_K
                #logging.exp('Params:\n'+str(params))

            if not self.params['genstim']:
                # An ExperimentHandler isn't essential but helps with data saving
                self.thisExp = data.ExperimentHandler(name=self.params['name'],
                                                      version='',
                                                      extraInfo=self.expInfo,
                                                      runtimeInfo=runtime_info,
                                                      savePickle=False,
                                                      saveWideText=True,
                                                      dataFileName=self.filename,
                                                      autoLog=True)

            for routine in self.routines:
                routine.create(self.win,exp_handler=self.thisExp)

            for routine in self.routines:
                routine.run(runmodel=self.model,genstim=self.params['genstim'],params=self.params,loopstate=Params())

            if self.thisExp:
                self.thisExp.close()
                logging.exp('Experiment Finished.')
                logging.exp('Params used in experiment:\n'+str(self.params))
                logging.exp('Done.--------------------------------')              

            if display_report:
                self.displayreport()
                #core.quit()
                
        except Exception as e:
            logging.error(traceback.format_exc())
            self.end()
            core.quit()

    def displayreport(self):
        def sorted_ls(path):
            mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
            return list(sorted(os.listdir(path), key=mtime))[::-1]

        def reportmain(reportclass=None,filenames=None,only_most_recent=False,dirname=''):
            if filenames is None:
                filenames = sorted_ls('data')
                filenames = ['data'+ os.sep + fname for fname in filenames]

            r = reportclass(filenames=filenames,only_most_recent=only_most_recent,dirname=dirname)
            r.generate()

        if not self.params['genstim']:
            print('Generating report for data files.')
            logging.info('Generating report for data files.')
            filenames = glob.glob(os.path.join(self.cwd, 'data', 'model-'+ self.name+'*.csv'))
            reportmain(reportclass=self.reportobj,filenames=filenames,only_most_recent=False,dirname=self.params['cwd']+os.sep+'report')
        
    def end(self):
        for component in self.routines:
            component.end()


