#!/usr/bin/python
"""
=========================================================
Module: Framework
=========================================================

Contains functions for building and running an experiment
with parameters defined in the experiment.cfg file.

"""
import ConfigParser
import datetime
import itertools
import sys, getopt, os, glob, os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft2, ifft2, ifftshift
from scipy.spatial.distance import cdist

from experiments.library import compute_response,plot_decision_func,plot_data,plot_data_as_barplot
from library import get_pixels_at_degrees, get_sigma_map, sorted_ls, normalize, get_correct_coords
from stimulus import load_image, load_df, save_df, save_image
pd.options.mode.chained_assignment = None

#print(__doc__)

np.set_printoptions(suppress=True)
#matplotlib.use("Agg")

#------------------------------------------------------------------------------------------------------

#@profile


def get_op(sigma=0.01,K=1.0,field_height=100,field_width=100,pixel_offset=0,viewing_distance=12.0,screen_pixel_size=0.282):
    pixel_offset = np.round(get_pixels_at_degrees(degrees=pixel_offset,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size))
    coords = get_correct_coords(start_x=pixel_offset,viewing_distance=0,field_height=field_height,field_width=field_width,pixel_width=screen_pixel_size,pixel_height=screen_pixel_size)
    coords = coords / screen_pixel_size
    dist = cdist(np.matrix([[int(pixel_offset),0]]),coords)

    J = np.exp(-(np.square(dist)/(2.0*np.square(sigma*K))))

    J = J.reshape((field_height,field_width))
    J = J/np.sum(J)
    return J
    
def build_operators(offsets,viewing_distance=22.0,screen_pixel_size=0.282,field_height=100,field_width=100):
    """
    NOTE: Ji and Javg as filters should each sum to one
    """
    
    offsets = list(np.unique(np.array(offsets)))       
    operators = dict()

    for offset in offsets:
        sigma_offset = offset
        sigma_midget,sigma_parasol = get_sigma_map(start_x = offset,field_height=field_height,field_width=field_width,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
        sigma_midget = np.mean(sigma_midget[:,field_width/2:field_width]) 
        sigma_parasol = np.mean(sigma_parasol[:,field_width/2:field_width])

        K = 5.0
        sigma = sigma_midget
        print 'offset:', offset, 'sigma:', sigma, 'K:', K, 'sigma_parasol:', sigma*K

        Ji = get_op(pixel_offset=offset,sigma=sigma,K=1.0,field_height=field_height,field_width=field_width,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
        Javg = get_op(pixel_offset=offset,sigma=sigma,K=K,field_height=field_height,field_width=field_width,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
        Jtot = Ji - Javg
        J = fft2(Jtot)

        g = fft2(np.real(ifftshift(ifft2(np.multiply(J,J)))))
        if False:
            filename = 'op-'+'-'.join(['{:02}'.format(x) for x in [offset]])+'-identity'+'.png'
            save_image(normalize(Ji)*255,fname=filename,unnormalize=False)
            filename = 'op-'+'-'.join(['{:02}'.format(x) for x in [offset]])+'-average'+'.png'
            save_image(normalize(Javg)*255,fname=filename,unnormalize=False)
            filename = 'op-'+'-'.join(['{:02}'.format(x) for x in [offset]])+'-combined'+'.png'
            save_image(normalize(Jtot)*255,fname=filename,unnormalize=False)
        
        operators[offset] = (sigma,K,g)
    return operators

def process(data=None,pixel_offset=0,dataname=['Default'],operator=None,sigma=0.0012,k=4.6,image_height=100,image_width=100,experiment_name='',image_fname='',**config):
    raw_image = data

    op_field_height, op_field_width = operator.shape
    h_mid,w_mid = (op_field_height-image_height)/2,(op_field_width-image_width)/2
    g = operator[h_mid:op_field_height-h_mid,w_mid:op_field_width-w_mid]

    fft_raw_image = fft2(raw_image)
    g_image = np.multiply(fft_raw_image,g)
    g_image = np.real(ifftshift(ifft2(g_image)))

    rrms_contrast = np.sqrt(np.sum(np.multiply(raw_image,g_image)))
    rms_contrast = np.sqrt(np.sum(np.multiply(raw_image-np.mean(raw_image),raw_image-np.mean(raw_image))))
    print '  Results - RMS: %0.3f GRMS: %0.3f  ' % (rms_contrast, rrms_contrast)
    result = rms_contrast,rrms_contrast        
    return result

def get_params(process_params_func=None):
    params_changed = False
    oldconfig_fname = '.params'
    config_fname = glob.glob('*.cfg')[0] 
    experiment_name = config_fname.split('.')[0]
    config = ConfigParser.SafeConfigParser()
    config.read(config_fname)
    params = config.items(experiment_name)

    oldconfig = ConfigParser.SafeConfigParser()
    oldconfig.read(oldconfig_fname)
    oldparams = []
    try:
        oldparams = oldconfig.items(experiment_name)
    except ConfigParser.NoSectionError:
        pass
    if params <> oldparams:
        with open(oldconfig_fname,'wb') as f:
            config.write(f)
        params_changed = True

    params = dict([(x,eval(y)) for x,y in config.items(experiment_name)])

    params['config_fname'] = config_fname
    params['experiment_name'] = experiment_name
    params['params_changed'] = params_changed

    if process_params_func:
        params = process_params_func(params)

    return params

def format_as_str(value,fmt='{:02}'):
    if type(value) == str:
        return value
    else:
        return fmt.format(value)
        
# Builds the experiment as a dataframe
def build_experiment(experiment_name='',build_stimulus_func=None,model_params=[],experiment_params=[], stimulus_params=[], **params):
    trials = []

    vals=list(itertools.product(*[params[x] for x in experiment_params]))
    experiment_param_names =[x[:-1] for x in experiment_params]
    experiment_param_lists = [dict(zip(experiment_param_names,x)) for x in vals]
    
    vals=list(itertools.product(*[params[x] for x in stimulus_params]))
    stimulus_param_names =[x[:-1] for x in stimulus_params]
    stimulus_param_lists = [dict(zip(stimulus_param_names,x)) for x in vals]

    for stimulus_param_list in stimulus_param_lists:
        stim_params = dict(params)
        stim_params.update(stimulus_param_list)
        image = build_stimulus_func(**stim_params)
        image_height,image_width = image.shape
        filename = experiment_name+'_'+'_'.join([format_as_str(x) for x in [stimulus_param_list[x] for x in stimulus_param_names]])+'.png'
        print 'Saving: ', filename
        save_image(image,fname=filename,unnormalize=True)
        for experiment_param_list in experiment_param_lists:
            all_stim_params = dict(stim_params)
            all_stim_params.update(experiment_param_list)
            cols= [params[x] for x in model_params] + [stimulus_param_list[x] for x in stimulus_param_names] + [experiment_param_list[x] for x in experiment_param_names] + [filename] 
            trials.append(cols)
            
    columns = model_params+stimulus_param_names+experiment_param_names+['image'] 
    df = pd.DataFrame(trials,columns=columns)
    save_df(df,'conditions.xlsx','.')
    return df

def run_experiment(build_stimulus_func=None,experiment_name='experiment_name',config_fname='config.cfg',params_changed=True,build_experiment_func=build_experiment,**params):
    stim_fname = 'conditions.xlsx'
    if params_changed:
        print 'Experiment paramaters changed... rebuilding experiment...'
        filelist = glob.glob(os.path.join('images/', experiment_name+'*.png'))
        for f in filelist:
            print 'Removing: ', f
            os.remove(f)
        
        print 'Building experiment. Saving stimuli...',
        sys.stdout.flush()
        df = build_experiment_func(experiment_name=experiment_name,build_stimulus_func=build_stimulus_func,**params)
        print 'done.'
    else:
        df = load_df(stim_fname,'.')

    results = []
    session_date = datetime.datetime.now().strftime('%Y_%b_%d_%H%M')

    viewing_distance,screen_pixel_size,image_height,image_width=df[['viewing_distance','screen_pixel_size','image_height','image_width']].values[0]
    image_height,image_width = int(image_height),int(image_width)
    
    print 'Building operators...',
    operators = build_operators(offsets=df['offset'].values,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size,field_height=image_height,field_width=image_width)
    print 'done.'
    
    time1 = time.time()
    for index, row in df.iterrows():
        image_fname = row['image']
        offset = row['offset']
        pixel_offset = offset

        print 'Processing:', image_fname
        raw_image = load_image(dname='images/',fname=image_fname)

        image = np.multiply(raw_image,1.0/np.sqrt(image_height*image_width))

        sigma,K,operator = operators[pixel_offset]

        rms,rrms = process(data=image,pixel_offset=pixel_offset,operator=operator,dataname='Default',experiment_name=experiment_name,image_fname=image_fname,sigma=sigma,k=K,**row)
        results.append([sigma, K, rms, rrms, 0, 0, experiment_name, 0, session_date, 'model'])

    time2 = time.time()
    print 'Completed %03d images in %0.3f s, %0.3f FPS' % (index, (time2-time1), index/(time2-time1))
        
    columnnames = ['sigma','K','rms','rrms','resp.keys','resp.rt','expName','session','date','participant']
    result_df = pd.DataFrame(results,columns=columnnames)
    combined_df = df.merge(result_df,left_index=True,right_index=True)

    filename = '%s_%s.csv' % ('model', session_date)
    save_df(combined_df,filename,'data')
    return combined_df

def plot_main_result(result_df=None,experiment_name='pelli',subject='model',**params):
    columns_title='Eccentricity'
    if subject=='model':
        df = pd.pivot_table(result_df,columns=params['plot_columns'],index=params['plot_rows'],values=['rms','rrms'])
        has_sub_experiments = len(params['plot_rows'])>1
        if has_sub_experiments:
            name = df.index.names[0]
            experiments = np.unique(result_df[name].values).tolist()
        else:
            experiments = [experiment_name]

        exp_results = []    
        for exp_name in experiments:
            if has_sub_experiments:
                df_exp = df.loc[exp_name]['rrms']
                exp_name = format_as_str(exp_name,fmt='{:1}')
            else:
                df_exp = df['rrms']

            df_exp.iloc[0] = params['target_scaling_parameter']*df_exp.iloc[0]
            upper_limit = params['upper_limit_scaling_parameter']

            df_resp,mean_target,sigma_target,K = compute_response(df_exp,compute_error=params['compute_error'])
            exp_results.append(df_resp)

            if experiment_name == 'pelli':  # ensure target-alone condition is at the top of the Pelli graph
                df_resp.iloc[0] = upper_limit*100.0
                
            print 'Values for: ', exp_name
            print 'Contrast:'
            print df_exp
            print 'Contrast computed params: sigma,K'
            print result_df['sigma'].values[0],result_df['K'].values[0]

            print 'Decision:'
            print df_resp
            print 'Decision computed params: mean_target,sigma_target,K:'
            print mean_target,sigma_target,K
            print '============================================================================================='

            if has_sub_experiments:
                exp_title = experiment_name.title() + ':' + exp_name
            else:
                exp_title = experiment_name.title()

            if params['compute_error']:
                ylabel = 'Percent Error'
            else:
                ylabel = 'Percent Correct'

            plot_decision_func(df=df_resp/100.0,df_contrast=df_exp,fname='plot-'+exp_name+'-decision-func',target_contrast=mean_target,sigma=sigma_target,K=K,exp_name=exp_name,experiment_name=experiment_name,title='Decision function for '+exp_title,scale_range=[params['new_response_min'],params['new_response_max']],xlabel='Contrast',ylabel="Percent Correct",upper_limit=upper_limit,x_scale_factor=params['plot_x_scale_factor'],show_target_as_dash=params['plot_show_target_as_dash'],**params)
            
            plot_decision_func(df=df_resp/100.0,df_contrast=df_exp,fname='plot-'+exp_name+'-decision-func-zoomed',target_contrast=mean_target,sigma=sigma_target,K=K,exp_name=exp_name,experiment_name=experiment_name,title='Decision function for '+exp_title,scale_range=[params['new_response_min'],params['new_response_max']],zoomed=True,xlabel='Flank Distance',ylabel="Percent Correct",upper_limit=upper_limit,x_scale_factor=params['plot_x_scale_factor'],show_target_as_dash=params['plot_show_target_as_dash'],**params)
            
            plot_data(df_resp,fname='plot-'+exp_name,experiment_name=experiment_name,exp_name=exp_name,title=ylabel+' vs. Flank Distance for '+exp_title,xlabel=params['plot_data_xlabel'],ylabel=ylabel,x_scale_factor=params['plot_x_scale_factor'],x_scale_addition=params['plot_x_scale_addition'],show_target_as_dash=params['plot_show_target_as_dash'],**params) 
            plot_data(df_exp,fname='plot-'+exp_name+'-contrast-vs-flank_distance',experiment_name=experiment_name,exp_name=exp_name,title='Contrast vs. Flank Distance for '+exp_title,xlabel=params['plot_data_xlabel'],ylabel='Contrast',scale_plot=False,x_scale_factor=params['plot_x_scale_factor'],x_scale_addition=params['plot_x_scale_addition'],show_target_as_dash=params['plot_show_target_as_dash'],**params)

        if params['plot_data_as_barplot']:
            df_resp = pd.concat(exp_results,axis=1)
            df_resp.columns = pd.MultiIndex(levels=[[1, 5], [0, 1]],labels=[[0, 0, 1, 1], [0, 1, 0, 1]],names=[u'num_flank', u'gap'])
            plot_data_as_barplot(df_resp,fname='plot-'+experiment_name+'-barplot',experiment_name=experiment_name,exp_name=experiment_name,title='Perceptual error vs. Number of flankers for '+experiment_name,xlabel='Number of flankers',ylabel='Perceptual error',**params)
            
    else:
        pd.pivot_table(result_df,index='flank_distance', columns=['sd','offset'], values='resp.rt').plot(ax=fig.gca())


def plot_result(experiment_name='',plot_func=None,subject='model',only_most_recent=True,**params):
    print 'Generating report for: ', experiment_name
    fnames = []
    filenames = sorted_ls('data')
    
    for filename in filenames:
        if filename.startswith(subject) and filename.endswith('.csv'):
            fnames.append(filename)
            if only_most_recent:
                break

    result_df = pd.concat((pd.read_csv('data'+os.sep+f) for f in fnames))
    plot_func(result_df=result_df,experiment_name=experiment_name,subject='model',**params)

def main(build_stimulus_func,plot_func=plot_main_result,process_params_func=None,build_experiment_func=build_experiment):
    usage = 'Usage: experiment_name.py {run|report}'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "output="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print usage
        return

    if len(args) <> 1:
        print usage
        return

    params = get_params(process_params_func)
    
    if args[0] == "run":
        run_experiment(build_stimulus_func=build_stimulus_func,build_experiment_func=build_experiment_func,**params)
    elif args[0] == "report":
        plot_result(subject='model',plot_func=plot_func,only_most_recent=True,**params)
    else:
        print usage



