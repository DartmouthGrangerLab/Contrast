#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================
Model
=========================================================

Contains Model Class for building Jacobian operators and processing an image.

"""
from past.builtins import basestring
from builtins import str
from builtins import object

import datetime
import itertools
import sys, getopt, os, glob, os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft2, ifft2, ifftshift
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.optimize import curve_fit

import random
import re
import datetime
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    import Image
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from Contrast.model.library import get_pixels_at_degrees, get_sigma_map, sorted_ls, normalize, get_correct_coords
from Contrast.model.stimulus import save_image
pd.options.mode.chained_assignment = None

#print(__doc__)

np.set_printoptions(suppress=True)
#matplotlib.use("Agg")

#########################################

def decision_func(contrast,target_contrast=0,decision_sigma=0,decision_K=0,upper_limit=0,lower_limit=0,return_all=False):
    """Decision function that maps contrast values to a proportion of correct response.

    :Parameters:
        contrast : a numpy array or single contrast value
        target_contrast : the contrast value of the target_alone, or mu_t
        decision_sigma : the standard deviation of the target_alone, or sigma_tau
        decision_K : the value K_phi, so sigma_phi = K_phi*sigma_tau
        upper_limit : the upper limit of proportion correct resp e.g. 0.85 
        lower_limit : the lower limit of proportion correct resp e.g. 0.0
        return_all : if True, returns result and foreground and background Jacobians

    :returns: a numpy array of proportion correct responses, one for each contrast value in the contrast parameter.

    """        
    J_f = np.exp(-(np.square(contrast-target_contrast)/(2.0*np.square(decision_sigma))))
    J_b = np.exp(-(np.square(contrast-target_contrast)/(2.0*np.square(decision_sigma*decision_K))))
    result = (J_f-J_b+1)*(upper_limit-lower_limit)+lower_limit
    if return_all:
        return result,J_f,J_b
    else:
        return result

class Model(object):
    """Class for building Jacobians and processing an image


    """        

    def __init__(self,
                 eccentricities=[3.88],
                 viewing_distance=29.5,
                 screen_pixel_size=0.282,
                 view_size=(350,350),
                 view_pos=(0,0), 
                 target_contrast=0.0,
                 est_max=0.01,
                 K=5.0,
                 upper_limit=0.85,
                 lower_limit=0.0,
                 saveimages = False,
                 cwd = '',
                 name='',
                 logging=None,**params):
        super(Model, self).__init__()
        self.logging = logging
        self.view_pos = view_pos
        self.view_size = view_size
        self.eccentricity = eccentricities[0]
        self.eccentricities = eccentricities
        self.target_contrast = target_contrast
        self.est_max = est_max
        self.K = K
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.saveimages = saveimages
        self.cwd = cwd
        self.name = name
        self.params = params
        self.operators = self.build_operators(eccentricities=eccentricities,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size,field_height=view_size[0],field_width=view_size[1])
        self.decision_sigma, self.decision_K = self.get_decision_params()

    def get_decision_params(self):
        sigma_target = np.sqrt((-np.square(1.1*self.target_contrast-self.target_contrast))/(2*np.log(0.01)))       
        sigma_target_and_distractor = np.sqrt((-np.square(self.est_max-self.target_contrast))/(2*np.log(0.01)))  # 0.045 matches previous K
        K = sigma_target_and_distractor/sigma_target
        return sigma_target,K  # decision_sigma, decision_K

    def update_decision_params(self):
        self.decision_sigma, self.decision_K = self.get_decision_params()
        
    def compute_decision(self,contrast,sigma=1.0,compute_error=False,est_max=-1.0,compute_relative_to_chance=False,chance=0.0,update=False):
        self.decision_sigma, self.decision_K = self.get_decision_params()
        resp = decision_func(contrast,target_contrast=self.target_contrast,decision_sigma=self.decision_sigma,decision_K=self.decision_K,upper_limit=self.upper_limit,lower_limit=self.lower_limit)

        if compute_relative_to_chance:
            resp = ((contrast-chance) / (self.upper_limit - chance)) * self.upper_limit

        if compute_error:
            resp = 1 - resp

        return resp      

    def response(self,data, name='', correct_answer='left', incorrect_answer='right',save_conv_filename=None,target_data=None):
        contrast = self.process(data, name=name, save_conv_filename=save_conv_filename,target_data=target_data)
        decision = self.compute_decision(contrast)
        if np.random.random() < decision:
            resp = correct_answer
        else:
            resp = incorrect_answer
        return resp, decision, contrast
        
    #@profile
    def get_op(self,sigma=0.01,K=1.0,field_height=100,field_width=100,pixel_eccentricity=0,viewing_distance=12.0,screen_pixel_size=0.282):
        pixel_eccentricity = np.round(get_pixels_at_degrees(degrees=pixel_eccentricity,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size))
        coords = get_correct_coords(start_x=pixel_eccentricity,viewing_distance=0,field_height=field_height,field_width=field_width,pixel_width=screen_pixel_size,pixel_height=screen_pixel_size)
        coords = coords / screen_pixel_size
        dist = cdist(np.matrix([[int(pixel_eccentricity),0]]),coords)

        J = np.exp(-(np.square(dist)/(2.0*np.square(sigma*K))))

        J = J.reshape((field_height,field_width))
        J = J/np.sum(J)
        return J

    def build_operators(self,eccentricities,viewing_distance=22.0,screen_pixel_size=0.282,field_height=100,field_width=100):
        """
        NOTE: Ji and Javg as filters should each sum to one
        """
        print('Building operators...',)
        eccentricities = list(np.unique(np.array(eccentricities)))       
        operators = dict()

        for eccentricity in eccentricities:
            sigma_eccentricity = eccentricity

            sigma_midget,sigma_parasol = get_sigma_map(start_x = eccentricity,field_height=field_height,field_width=field_width,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)

            sigma_midget = np.mean(sigma_midget[:,int(field_width/2):field_width]) 

            sigma = sigma_midget
            print('eccentricity:', eccentricity, 'contrast_sigma:', sigma, 'contrast_K:', self.K)

            Ji = self.get_op(pixel_eccentricity=eccentricity,sigma=sigma,K=1.0,field_height=field_height,field_width=field_width,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
            Javg = self.get_op(pixel_eccentricity=eccentricity,sigma=sigma,K=self.K,field_height=field_height,field_width=field_width,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
            Jtot = Ji - Javg
            J = fft2(Jtot)

            g = fft2(np.real(ifftshift(ifft2(np.multiply(J,J)))))
            if self.saveimages and False:
                filename = self.cwd + '/images/op-'+'-'.join(['{:02}'.format(x) for x in [eccentricity]])+'-identity'+'.png'
                save_image(Ji,filename,xlabel='pixels',ylabel='pixels',title='J foreground')
                
                filename = self.cwd + '/images/op-'+'-'.join(['{:02}'.format(x) for x in [eccentricity]])+'-average'+'.png'
                save_image(Javg,filename,xlabel='pixels',ylabel='pixels',title='J background')
                filename = self.cwd + '/images/op-'+'-'.join(['{:02}'.format(x) for x in [eccentricity]])+'-combined'+'.png'
                save_image(Jtot,filename,xlabel='pixels',ylabel='pixels',title='J foreground-background')
            operators[eccentricity] = (sigma,self.K,g,Ji,Javg,Jtot)

        print('done.')
        if self.saveimages and False:
            fig = plt.figure()
            ax = plt.gca()
            width = 100
            x = np.linspace(-width/2,width/2,width+1)
            for eccentricity in self.eccentricities:
                op = operators[eccentricity][5][int(field_height/2)]
                op = op[int(field_width/2)-int(width/2):int(field_width/2)+int(width/2)+1]
                plt.plot(x,op,label=eccentricity)
            plt.legend(title='Eccentricity')
            #plt.xticks(xi, x)
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Activation')
            ax.set_title('Activation spread of Jacobian (foreground-background)')
            plt.savefig(self.cwd + '/images/op-plot-'+self.name+'.png')
            plt.close(fig)
            
        return operators

    def process(self,data=None,name='',save_conv_filename=None,target_data=None):
        if np.max(data) > 1.0:
            data = np.divide(np.array(data,dtype=np.float),255)

        image_height,image_width = self.view_size
        pixel_eccentricity = self.eccentricity
        sigma,K,operator,Ji,Javg,Jtot = self.operators[pixel_eccentricity]

        image = np.multiply(data,1.0/np.sqrt(image_height*image_width))
        raw_image = image

        op_field_height, op_field_width = operator.shape
        g = operator

        fft_raw_image = fft2(raw_image)
        #import pdb; pdb.set_trace()

        g_image = np.multiply(fft_raw_image,g)
        g_image = np.real(ifftshift(ifft2(g_image)))

        #save_conv_filename = False
        if save_conv_filename is not None:
            filename = save_conv_filename
            background_intensity = data[0,0]

            # process target so we can subtract it from the data image
            if np.max(target_data) > 1.0:
                target_data = np.divide(np.array(target_data,dtype=np.float),255)

            target_image = np.multiply(target_data,1.0/np.sqrt(image_height*image_width))

            target_raw_image = target_image

            target_fft_raw_image = fft2(target_raw_image)
            target_g_image = np.multiply(target_fft_raw_image,g)
            target_g_image = np.real(ifftshift(ifft2(target_g_image)))
            intersection = np.multiply(target_g_image,(g_image-target_g_image))
            img_data = intersection

            if False:
                fig = plt.figure()
                plt.plot(img_data[int(image_height/2)])
                ax = plt.gca()
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Activation')
                ax.set_title('Activation on intersection\n between target and flank on image')
                plt.savefig(self.cwd + '/images/plot-intersection-'+filename+'.png')
                plt.close(fig)
                save_image(img_data,self.cwd + '/images/intersection-'+filename+'.png',xlabel='Pixels',ylabel='Pixels',title='Activation on intersection\n between target and flank on image')


            if False:
                fig = plt.figure()
                plt.plot(g_image[int(image_height/2)],label='image convoled') 
                plt.plot(raw_image[int(image_height/2)],label='image')
                plt.plot(np.sqrt(image_width*image_height)*np.multiply(raw_image,g_image)[int(image_height/2)],label='result')
                plt.legend(title='Legend')
                ax = plt.gca()
                ax.set_xlabel('Pixels')
                ax.set_ylabel('Activation')
                ax.set_title('Activation on cross-section\n of image')
                plt.savefig(self.cwd + '/images/plot-result-'+filename+'.png')
                plt.close(fig)

            if True:
                #img_data = normalize(g_image)
                img_data = g_image
                if False:
                    fig = plt.figure()
                    plt.plot(img_data[int(image_height/2)])
                    ax = plt.gca()
                    ax.set_xlabel('Pixels')
                    ax.set_ylabel('Activation')
                    ax.set_title('Activation on cross-section\n of image')
                    plt.savefig(self.cwd+'/images/plot-convolved-'+filename+'.png')
                    plt.close(fig)
                #import pdb; pdb.set_trace()

                save_image(img_data,self.cwd+'/images/convolved-'+filename+'.png',xlabel='Pixels',ylabel='Pixels',title='Activation on cross-section\n of image',overlay=raw_image,reverse_overlay=self.params['params']['window_color'][0]>0,valmax=0.0002) #Herzpg: 0.00005

        rrms_contrast = np.sqrt(np.sum(np.multiply(raw_image,g_image)))
        rms_contrast = np.sqrt(np.sum(np.multiply(raw_image-np.mean(raw_image),raw_image-np.mean(raw_image))))
        print('  Results - RMS: %0.3f GRMS: %0.3f  ' % (rms_contrast, rrms_contrast))
        result = rrms_contrast
        return result


   
# CALL WITH:
# ./framework.py ../herzog/screenshots/Herzog2012*.png
# OR
# ./framework.py ../herzog/screenshots/Herzog2012_expnum\=1_items\=0_jitter\=0_flank_target_height_ratio\=0.5_num_flank\=0_eccentricity\=16.66_orientation\=01.png

# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(sys.argv[1:])
    
