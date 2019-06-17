#!/usr/bin/python

"""
=========================================================
Experiment:Bex
=========================================================
"""

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import rotate
import os
from subprocess import call
import matplotlib.pyplot as plt

from Contrast.framework import main
from Contrast.library import normalize, set_at, sorted_ls, get_image_width_in_pixels, get_pixels_at_degrees
from Contrast.stimulus import save_df,save_image

def get_landolt_c(ndim=25,radius=9,thickness=1,angle=0.0,gap=1,gap_thickness=2):
    x = np.arange(ndim) # x values
    y = np.arange(ndim) # y values
    xx, yy = np.meshgrid(x, y) # combine both vectors

    center_x = x.shape[0]/2
    center_y = y.shape[0]/2
    dist = np.sqrt( (xx - center_x)**2 + (yy - center_y)**2 )
    
    outer_c = np.array(dist<radius,dtype=np.int)
    inner_c = np.array(dist<radius-thickness,dtype=np.int)
    c = outer_c - inner_c

    degrees = np.arctan2(np.array(yy-center_y,dtype=np.float),np.array(xx-center_x,dtype=np.float))*(180/np.pi)
    dd = np.vstack((np.flipud(degrees[center_x:,:]),degrees[center_x:,:]))    
    dd = rotate(dd,-(angle-90),reshape=False)

    if gap == 1:
        c[dd<gap_thickness*(ndim/radius)] = 0
        
    return c
    
def build_stimulus(line_width=0.0,gap_width=0.0,target_diameter=0.0,viewing_distance=22.0,screen_pixel_size=0.282,image_height=400,image_width = 400,flank_orientation=0,target_orientation=0,flank_distance=0.1,gap=1,num_flank=1,stimulus_contrast=0.1,**params):
    line_width = get_image_width_in_pixels(line_width,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
    gap_width = get_image_width_in_pixels(gap_width,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
    target_diameter = get_image_width_in_pixels(target_diameter,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)

    target_radius = target_diameter/2
    outer_angle = flank_orientation
    target_angle = target_orientation
    center_x,center_y = image_height/2,image_width/2
    image = np.ones((image_height,image_width))*0.5
    target_c = get_landolt_c(ndim=image_width,radius=target_radius,thickness=line_width,angle=target_angle)
    if flank_distance <> -1:
        flank_distance = get_pixels_at_degrees(flank_distance,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
        outer_radius = target_radius + flank_distance
        outer_c = get_landolt_c(ndim=image_width,radius=outer_radius,thickness=line_width,angle=outer_angle,gap=gap)
        image = set_at(image,outer_c,center_x-outer_c.shape[0]/2,center_y-outer_c.shape[1]/2)
        for i in range(1,num_flank):
            extra_flank_distance = get_pixels_at_degrees(0.9,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
            outer_c = get_landolt_c(ndim=image_width,radius=outer_radius+i*extra_flank_distance,thickness=line_width,angle=outer_angle,gap=gap)
            image = set_at(image,outer_c,center_x-outer_c.shape[0]/2,center_y-outer_c.shape[1]/2)

    image = set_at(image,target_c,center_x-target_c.shape[0]/2,center_y-target_c.shape[1]/2)

    return image

def process_params(params):
    viewing_distance = params['viewing_distance']
    screen_pixel_size = params['screen_pixel_size']
    offset_degrees = params['offsets']
    
    params['offset_pixels'] = list(get_pixels_at_degrees(np.array(offset_degrees),viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size))
    return params

# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(build_stimulus_func=build_stimulus,process_params_func=process_params)
    


    
