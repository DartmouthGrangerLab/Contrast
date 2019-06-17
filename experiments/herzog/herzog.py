#!/usr/bin/python
"""
=========================================================
Experiment:Herzog
=========================================================
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import rotate

from Contrast.framework import main
from Contrast.library import get_image_width_in_pixels, set_at


######## BEGIN: STIMULI #######################################################################

def get_line(bbox_height=100,bbox_width=100,col=0,row=0,height=40,width=40):
    bbox = np.zeros((bbox_height,bbox_width))
    bbox[row:row+height,col:col+width] = 1
    return bbox

def get_flank_shape(sd=50,box=0):
    vline_width = sd/5
    vline1 = get_line(bbox_height=sd,bbox_width=sd,col=0,row=0,height=sd,width=vline_width)==1
    midpoint = sd/2
    vline2 = get_line(bbox_height=sd,bbox_width=sd,row=0,height=sd,width=vline_width,col=midpoint-vline_width/2)==1
    vline3 = get_line(bbox_height=sd,bbox_width=sd,row=0,height=sd,width=vline_width,col=sd-vline_width)==1
    
    hline1 = get_line(bbox_height=sd,bbox_width=sd,col=0,row=0,height=vline_width,width=sd)==1
    hline2 = get_line(bbox_height=sd,bbox_width=sd,col=0,row=sd-vline_width,height=vline_width,width=sd)==1
    if box==1:
        flank = vline1 | vline3 | hline1 |  hline2
    else:
        flank = vline3
    return flank*1

def get_flank_shapes(sigma=30,box_flank=0):
    h_flank = get_flank_shape(sd=sigma,box=box_flank)
    v_flank = np.rot90(h_flank,2)
    return [h_flank,v_flank]

def get_target_shape(sd=50,orientation=0):
    vline_width = sd/5
    midpoint = sd/2
    vline1 = get_line(bbox_height=sd,bbox_width=sd,row=0,height=midpoint,width=vline_width,col=midpoint-vline_width)==1
    vline2 = get_line(bbox_height=sd,bbox_width=sd,row=midpoint,height=midpoint,width=vline_width,col=midpoint)==1
    
    target = vline1 | vline2
    if orientation==1:
        target = np.fliplr(target*1)

    return target
    
def build_stimulus(image_height=500,image_width=500,target_size=0.1,flank_distance=3.0,viewing_distance=18.0,target_orientation=0,box_flank=0,num_flank=1,screen_pixel_size=0.282,**params):
    target_size = int(get_image_width_in_pixels(target_size,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size))

    if np.mod(target_size,2) == 1:  # if odd, make even
        target_size = target_size + 1
    h_flank,v_flank = get_flank_shapes(sigma=target_size,box_flank=box_flank)
    target = get_target_shape(sd=target_size,orientation=target_orientation)
    center_x,center_y = image_height/2,image_width/2
    image = np.zeros((image_height,image_width))
    
    image = set_at(image,target,center_x-target.shape[0]/2,center_y-target.shape[1]/2)

    if flank_distance >= 0:
        flank_gap = int(get_image_width_in_pixels(flank_distance,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size))
        image = set_at(image,v_flank,center_x-target.shape[0]/2,center_y+flank_gap)
        image = set_at(image,h_flank,center_x-target.shape[0]/2,center_y-h_flank.shape[1]-flank_gap)

        for i in range(2,num_flank+1):
            n_flank_gap = int(3)
            image = set_at(image,v_flank,center_x-target.shape[0]/2,center_y+flank_gap+(i-1)*(h_flank.shape[1]+n_flank_gap))
            image = set_at(image,h_flank,center_x-target.shape[0]/2,center_y-h_flank.shape[1]-flank_gap-(i-1)*(h_flank.shape[1]+n_flank_gap))
    return image


######## END: STIMULI #######################################################################

     
# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(build_stimulus_func=build_stimulus)


