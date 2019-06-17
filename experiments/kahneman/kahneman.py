#!/usr/bin/python
"""
=========================================================
Experiment:Kahneman
=========================================================
"""


import numpy as np
from scipy.ndimage.interpolation import rotate

from Contrast.framework import main
from Contrast.library import get_image_width_in_pixels, set_at


######## BEGIN: STIMULI #######################################################################
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
    
def build_stimulus(image_height=1000,image_width=1000,target_size=0.1,gap_size=0.1,flank_distance=3.0,viewing_distance=18.0,target_orientation=0,screen_pixel_size=0.282,**params):
    gap_size = int(np.round(get_image_width_in_pixels(gap_size,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)))
    flank_gap = int(np.round(get_image_width_in_pixels(flank_distance,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)))
    target_size = int(np.round(get_image_width_in_pixels(target_size,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)))

    if np.mod(target_size,2) == 1:  # if odd, make even
        target_size = target_size + 1

    target = get_landolt_c(ndim=target_size,radius=target_size/2,thickness=gap_size,angle=target_orientation,gap_thickness=gap_size)

    h_flank = np.ones((gap_size,target.shape[1]))
    v_flank = np.ones((target.shape[1],gap_size))

    center_x,center_y = image_height/2,image_width/2
    image = np.zeros((image_height,image_width))

    target_x,target_y = center_x-target.shape[0]/2,center_y-target.shape[1]/2
    # x is row, y is col    
    image = set_at(image,target,target_x,target_y)

    if flank_distance >= 0:
        # right flank
        image = set_at(image,v_flank,target_x,center_y+target_size/2+flank_gap)
        # left flank
        image = set_at(image,v_flank,target_x,center_y-(target_size/2+flank_gap+h_flank.shape[0]))
        # top flank
        image = set_at(image,h_flank,center_x-(target_size/2+flank_gap+h_flank.shape[0]),target_y)
        # bottom flank
        image = set_at(image,h_flank,center_x+target_size/2+flank_gap,target_y)

    image = np.abs(image-1.0)  # reverse the image to match the paper
    image2 = image*0.88   # contrast foreground to background is 88%
    return image2

######## END: STIMULI #######################################################################


# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(build_stimulus_func=build_stimulus)


