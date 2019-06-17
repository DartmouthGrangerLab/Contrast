#!/usr/bin/python
"""
=========================================================
Experiment:Pachai
=========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Contrast.framework import main
import Contrast.experiments.bex.bex as bex
import matplotlib
import os

#--------------------------------------------------------------------------------------------------------
    
def build_stimulus(line_width=0.0,gap_width=0.0,target_diameter=0.0,viewing_distance=22.0,screen_pixel_size=0.282,image_height=600,image_width = 600,flank_orientation=0,target_orientation=0,flank_distance=0.1,gap=1,num_flank=1,stimulus_contrast=0.1,**params):
    return bex.build_stimulus(line_width=line_width,gap_width=gap_width,target_diameter=target_diameter,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size,image_height=image_height,image_width = image_width,flank_orientation=flank_orientation,target_orientation=target_orientation,flank_distance=flank_distance,gap=gap,num_flank=num_flank,stimulus_contrast=stimulus_contrast,**params)

def process_params(params):
    return bex.process_params(params)

#--------------------------------------------------------------------------------------------------------
    
# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(build_stimulus_func=build_stimulus,process_params_func=process_params)
    


    
