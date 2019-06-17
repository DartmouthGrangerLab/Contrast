#!/usr/bin/python
"""
=========================================================
Experiment:Pelli
=========================================================
"""

import collections

import numpy as np
import pandas as pd

from Contrast.framework import main
from Contrast.library import get_image_width_in_pixels
from Contrast.stimulus import save_df, save_image, load_images


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def build_stimulus(unit_images = None,combinations = (1, 2, 1),image_height=600,image_width = 600):
    image = np.ones((image_height,image_width))
    center_x,center_y = image_height/2,image_width/2

    subimage = np.hstack([unit_images[i] for i in combinations])/255.0

    x,y = center_x-subimage.shape[0]/2, center_y-subimage.shape[1]/2
    image[x:x+subimage.shape[0],y:y+subimage.shape[1]] = subimage
    return image

def build_experiment(experiment_name='',displayname='Default',image_height=600,image_width=600,viewing_distance=12.0,screen_pixel_size=0.282,offset_max=850,offset_nsteps=40,build_stimulus_func=build_stimulus,flank_distances=[0],**params):
    offsets = params['offsets']

    if displayname == 'generated':
        stims = {'Generated': ([None,None,None,None],['blank', 'a', 'r', 'm'],[(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)])}
    else:
        stims = {'SmallLetters': (['stim-02-item-04.png','stim-02-item-01.png','stim-02-item-02.png','stim-02-item-03.png'],['blank', 'a', 'r', 'a'],[(1, 0, 2, 0, 1)]),
                 'LargeLetters': (['stim-03-item-04.png','stim-03-item-01.png','stim-03-item-02.png','stim-03-item-03.png'],['blank', 'a', 'r', 'a'],[(1, 0, 2, 0, 1)])}

    trials = []
    for name,details in stims.items():
        fnames,conditions,combinations = details
        if type(fnames[0]) == int:
            unit_images = []
            for freq in fnames:
                unit_images.append(get_test_image(freq=freq,field_height=image_height,field_width=image_width))
        else:
            unit_images = load_images(dname='images/',fnames = fnames[1:],normalize=False)

        stim_i = 0
        for image_combination in combinations:
            for flank_distance in flank_distances:
                if flank_distance >= 0.0:
                    blank_width = int(get_image_width_in_pixels(degrees=flank_distance,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size))
                    combination = image_combination
                else:
                    blank_width = unit_images[0].shape[1]
                    list_combination = list(image_combination)
                    list_combination[0] = 0
                    list_combination[-1] = 0
                    combination = tuple(list_combination)
                blank_height = unit_images[0].shape[0]
                blank_image = np.ones((blank_height,blank_width))*255
                image = build_stimulus(unit_images=[blank_image]+unit_images,combinations=combination,image_height=image_height,image_width=image_width)
                filename = name+'-'+'{:02d}'.format(stim_i)+'-'+str(flank_distance)+'.png'
                save_image(image,fname=filename,unnormalize=True)

                for offset in offsets:
                    cond = [conditions[x] for x in combination]
                    trial = list(flatten([name,offset,cond,flank_distance,stim_i,viewing_distance,screen_pixel_size,image_height,image_width,filename]))

                    trials.append(trial)
            stim_i = stim_i + 1

    columns = ['name','offset','left','blank','middle','blank','right','flank_distance','condition','viewing_distance','screen_pixel_size','image_height','image_width','image']
    df = pd.DataFrame(np.array(trials,dtype=np.object),columns=columns)
    save_df(df,'conditions.xlsx','.')
    return df

# mark functions to profile with @profile
# profile with: kernprof -l -v framework.py
if __name__ == "__main__":
    main(build_stimulus_func=build_stimulus,build_experiment_func=build_experiment)
    
