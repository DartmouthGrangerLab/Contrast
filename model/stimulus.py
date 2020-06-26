#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
=========================================================
Stimulus Library
=========================================================
"""

import numpy as np
import pandas as pd
import PIL.Image as img
import matplotlib.pyplot as plt
from Contrast.model.library import normalize

def save_image(data,filename,xlabel='pixels',ylabel='pixels',title='title',overlay=None,reverse_overlay=False,valmax=False):
    fig = plt.figure(dpi = 600, tight_layout=True)
    if not valmax:
        valmax = np.max(np.abs(np.array([np.min(data),np.max(data)])))
    ysize,xsize = data.shape
    xmiddle,ymiddle = xsize/2,ysize/2
    #Ticks at even numbers, data centered at 0
    xticks=np.arange(-xmiddle,xmiddle+2,2)
    yticks=np.arange(-ymiddle,ymiddle+2,2)
    extent=(-xmiddle,xmiddle,-ymiddle,ymiddle)
    if overlay is not None:
        if reverse_overlay:
            overlay_cmap = 'Greys_r'
        else:
            overlay_cmap = 'Greys'
        plt.imshow(np.flipud(overlay),cmap=overlay_cmap,interpolation='none',origin='center',extent=extent)  # since we use origin=center we need to flip image see (https://stackoverflow.com/questions/56916638/invert-the-y-axis-of-an-image-without-flipping-the-image-upside-down)
        
    plt.imshow(np.flipud(data),vmin=-valmax,vmax=valmax,cmap='seismic',interpolation='none',origin='center',extent=extent,alpha=0.5) # since we use origin=center we need to flip image 
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(filename,dpi=600,quality=100)
    plt.close(fig)

