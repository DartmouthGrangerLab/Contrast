#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Stimulus Library
=========================================================
"""

#print(__doc__)

import numpy as np
import pandas as pd
import PIL.Image as img

## LOAD IMAGES #####################################################################

def load_image(dname='images/',fname='stim-01-item-01.png',normalize=True):
    fullname = dname+fname
    im = img.open(fullname).convert('L')
    a = np.array(list(im.getdata()))
    width, height = im.size
    a = a.reshape((height,width))

    #normalize = False
    #a = np.abs(a-255) # reverse image values

    if normalize:
        a = np.divide(np.array(a,dtype=np.float),255)
        #a = 2*a-1  # reverses pixel values so white=0 and black = 1
    return a

def load_images(dname='images/',fnames = ['stim-01-item-03.png','stim-01-item-01.png','stim-01-item-02.png'],normalize=True):
    images = []
    for fname in fnames:
        images.append(load_image(dname=dname,fname=fname,normalize=normalize))
    return images

def get_combined_image(unit_images_list=None,combinations=(1, 2, 1)):
    unit_images = np.array(unit_images_list)
    image = np.hstack(unit_images[combinations,:,:])
    return image

def get_combined_images(unit_images_list=None,combinations=[(1, 2, 1), (1, 1, 1),(0, 2, 1),(0, 1, 1)]):
    unit_images = np.array(unit_images_list)
    images = []
    for c in combinations:
        images.append(np.hstack(unit_images[c,:,:]))
    return images

def save_image(image, dname='images/',fname='stim-01-item-01.png',unnormalize=True):
    if unnormalize:
        #image = np.round(((image+1.0)/2.0)*255)
        image = np.round(image*255)
    fullname = dname+fname
    im = img.fromarray(image).convert('RGB')
    im.save(fullname,"PNG")
    
def normalize_images(images,rms=0.8):
    normalized_images = []
    for image in images:
        raw_img = image

        #raw_img = -1*raw_img + 255

        # we first need to convert it to the -1:+1 range
        #raw_img = (raw_img / 255.0) * 2.0 - 1.0
        #raw_img = np.abs((raw_img / 255.0)-1) * 2.0 - 1.0
        raw_img = raw_img - np.min(raw_img)
        raw_img = raw_img/float(np.max(raw_img))
        raw_img = raw_img*2.0 - 1.0

        raw_img = raw_img * -1.0

        # make the mean to be zero
        raw_img = raw_img - np.mean(raw_img)
        # make the standard deviation to be 1
        raw_img = raw_img / np.std(raw_img)
        # make the standard deviation to be the desired RMS
        raw_img = raw_img * rms

        #raw_img = (raw_img + 1.0) * 0.5
        #raw_img = raw_img - np.min(raw_img)
        #raw_img = raw_img/np.max(raw_img)
        #raw_img = raw_img*2.0 - 1.0

        normalized_images.append(raw_img)
    return normalized_images


###############################################3


def save_df(df,fname='data.xlsx',dname='report'):
    filename = dname+'/'+fname
    print 'Saving:', filename, '...',
    #df.to_msgpack(path_or_buf=filename) # doesn't work in all cases
    #df.to_pickle(path=filename,compression=None,protocol=0)
    #df.to_pickle(path=filename,compression='gzip',protocol=0)
    if fname[-3:] == 'csv':
        df.to_csv(filename)
    else:
        df.to_excel(filename)
    print 'done.'
    
def load_df(fname='data.xlsx',dname='report'):
    filename = dname+'/'+fname
    print 'Loading:', filename, '...',
    #df = pd.read_msgpack(path_or_buf=filename) # doen't work in all cases
    try:
        #df = pd.read_pickle(path=filename,compression=None)
        #df = pd.read_pickle(path=filename,compression='gzip')
        if fname[-3:] == 'csv':
            df = pd.read_csv(filename)
        else:
            df = pd.read_excel(filename)
        print 'done.'
    except IOError:
        df = None
        print 'not found.'
    return df

