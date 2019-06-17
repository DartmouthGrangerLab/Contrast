# -*- coding: utf-8 -*-

"""
=========================================================
Framework Library
=========================================================

Contains functions that are independent of any specific experiment.
"""
# print(__doc__)

import os

import numpy as np
from scipy.spatial.distance import cdist


def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))[::-1]

def set_at(image=None, unit=None, x=0, y=0):
    """Sets the value of a pixel on image."""
    
    image[x:x+unit.shape[0],y:y+unit.shape[1]] = image[x:x+unit.shape[0],y:y+unit.shape[1]] + unit
    return image

def normalize(data):
    xdata = data - np.min(data)
    xdata = np.divide(xdata,np.max(xdata))
    return xdata

def get_correct_coords(start_x=0,viewing_distance=12.0,field_height=10,field_width=10,pixel_width=0.282,pixel_height=0.282,**config):
    """
    returns the coords in terms of degree of visual angle
    converts Euclidean to Polar coordinates based on a fixation point, viewing distance, and a window size
    polar coordinate conversion:
        
        - r = np.sqrt(np.square(x) + np.square(y))
        - th = np.arctan2(y,x)

    log-polar coordinate conversion based on degrees of visual angle from fixation:

        - r = np.rad2deg(np.arctan2(np.sqrt(np.square(x) + np.square(y)),viewing_distance*25.4))
    """
    
    x = (start_x + np.arange(np.ceil(-field_width/2.0),np.ceil(field_width/2.0),1))*pixel_width
    y = np.arange(np.ceil(-field_height/2.0),np.ceil(field_height/2.0),1)*pixel_height
    x,y = np.meshgrid(x,y)
    coords = np.vstack((x.ravel(),y.ravel())).T
    return coords


def get_viewing_distance_to_span_image(image_width=20,degree_span=1.0,screen_pixel_size=0.282):
    """
    degrees is viewing angle of the entire image
    image_width is size of entire in pixels
    """

    image_width_mm = image_width * screen_pixel_size
    inch_per_mm = 1.0/25.4   # 1 inch / 25.4 mm
    image_width_inches = image_width_mm * inch_per_mm
    distance_inches = (0.5*image_width_inches)/np.tan(np.radians(degree_span*0.5))
    return distance_inches

def get_image_width_in_degrees(image_width=100,viewing_distance=24.0,screen_pixel_size=0.282):
    """
    image_width is size of entire in pixels
    returns: degrees to span the entire image
    """
    
    mm_per_inch = 25.4
    degrees_per_image = np.degrees(np.arctan(((image_width*0.5)*screen_pixel_size)/(viewing_distance*mm_per_inch))*2.0)
    return degrees_per_image

def get_image_width_in_pixels(degrees=1.0,viewing_distance=24.0,screen_pixel_size=0.282):
    """
    degrees is viewing angle of the entire image
    returns: num of pixels that span the entire image
    """
    
    mm_per_inch = 25.4
    pixels = ((viewing_distance*mm_per_inch) * np.tan(np.radians(degrees*0.5))*2.0)/screen_pixel_size
    return pixels

def get_degrees_at_pixels(pixels=10,viewing_distance=24.0,screen_pixel_size=0.282):
    """
    pixels - if fovea is centered on an image, pixels is half the image width in pixels
    returns - half the viewing_angle
    """
    return  0.5*get_image_width_in_degrees(image_width=2.0*pixels,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)

def get_pixels_at_degrees(degrees=1.0,viewing_distance=24.0,screen_pixel_size=0.282):
    """
    degrees - if fovea is centered on an image, degrees is half the viewing angle
    returns: pixels - if fovea is centered on an image, pixels is half the image width in pixels
    """
    return 0.5*get_image_width_in_pixels(degrees=2.0*degrees,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size)
        

def get_sigma_map(start_x = 0,field_height=100,field_width=100,viewing_distance=12.0,screen_pixel_size=0.282,debug=False):
    """
    For each point on the image (image_height x image_width) returns the sigma associated
    with each point due to the offset from the fovea of the image.  The average of all the sigmas
    may be used as an approximation to the full set of all sigmas.  Each sigma is used as the basis
    for creating the J operator which is the weighting of all the pixels given one pixel as a focal point. 

    :param start_x: is in degrees of visual angle
    :returns: an entire field_height x field_width array of sigma values
    """
    start_x_pixels = np.round(get_pixels_at_degrees(degrees=start_x,viewing_distance=viewing_distance,screen_pixel_size=screen_pixel_size))
    optical_nodal_distance = 17.0 # mm from lens to fovea
    viewing_distance_inches = viewing_distance
    viewing_distance = viewing_distance * 25.4 # mm
    center_y, center_x = 0,0
    x_coords = (start_x_pixels + np.arange(-field_width/2.0,field_width/2,1))*screen_pixel_size
    y_coords = np.arange(-field_height/2.0,field_height/2,1)*screen_pixel_size
    x,y = np.meshgrid(x_coords,y_coords)
    coords = np.vstack((y.ravel(),x.ravel())).T

    image_dist = cdist(np.matrix([center_y,center_x]),coords)
    fovea_dist = (np.pi/180.0)*optical_nodal_distance*get_degrees_at_pixels(pixels=image_dist/screen_pixel_size,viewing_distance=viewing_distance_inches,screen_pixel_size=screen_pixel_size)
    midget_dendritic_field_diameter_micrometers = 8.64 * np.power(fovea_dist,1.04)  # midget from Dacey and Peterson, 1994
    midget_dendritic_field_diameter_millimeters = midget_dendritic_field_diameter_micrometers/1000.0
    midget_projected_field_diameter_on_image = get_pixels_at_degrees(degrees=start_x+np.degrees(np.arctan((midget_dendritic_field_diameter_millimeters/2.0)/optical_nodal_distance)),viewing_distance=viewing_distance_inches,screen_pixel_size=screen_pixel_size) - get_pixels_at_degrees(degrees=start_x-np.degrees(np.arctan((midget_dendritic_field_diameter_millimeters/2.0)/optical_nodal_distance)),viewing_distance=viewing_distance_inches,screen_pixel_size=screen_pixel_size)

    midget_sigma_map = midget_projected_field_diameter_on_image / 6.0  # ensures 99.7% of dendrites are connected to field diameter
    midget_sigma_map = midget_sigma_map.reshape((field_height,field_width))

    parasol_dendritic_field_diameter_micrometers = 70.2 * np.power(fovea_dist,0.65)  # parasol from Dacey and Peterson, 1994
    parasol_dendritic_field_diameter_millimeters = parasol_dendritic_field_diameter_micrometers/1000.0
    parasol_projected_field_diameter_on_image = get_pixels_at_degrees(degrees=start_x+np.degrees(np.arctan((parasol_dendritic_field_diameter_millimeters/2.0)/optical_nodal_distance)),viewing_distance=viewing_distance_inches,screen_pixel_size=screen_pixel_size) - get_pixels_at_degrees(degrees=start_x-np.degrees(np.arctan((parasol_dendritic_field_diameter_millimeters/2.0)/optical_nodal_distance)),viewing_distance=viewing_distance_inches,screen_pixel_size=screen_pixel_size)
    parasol_sigma_map = parasol_projected_field_diameter_on_image / 6.0  # ensures 99.7% of dendrites are connected to field diameter
    parasol_sigma_map = parasol_sigma_map.reshape((field_height,field_width))

    return midget_sigma_map,parasol_sigma_map


    
