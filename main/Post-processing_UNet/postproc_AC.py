#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for helping deep learning to detect harder edges. 
Use nuclei centroids as seed points to perform level set region-growing 
to detect overlapped edges.

Created on Mon Aug 10 16:04:21 2020

@author: Calum
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, watershed, circle_level_set
import imutils
import cv2 # use for finding centroids of nuclei in bw image, and running graph cut algorithm

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store

def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    
    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    
    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
    
    """
    
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)
    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):
        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback

def find_nuclei_centroids(bw_image):
    """ Given a binary image of the detected nuclei, find the centroids of each nucleus """
    """ Will also need to remove non-nuclei objects with thresholding """
    nuc_contours = cv2.findContours(bw_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuc_contours = imutils.grab_contours(nuc_contours)
    centroids = []
    for c in nuc_contours:
        # compute the center of the contour using object moment(statistical properties) calculations
    	M = cv2.moments(c)
    	cX = int(M["m10"] / M["m00"])
    	cY = int(M["m01"] / M["m00"])
        centroids.append(cX)
    return centroids

def example_cells():
    """ 
    Example of using the morphological_chan_vese algorithm to detect overlapped 
    cell edges.
    
    ->  uses the nuclei centroid coordinates as initial seed points
    for circle_level_set, to initialise the algorithm and evolve towards
    the edge boundaries. 
    
    """
    # Load the specific test image 
    img = ()
    
    # Load binary image with detected nuclei
    nuc_bw = ()
    nuc_centroids = find_nuclei_centroids(nuc_bw)
    nuc_centroid = centroids[0] # pick centroid of one nucleus 
    
    # Initialise level set with this centroid.
    # not sure what to set for radius, but it should be small since we are 
    # performing region growing from nuclei centre.
    init_ls = circle_level_set(img.shape, center=nuc_centroid, radius=25) 
    
    # Callback for visual plotting
    callback = visual_callback_2d(img)
    
    # Run snakes to find edges. 
    # Will need to test different parameters for these, but run on default first.
    morphological_chan_vese(img, iterations=30, init_level_set=init_ls, 
                            smoothing=1, lambda1=1, lambda2=1, iter_callback=callback)
    

if __name__ == "__main__":
    
    example_cells()
    plt.show()
