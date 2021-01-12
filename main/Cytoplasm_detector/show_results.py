# -*- coding: utf-8 -*-
"""

Script for showing the segmented classes on top of the input image.
Read in images and predicted classes from separate folders, then overlay predictions
onto images. 

Colour them for easy visuals.

Started: 03.03.2020.

Updates:
    04.03.2020:
        - got it working for 1 image; next steps are to iterate over the entire test set

@author: Calmac
"""

from PIL import Image, ImageFilter
import numpy as np
import os

def main():
    
    root_dir = './Unet_experiments/Focal_loss/OneSided_FL'
    results_root = os.path.join(root_dir, 'edge_results_images')
    if not os.path.exists(results_root): os.mkdir(results_root)
    
    for i in range(0, 90):
        
        # Load main image - desaturate and revert to RGB so we can draw on it in colour
        main = Image.open(os.path.join(root_dir, ('input_images/img%d_input.png' % (i+1) ))).convert('L').convert('RGB')
        
        """ Call drawContour() to overlay colour onto input image (main) """
        main_pred = np.array(main)
        cyto_seg = Image.open(os.path.join(root_dir, ('cyto_images/img%d_result.png' % (i+1) ))).convert('L')
        nuclei_seg = Image.open(os.path.join(root_dir, ('nuclei_images/img%d_result.png' % (i+1) ))).convert('L')
        pred_edges = draw_edges(main_pred, cyto_seg,   (255,255,0)) # draw cytoplasms in yellow
        pred_edges = draw_edges(main_pred, nuclei_seg, (255,0,0))   # draw nuclei in red
        
        # Save result
        pred_result = Image.fromarray(pred_edges)
        pred_result.save('%s/img%d_pred_edges.png' % (results_root, i+1) )
        
        """Repeat for ground truths """
        main_gt = np.array(main)
        cyto_GT = Image.open(os.path.join(root_dir, ('cyto_images/img%d_target.png' % (i+1) ))).convert('L')
        nuclei_GT = Image.open(os.path.join(root_dir, ('nuclei_images/img%d_target.png' % (i+1) ))).convert('L')
        gt_edges = draw_edges(main_gt, cyto_GT,   (255,255,0)) # draw cytoplasms in yellow
        gt_edges = draw_edges(main_gt, nuclei_GT, (255,0,0))   # draw nuclei in red
        
        # Save result
        gt_result = Image.fromarray(gt_edges)
        gt_result.save('%s/img%d_gt_edges.png' % (results_root, i+1) )


def draw_edges(m, s, RGB):
    
    """Draw edges of contours from binary image 's' onto 'm' in colour 'RGB'"""
    # Find edge coordinates of the contour and store into Numpy array
    thisEdges   = s.filter(ImageFilter.FIND_EDGES)
    thisEdgesN  = np.array(thisEdges)

    # Paint locations of found edges in colour "RGB" onto "main"
    m[np.nonzero(thisEdgesN)] = RGB
    return m

if __name__ == '__main__':
    main()
    
    
    
    
    