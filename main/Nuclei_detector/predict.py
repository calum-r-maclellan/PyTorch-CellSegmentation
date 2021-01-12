# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:55:29 2019

@author: szb18149

For taking a new image, transforming into tensor, and sending through the pre-trained model.
The model output is then converted into a PIL image for displaying the edge prediction result.

Updates:
    02.03.2020:
        - 
        - 
"""

import argparse
import logging
import os
import time
import matplotlib as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data
from edgenet_resunits import Edge_Att_Unet_res
from models import Unet, Att_Unet, Edge_Att_Unet
from models_DS import Unet_DS, Att_Unet_DS, Edge_Att_Unet_DS
from utils import compute_average_stats
from ISBI14_dataset import TrainingDataset

# Create a root path to where we want the results to be stored (ie same folder as model_checkpoint folder)
root_dir = './Unet_experiments/Focal_loss/OneSided_FL'

# Now add to that root the folders where the results are kept
test_images_root = os.path.join(root_dir, 'input_images')
cyto_save_root   = os.path.join(root_dir, 'cyto_images')
nuclei_save_root = os.path.join(root_dir, 'nuclei_images')
if not os.path.exists(test_images_root): os.mkdir(test_images_root)
if not os.path.exists(cyto_save_root): os.mkdir(cyto_save_root)
if not os.path.exists(nuclei_save_root): os.mkdir(nuclei_save_root)

dice_log_root = os.path.join(root_dir, 'dice_log')
if not os.path.exists(dice_log_root): os.mkdir(dice_log_root)
LOG_Dice_cyto = open(os.path.join(dice_log_root, 'dice_cyto.txt'), 'w')
LOG_Dice_nuclei = open(os.path.join(dice_log_root, 'dice_nuclei.txt'), 'w')

prec_log_root = os.path.join(root_dir, 'prec_log')
if not os.path.exists(prec_log_root): os.mkdir(prec_log_root)
LOG_prec_cyto = open(os.path.join(prec_log_root, 'precision_cyto.txt'), 'w')
LOG_prec_nuclei = open(os.path.join(prec_log_root, 'precision_nuclei.txt'), 'w')

recall_log_root = os.path.join(root_dir, 'recall_log')
if not os.path.exists(recall_log_root): os.mkdir(recall_log_root)
LOG_recall_cyto = open(os.path.join(recall_log_root, 'recall_cyto.txt'), 'w')
LOG_recall_nuclei = open(os.path.join(recall_log_root, 'recall_nuclei.txt'), 'w')

tnr_log_root = os.path.join(root_dir, 'tnr_log')
if not os.path.exists(tnr_log_root): os.mkdir(tnr_log_root)
LOG_tnr_cyto = open(os.path.join(tnr_log_root, 'tnr_cyto.txt'), 'w')
LOG_tnr_nuclei = open(os.path.join(tnr_log_root, 'tnr_nuclei.txt'), 'w')

""" Cytoplasms results directories """
def log_cyto_dice(out_str):
    LOG_Dice_cyto.write(out_str+'\n')
    LOG_Dice_cyto.flush()
    
    
def log_cyto_prec(out_str):
    LOG_prec_cyto.write(out_str+'\n')
    LOG_prec_cyto.flush()

def log_cyto_recall(out_str):
    LOG_recall_cyto.write(out_str+'\n')
    LOG_recall_cyto.flush()
    
def log_cyto_tnr(out_str):
    LOG_tnr_cyto.write(out_str+'\n')
    LOG_tnr_cyto.flush()
    
""" Nuclei results directories """
def log_nuclei_dice(out_str):
    LOG_Dice_nuclei.write(out_str+'\n')
    LOG_Dice_nuclei.flush()
    
def log_nuclei_prec(out_str):
    LOG_prec_nuclei.write(out_str+'\n')
    LOG_prec_nuclei.flush()

def log_nuclei_recall(out_str):
    LOG_recall_nuclei.write(out_str+'\n')
    LOG_recall_nuclei.flush()
    
def log_nuclei_tnr(out_str):
    LOG_tnr_nuclei.write(out_str+'\n')
    LOG_tnr_nuclei.flush()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./cellnet_model.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask_threshold', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()



def predict_img(net,
                img,
                device,
                scale_factor,
                out_threshold,
                use_dense_crf=False):
    net.eval()
    img_height = img.size()[2]
    
    with torch.no_grad():
        
        img = img.to(device)                  # load data onto GPU
        
        """ Outputs """
        # Regular output
        output = net(img)               # send test_img through trained model
        
        # For Deep supervision models
#        _, _, output = net(img)   # ignore (_) edge outputs from net since not needed for inference, just training

        output = F.softmax(output, dim=1)     # .size() = [1, k=3, 512, 512]
        
        """ Transform output into image format """
        output_img = output.squeeze(0)            # .size() = [k=3, 512, 512]
        
        # Select the classes we want to see 
        cyto_edgemap = output_img[1, :, :]
        nuclei_edgemap = output_img[2, :, :]
        
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),        # convert tensor of shape [k, 512, 512] to a PIL image
                transforms.Resize(img_height),  
                transforms.ToTensor()
            ]
        )

        cyto_mask = tf(cyto_edgemap.cpu())
        cyto_mask = cyto_mask.squeeze().cpu().numpy()  # convert to numpy array: .shape = [512, 512]
        
        nuclei_mask = tf(nuclei_edgemap.cpu())
        nuclei_mask = nuclei_mask.squeeze().cpu().numpy()  # convert to numpy array: .shape = [512, 512]

    return output, cyto_mask >= out_threshold, nuclei_mask >= out_threshold   # make probs >= 0.5 all 1s, whilst others become 0s


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def raw2indices(outputs, targets, n_classes):
     
     # Outputs
     pred_indices = outputs.data.max(1)[1].cpu().numpy()   # get the class indices corresponding to confidence scores

     # Targets
     target_indices = targets.data.max(1)[1].cpu().numpy() # label each of pixels (262144) with class index
     
     return pred_indices, target_indices


if __name__ == "__main__":

    """ Directories for paths to test data."""
    images_dir  = "/home/hsijcr/calummac/Main_Dataset/test_dataset/test_images.mat"
    targets_dir = "/home/hsijcr/calummac/Main_Dataset/test_dataset/test_targets.mat"
    data_dirs = [images_dir, targets_dir]
    test_dataset = TrainingDataset(data_dirs)
    
    # Assign DataLoader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Load the model
    net = Unet(1, 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    model_path =  os.path.join(root_dir, 'model_checkpoint/cellnet_model_30.pth')
    net.load_state_dict(torch.load(model_path))
    
    logging.info("Model loaded with pre-trained parameters!")
    
    start_time = time.time()
    
    for i, (test_img, test_targets) in enumerate(test_dataloader):
        
        """ Call predict_img function to predict edgemap with trained model """
        outputs, cyto_mask, nuclei_mask = predict_img(net,
                                             test_img,
                                             device,
                                             scale_factor=1,
                                             out_threshold=0.5,
                                             use_dense_crf=False)
        
        
        """ Convert binary mask into an image """
        cyto_result = mask_to_image(cyto_mask)   # convert binary mask to .png image
        nuclei_result = mask_to_image(nuclei_mask)   # convert binary mask to .png image

        # Get target image for comparing with prediction
        cyto_target = test_targets.squeeze(0) # remove bs dimension
        cyto_target = cyto_target[1,:,:]  # extract cytoplasm edges
        cyto_target = cyto_target.cpu().numpy() # send to cpu and convert to numpy
        cyto_target = mask_to_image(cyto_target) # return binary gt map of cytoplasm edges
        
         # Get target image for comparing with prediction
        nuclei_target = test_targets.squeeze(0) # remove bs dimension
        nuclei_target = nuclei_target[2,:,:]  # extract cytoplasm edges
        nuclei_target = nuclei_target.cpu().numpy() # send to cpu and convert to numpy
        nuclei_target = mask_to_image(nuclei_target) # return binary gt map of cytoplasm ed
        
        # Save input image to folder
        test_img = test_img.squeeze().cpu().numpy()
        input_image = Image.fromarray( test_img.astype(np.uint8) )
        input_image.save('%s/img%d_input.png' % (test_images_root, i+1))
        
        # Save results/targets as .png images to folders  
        # Cytoplasm edges into ./cyto_results folder
        cyto_result.save('%s/img%d_result.png' % (cyto_save_root, i+1))
        cyto_target.save('%s/img%d_target.png' % (cyto_save_root, i+1))
        
        # Nuclei edges into ./nuclei_results folder
        nuclei_result.save('%s/img%d_result.png' % (nuclei_save_root, i+1))
        nuclei_target.save('%s/img%d_target.png' % (nuclei_save_root, i+1))
        
        
        """ Compute performance metrics """
        # First convert mask back into torch 
#        mask = torch.from_numpy(cyto_mask).float().unsqueeze(0)
#        target = test_targets[:, 1, :, :] # extract cyto_edge class
        
        # Now send to raw2indices for converting into 
        outputs, targets = raw2indices(outputs, test_targets, 3) # convert into indexed array for accuracy metrics
        
        # Compute statistics
        dice_list, precision_list, recall_list, tnr_list = compute_average_stats(outputs, targets, 3)

        """ Store cytoplasm results for image i in .txt file """
        log_cyto_dice( ('%f')   % (dice_list[0]) )
        log_cyto_prec( ('%f')   % (precision_list[0]) )
        log_cyto_recall( ('%f') % (recall_list[0]) )
        log_cyto_tnr( ('%f')    % (tnr_list[0]) )

        """ Store cytoplasm results for image i in .txt file """
        log_nuclei_dice( ('%f')   % (dice_list[1]) )
        log_nuclei_prec( ('%f')   % (precision_list[1]) )
        log_nuclei_recall( ('%f') % (recall_list[1]) )
        log_nuclei_tnr( ('%f')    % (tnr_list[1]) )
    
    # Record inference time (/image)
    print(" approx. inference time: %4.4f secs per image (of %d)" % ( (time.time() - start_time) / (i+1), i+1))
        
            
            
            
            