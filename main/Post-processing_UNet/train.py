#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training the post-processing edge-correction module. 
Try a basic Unet first.

Data: 10_cells_edge_data from before, to give a pre-trained network.
                                                
@author: Calum
Date created: 16/3/2020.

Updates:
    30.03.2020:
        - first code to try linking output1 (SegNet) to input of CorrNet

---------------------------------
---------------------------------

"""


from datetime import datetime
import argparse
import os
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from dataset import TrainingDataset
from models import init_weights, Unet_Seg, Unet_Corr
from loss import DiceLoss, 
from torch.autograd import Variable


# Create log for storing training and validation loss and accuracy results.
# NB: train results are odd, validation are even!!!
train_log_root = "./train_log"
if not os.path.exists(train_log_root): os.mkdir(train_log_root)
LOG_Overall = open(os.path.join(train_log_root, 'train.txt'), 'w')
LOG_Loss = open(os.path.join(train_log_root, 'LossPerEpoch.txt'), 'w')

# Create folder to store the best model
os.system('mkdir {0}'.format('model_checkpoint'))

def log_string_overall(out_str):
    # Overall training results
    LOG_Overall.write(out_str+'\n')
    LOG_Overall.flush()

def log_string_loss(out_str):  
    # training
    LOG_Loss.write(out_str+'\n')
    LOG_Loss.flush()
    
parser = argparse.ArgumentParser(description = '2D u-net')
parser.add_argument('--model',      default='UNet', type=str, help='choose a type of model')
parser.add_argument('--lr',         default=1e-4, type=float, help='learning rate')
parser.add_argument('--momentum',   default=0.999, type=float, help='momentum in optimizer')
parser.add_argument('--batch_size', default=1,  type=int, help='batch size') # choose batch_size of 1 initially
parser.add_argument('--epochs',     default=30, type=int, help='epochs to train')
parser.add_argument('--out1',       default='./SegNet_checkpoints', type=str, help='path to save model checkpoints')
parser.add_argument('--out2',       default='./CorrNet_checkpoints', type=str, help='path to save model checkpoints')
parser.add_argument('--lr-mode',    default='step', type=str)
parser.add_argument('--step',       default=20, type=int)

# Hyperparameters (UPDATED: 27.01.20)
args = parser.parse_args()
batch_size = args.batch_size 
num_epochs = args.epochs   
in_channels = 1                 
n_classes = 2   # 0=background, 1=cytoplasm edges

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

"""Directories for paths to data."""
images_dir  = "/home/hsijcr/calummac/10_cells_edge_data/no_deep_supervision/training_images_shuf.mat"
targets_dir = "/home/hsijcr/calummac/10_cells_edge_data/no_deep_supervision/training_targets_shuf.mat"
data_dirs = [images_dir, targets_dir]
     

if __name__ == '__main__':
    
    """
    Datasets
    """
    # Get the full set of data
    dataset = TrainingDataset(data_dirs)
     
    # Assign DataLoader to data
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

    """ 
    Build the segmentation network: Unet_Seg
    """
    # Initialise
    model_seg = Unet_Seg(in_channels, n_classes) # Load the model and assign input and output channels
    model_seg.to(device)                     # send to GPU/CPU
    model_seg.apply(init_weights)            # apply normal distribution to initialise weights
    model_seg.train()                        # set to training mode
    
    # Count number of parameters
    segnet_total_params = sum(p.numel() for p in model_seg.parameters())
    print('Total number of parameters in Unet_Seg: {:4.2f} M'.format(segnet_total_params / 1e6))
    
    # Assign optimiser and learning rate scheduler
    optimizer_seg = optim.Adam(model_seg.parameters(), lr=args.lr, weight_decay = 0)
    scheduler_seg = optim.lr_scheduler.MultiStepLR(optimizer_seg, milestones = [10, 20], gamma=0.1)
    
    """ 
    Build the correction network: Unet_Corr
    """
    # Initialise
    model_corr = Unet_Corr(in_channels, n_classes) # Load the model and assign input and output channels
    model_corr.to(device)                     # send to GPU/CPU
    model_corr.apply(init_weights)            # apply normal distribution to initialise weights
    model_corr.train()                        # set to training mode
    
    # Count number of parameters
    corrnet_total_params = sum(p.numel() for p in model_corr.parameters())
    print('Total number of parameters in Unet_Corr: {:4.2f} M'.format(corrnet_total_params / 1e6))
    
    # Assign optimiser and learning rate scheduler
    optimizer_corr = optim.Adam(model_corr.parameters(), lr=1e-3, weight_decay = 0)
    scheduler_corr = optim.lr_scheduler.MultiStepLR(optimizer_corr, milestones = [10, 20], gamma=0.1)
    
    
    """ Loss unctions (06.02.20) """
    Dice_loss = DiceLoss(n_classes)
    

    """ Training loop """
    
    for epoch in range(num_epochs):
        print('======================')
        print('=== EPOCH %03d ===' % (epoch+1))
        log_string_overall('**** EPOCH %03d ****' % (epoch+1))
        log_string_overall(str(datetime.now()))
        
        # Initialise storage for loss results.
        seg_loss_epoch = []
        corr_loss_epoch = []

        # Iterate over data.
        for i, (inputs, targets) in enumerate(train_dataloader):
            
            """ Input images and targets """
            inputs, targets = Variable(inputs), Variable(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            
            """ Zero both networks' gradients after each image/target batch iteration """
            optimizer_seg.zero_grad()
            optimizer_corr.zero_grad()

            """ Forward passes """
            # Send images through model: outputs.size() = [1, n_classes=2, 512, 512]
            outputs_seg = model_seg(inputs) 
            # Send binary edge-map output from Unet_Seg through Unet_Corr
            outputs_corr = model_corr(outputs_seg)
            
            """ Compute losses """
            # Compute loss between predicted edges and GT edges
            DiceLoss_seg = Dice_loss(outputs_seg, targets)
            # Compute loss between generated/corrected edge-map and GT edges
            DiceLoss_corr = Dice_loss(outputs_corr, targets)

            # Total Unet_Seg loss
            loss_seg = DiceLoss_seg + DiceLoss_corr
            # Total Unet_Corr loss
            loss_corr = DiceLoss_corr
               
            """ Backward passes """
            # Unet_Seg gradients: errors between edge map and predicted edges
            loss_seg.backward()  
            optimizer_seg.step() # update weights/biases 
           
            # Unet_Corr gradients: errors between corrected edges and GT edges
            loss_corr.backward()  # propagate errors backwards through network
            optimizer_corr.step() # update weights/biases 
           
            """ Append loss lists """
            # Append total training loss of each image to list for averaging later   
            seg_loss_epoch.append(loss_seg.detach().cpu().numpy())   
            corr_loss_epoch.append(loss_corr.detach().cpu().numpy())   
            
        """ Average total loss for all images """
        epoch_segloss  = np.mean(seg_loss_epoch) 
        epoch_corrloss = np.mean(corr_loss_epoch) 

        """ Print the training/validation results for that epoch """
        print('Epoch {} stats.'.format(epoch+1))
        print('   Avg. SegNet Loss: {:.4f}'.format(epoch_segloss)) # Averaged results
        print('   Avg. CorrNet Loss: {:.4f}'.format(epoch_corrloss)) # Averaged results

        # Write results to training and validation logs
        log_string_loss( ('%f') % (epoch_segloss) )
        log_string_loss( ('%f') % (epoch_corrloss))
        
        # Update learning rate at specified milestones
        scheduler_seg.step()
        scheduler_corr.step()
        
        # Save model every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model_seg.state_dict(), '%s/%s_model_%d.pth' % (args.out1, 'segnet', epoch+1))
            torch.save(model_corr.state_dict(),'%s/%s_model_%d.pth' % (args.out2, 'corrnet', epoch+1))

                  