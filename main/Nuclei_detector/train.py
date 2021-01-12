#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network no.2: 
    CYTOPLASM EDGE DETECTOR

Script for building the model and the datasets, then training the network.
This is for trying out the method used by the defect detection paper, where the 
network is built as a category-agnostic edge detector (ie just cares about edges, 
not what classes those edges belong to).

FOr ISBI dataset 
                                                



---------------------------------
---------------------------------

Latest updates: 
    
    18.03:
        - updated to run the new RankingLoss 
        
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
from ISBI14_dataset import TrainingDataset
from models import init_weights, Unet, Att_Unet, Edge_Att_Unet
from loss import DiceLoss
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
parser.add_argument('--out',        default='./model_checkpoint', type=str, help='path to save model checkpoints')
parser.add_argument('--lr-mode',    default='step', type=str)
parser.add_argument('--step',       default=20, type=int)

# Hyperparameters (UPDATED: 27.01.20)
args = parser.parse_args()
batch_size = args.batch_size 
num_epochs = args.epochs   
in_channels = 1              # grayscale images thus only 1 channel      
n_classes = 2       

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

"""Directories for paths to data."""
images_dir  = "/home/hsijcr/calummac/ISBI14_Dataset/TwoNetworks/NucleiDetector_data/train_dataset/NucDet_training_images.mat"
targets_dir = "/home/hsijcr/calummac/ISBI14_Dataset/TwoNetworks/NucleiDetector_data/train_dataset/NucDet_training_targets.mat"
data_dirs = [images_dir, targets_dir]
     

if __name__ == '__main__':
    
    
    """
    Datasets
    """
    # Get the full set of data
    train_dataset = TrainingDataset(data_dirs)
    
    # Assign DataLoader to data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)

    """ 
    Build the network 
    """
    model = Unet(in_channels, n_classes) # Load the model and assign input and output channels
    model.to(device)                     # send to GPU/CPU
    model.apply(init_weights)            # apply normal distribution to initialise weights
    model.train()                        # set to training mode
    
    # Count number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: {:4.2f} M'.format(pytorch_total_params / 1e6))
    
    # Assign optimiser and learning rate scheduler
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 20], gamma=0.1)
    
    """ New loss functions (06.02.20) """
    Dice_loss = DiceLoss(n_classes)

    """
    Training loop
    """
    
    for epoch in range(num_epochs):
        print('======================')
        print('=== EPOCH %03d ===' % (epoch+1))
        log_string_overall('**** EPOCH %03d ****' % (epoch+1))
        log_string_overall(str(datetime.now()))
        
        # Initialise storage for loss results.
        train_loss_epoch = []
        
        # Iterate over data.
        for i, (inputs, targets) in enumerate(train_dataloader):
            
            """Input images and targets"""
            inputs, targets = Variable(inputs), Variable(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # zero the parameter gradients after each image/target pass
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs) # send images through model: outputs.size() = [1, n_classes, 512, 512]

            """ Loss """
            loss = Dice_loss(outputs, targets)
             
            """ Backwards + optimize (only if in training phase) """
            loss.backward()  # propagate errors backwards through network
            optimizer.step() # update weights/biases 
           
            # Append total training loss of each image to list for averaging later                   
            train_loss_epoch.append(loss.detach().cpu().numpy())   
            
        epoch_loss = np.mean(train_loss_epoch) # average total loss (DC + CE) for all images
        
        # Print the training/validation results for that epoch
        print('Epoch {} stats.'.format(epoch+1))
        print('   Avg. Loss: {:.4f}'.format(epoch_loss)) # Averaged results

        
        # Write results to training and validation logs
        log_string_loss( ('%f') % (epoch_loss) )
        
        # Update learning rate at specified milestones
        scheduler.step()
        
        # Save model every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), '%s/%s_model_%d.pth' % (args.out, 'cellnet', epoch+1))


                  