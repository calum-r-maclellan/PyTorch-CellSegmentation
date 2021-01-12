#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for building the model and the datasets, then training the network.
This is for trying out the method used by the defect detection paper, where the 
network is built as a category-agnostic edge detector (ie just cares about edges, 
not what classes those edges belong to).

First method wasnt working, so I'm trying this out and seeing what happens.
                                                
@author: Calum
Date created: 21/01/20.

---------------------------------
---------------------------------

Latest updates: 
    
   25.02.2020:
       - updated to run the deep supervision models.
    
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
from cell_dataset import TrainingDataset
from models_DS import init_weights, Unet_DS, Att_Unet, Edge_Att_Unet # this file contains the weights/biases included for edge_conv()
from utils import split_dataset, split_train_val
from loss import DiceLoss, WeightedMultiClassSigmoidCE_loss
from torch.autograd import Variable
#import multiprocessing
import copy
from pytorchtools import EarlyStopping
#import test_unet as test


# Create log for storing training and validation loss and accuracy results.
# NB: train results are odd, validation are even!!!
train_log_root = "./train_val_log"
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
parser.add_argument('--lr',         default=0.0001, type=float, help='learning rate')
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
n_classes = 3       # set to 3: 0=background, 1=cyto_borders, 2 = nuclei

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

"""Directories for paths to data."""
images_dir  = "/home/hsijcr/calummac/10_cells_edge_data/deep_supervision/training_images_shuf.mat"
targets_dir = "/home/hsijcr/calummac/10_cells_edge_data/deep_supervision/training_targets_shuf.mat"
data_dirs = [images_dir, targets_dir]
     

if __name__ == '__main__':
    
    
    """
    Datasets
    """
    # Get the full set of data
    dataset = TrainingDataset(data_dirs)
    
    # Use split_dataset() from utils to get training and validation datasets
#    ds = split_train_val(dataset, val_percent=0.10)
    ds = split_dataset(dataset, val_percent=0.10, test_percent=0.20)

    # Assign datasets
    train_dataset = ds['train']
    val_dataset = ds['val']
    test_dataset = ds['test']
    
    # Assign DataLoaders to both the sets of data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=4)
    train_dataloaders = {'Training': train_dataloader, 'Validation': val_dataloader}

    """ 
    Build the network 
    """
    # Load the model and assign input and output channels
    model = Edge_Att_Unet(in_channels, n_classes)
    model.to(device) # send to GPU/CPU
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: {:4.2f} M'.format(pytorch_total_params / 1e6))
    
    # Initialise weights with normal distribution
    model.apply(init_weights)
    
    # Assign optimiser and learning rate scheduler
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [15, 25], gamma=0.1)
    
    """ New loss functions (06.02.20) """
    Dice_loss = DiceLoss(n_classes)
#    ce_weighted_loss = WeightedMultiClassSigmoidCE_loss(n_classes)
    edge_loss = WeightedMultiClassSigmoidCE_loss(1) # agnostic edgemap loss: repeated for each layer

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
     
    # Initialise early_stopping object 
#    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    """
    Training/validation
    """
    
    for epoch in range(num_epochs):
        print('======================')
        print('=== EPOCH %03d ===' % (epoch+1))
        log_string_overall('**** EPOCH %03d ****' % (epoch+1))
        log_string_overall(str(datetime.now()))
        
        # Initialise storage for loss results.
        train_loss_epoch = []
        
        # Each epoch has a training and validation phase
        for phase in ['Training', 'Validation']:
            if phase == 'Training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            for i, (inputs, targets) in enumerate(train_dataloaders[phase]):
                
                """Input images and targets"""
                inputs, targets = Variable(inputs), Variable(targets)
                inputs, targets = inputs.to(device), targets.to(device)
                
                # zero the parameter gradients after each image/target pass
                optimizer.zero_grad()

                # Forward
                # track history only in training mode
                with torch.set_grad_enabled(phase == 'Training'):
                    
                    outputs = model(inputs) # send images through model: outputs.size() = [1, n_classes, 512, 512]

                    """ Edge supervision loss """
                    # Compare edgemap extracted from each skip layer with the 
                    # category-agnostic edgemap GTs. 
#                    E1_loss = edge_loss(x1_edge, edge_targets) 
#                    E2_loss = edge_loss(x2_edge, edge_targets)
#                    E3_loss = edge_loss(x3_edge, edge_targets)
#                    
#                    total_edge_loss = E1_loss + E2_loss + E3_loss

                    """ Semantic supervision loss """
                    loss1 = Dice_loss(outputs, targets)
#                    loss2 = ce_weighted_loss(outputs, targets)
                    
                    """ Total loss """
                    loss = loss1 #+ loss2
                    
                    """ Backwards + optimize (only if in training phase) """
                    if phase == 'Training':
                        loss.backward()  # propagate errors backwards through network
                        optimizer.step() # update weights/biases 
               
                # Append total training loss and accuracy for each epoch                    
                train_loss_epoch.append(loss.detach().cpu().numpy())    # total loss for ith image 
            
            epoch_loss = np.mean(train_loss_epoch) # average total loss (DC + CE) for all images
            
            # Print the training/validation results for that epoch
            print('Epoch {} {} stats.'.format(epoch+1, phase))
            print('   Avg. Dice Loss: {:.4f}'.format(epoch_loss)) # Averaged results

            
            # Write results to training and validation logs
            log_string_loss( ('%f') % (epoch_loss) )
            
            # Update learning rate at specified milestones
            if phase == 'Training':
                scheduler.step()
            
            # Save model every 10 epochs
            if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), '%s/%s_model_%d.pth' % (args.out, 'cellnet', epoch+1))

            # EXTRA STUFF
            # call early_stopping module for tracking the validation loss. 
            # if epoch_loss continues to rise steadily for n epochs, stop the training 
            # since the model is overfitting the data.
            # save the model parameters at the last checkpoint before early_stopping kicks in.
#            if phase == 'Validation':
#                early_stopping(epoch_loss, model, args.out, epoch)
#                
#                if early_stopping.early_stop:
#                    print("Overfitting has occurred. Terminate training.")

                  