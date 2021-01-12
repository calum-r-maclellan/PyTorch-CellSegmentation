#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script is required for calling the images and targets from the ISBI datasets.

@author: calmac

Date created: 02.03.2020.

Updates:
    - 18.03:
        - included functions for reading in pre-computed rankmaps
"""


from __future__ import print_function
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms 
from scipy import io  # for loading matlab (.mat) files 
import matplotlib.pyplot as plt
from utils import split_dataset
from models import Unet

class TrainingDataset(data.Dataset):
    
    def __init__(self, data_dirs):

        """ Inputs (i.e. images from .mat file) """
        n = 1
        self.imagefile_list  = io.loadmat(data_dirs[0])     # INPUTS: read MATLAB .mat file containing 100 examples
        self.inputs = self.imagefile_list['images']   # matlab variable name: 'images' 
        self.inputs = np.squeeze(self.inputs) # squeezes object into one column   
        self.inputs = self.inputs[0:n] # use only X examples to test if the function is working properly
        
        """ Targets from .mat file"""
        self.targetfile_list = io.loadmat(data_dirs[1])    # TARGETS: read MATLAB .mat file containing 100 examples 
        self.targets = self.targetfile_list['targets']     # matlab variable name: 'targets'
        self.targets = np.squeeze(self.targets) # squeezes object into one column   
        self.targets = self.targets[0:n] # use only X examples to test if the function is working properly

    def __getitem__(self, index):
        
        """
        -------------------------
        Input images:    
        -------------------------
        """
        
        # Get image at index i (from DataLoader in main loop)
        inputfile_i = self.inputs[index]
        
        # Convert to torch tensor from numpy.
        train_img_tensor = torch.from_numpy(inputfile_i).float()
        
        # Reshape to 4-dimensions (ie include N batch size at index 0). Required for convolution during Unet. 
        train_img_tensor = train_img_tensor.unsqueeze(0)
        
        
        """
        ------------------------
        Targets (only 2 for both edge classes):
            - cytoplasm edges
            - nuclei (edges)
        ------------------------
        """
        # Get the ith target file from the list. 
        targetfile_i = self.targets[index]
        
        # Need to squeeze again because the targets file has Nx1 cell (hence first squeeze in __init__())
        # which contains separate Xx1 cell arrays. 
        # But then need to be squeezed in order to access inner info (ie 512x512 array of 1/0s)
        targetfile_i = np.squeeze(targetfile_i) 
        
        # Send the file to create_tensor() for one hot encoding. 
        train_target_tensor = self.create_tensor(targetfile_i)
        
        return [train_img_tensor, train_target_tensor]
    
   
    def create_tensor(self, file): 
        """
        create_tensor():
            
        Given a matlab file (.mat), spit out a torch tensor of one-hot encoded classes.
        This is for the targets only, which are already one-hot encoded.
        
        Since we need the targets to be in the form [n, 512, 512], where n is 
        the number of classes in the image, this function takes in a file, figures out 
        how many classes are in it (using C.shape[0]), and the image dimensions (512, 512)
        it then assigns each 512x512 array to each class using X[i] = C[i].

        """
        
        # Currently file is still an object, which we cant do anything with.
        # Create a numpy array by cycling through each image in that target file and pulling out the targets one by one.
        X = np.empty((file.shape[0], file[0].shape[0], file[0].shape[1])) # preallocate numpy array of size equivalent to number of cells in that image.
        
        # Assign one-hot encoded (OHE) targets to numpy array, X.
        for i in range(X.shape[0]):  # for each image (ie cell 1 to 5)
            X[i] = file[i]              # assign each (512, 512) array of 1/0 information to its own OHE class.
            # which gives a one-hot encoded numpy array for each cell class (shape = (5, 512, 512) )
        
        # Need to have as tensor and in float form:
        tensor = torch.from_numpy(X).float()
        
        return tensor 
    
    def __len__(self):
        assert len(self.inputs) == len(self.targets)
        return len(self.inputs)
        

if __name__ == "__main__":
    
    # Test utils.py performance metrics
    images_dir  = "/home/hsijcr/calummac/Main_Dataset/train_dataset/training_images.mat"
    targets_dir = "/home/hsijcr/calummac/Main_Dataset/train_dataset/training_targets.mat"
    rankmap_dir = "/home/hsijcr/calummac/Main_Dataset/train_dataset/training_rankmaps.mat"
    data_dirs = [images_dir, targets_dir, rankmap_dir]
     
    test_dataset = TrainingDataset(data_dirs)
    
    # Load the model and get an output
    net = Unet(1, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    model_path =  './Unet_experiments/Dice_loss/1st_exp/model_checkpoint/cellnet_model_30.pth'
    net.load_state_dict(torch.load(model_path))
      
    image, targets = test_dataset[0]
    image = image.to(device).unsqueeze(0)
    targets = targets.to(device).unsqueeze(0)
        
    output = net(image)
    
    # Outputs
    outputs_flat = output.view(-1, 3)  # .size() = [262144, n_classes]: reshape segmentation map into a 512^2=262144 grid for each class output .                
    pred_indices = outputs_flat.data.max(1)[1].cpu().numpy()   # get the class indices corresponding to confidence scores
    print(pred_indices.shape)
    
    # Targets
    targets_flat = targets.view(-1, 3).long() # .size() = [262144, n_classes]: repeat for the targets
    target_indices = targets_flat.data.cpu().numpy() # label each of pixels (262144) with class index
    print(target_indices.shape)


    
    # Compute statistics
#    dice_list, precision_list, recall_list, tnr_list = compute_average_stats(outputs, targets, 3)

   
    
    
    
    