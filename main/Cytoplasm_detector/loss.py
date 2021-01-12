#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss function repo.

Collection of useful loss functions.
    - DiceLoss
    - CrossEntropy
    - Beta-Weighted CE for edge detection
    - Focal Loss (added 28.02.2020)
    - Ranking loss (added 18.03.2020)


@author: calmac
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

 
"""
Dice loss function. 

Used to compute the loss only between the background and cell clump classes.
"""  
class DiceLoss(nn.Module):
    def __init__(self, class_num, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, output, target):
        output = F.softmax(output, dim=1)   #  converts raw outputs into probs
        Dice = Variable(torch.Tensor([0]).float())
        Dice = Dice.to(torch.device("cuda")) # need this line otherwise the loss wont be loaded to the GPU; spits out an error
        
        for i in range(1, self.class_num): 
            output_i = output[:, i, :, :]   # extract each prediction map and compare with target
            target_i = target[:, i, :, :]   # extract the OHE vector for each class 
            intersect = torch.sum(output_i * target_i)
            union = torch.sum(output_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - ( Dice / (self.class_num - 1) )
       
        return dice_loss


"""
Ranking loss: 
      
Date: 18.03.2020.
"""

class RankingLoss(nn.Module):

    def __init__(self, class_num, gamma=2, alpha=None, eps=1e-7):
        super(RankingLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        
    def forward(self, outputs, targets, rank_map):
        
        # Remove background class (i.e. outputs[:, 0, :, :] ) during loss calculations. 
        # Only need background for performance calculations.
        outputs = outputs[:, 1, :, :].unsqueeze(0) # to keep 4-dims
        targets = targets[:, 1, :, :].unsqueeze(0) # to keep 4-dims       
        # outputs and targets new size.() = [1], 1, 512, 512] for cyto_edges class.
        
        # Compute probs 
        p = torch.sigmoid(outputs) # compute probabilities from raw network outputs: use sigmoid as rec
        p = p.clamp(self.eps, 1 - self.eps) # limit outputs between {0, 1}

        # Compute ranked probability, pr
        k = torch.FloatTensor.abs(1 - rank_map) # k = power that upcoming exp is to be given
        pr = p / torch.exp(k)                   # ranked probability 
        
        # General form of ranking loss 
        loss = - ( (1 - pr)**self.gamma   * targets     * torch.log(pr)    \
                 + (pr)    **self.gamma   * (1-targets) * torch.log(1 - pr) \
                  )
        return loss.mean(dim=1).sum() # sum the loss results across all domains 
    

"""
Focal loss: used to enhance the weight of training on harder-to-detect pixels 
    - idea is to recover more overlapped edges for improving the geodesic path evolution 
      between connection points.
      
Date: 28.02.2020.
"""

class FocalLoss(nn.Module):

    def __init__(self, class_num, gamma=2, alpha=None, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        
    def forward(self, outputs, targets):
        
        # Remove background class (i.e. outputs[:, 0, :, :] ) from both 
        # outputs and targets to give .size() = [1, 2, 512, 512] for 3-class (background, cyto, nuclei) problem
        outputs = outputs[:, 1:self.class_num, :, :]
        targets = targets[:, 1:self.class_num, :, :]
        
        logit = torch.sigmoid(outputs) # compute probabilities from raw network outputs: use sigmoid as rec
        logit = logit.clamp(self.eps, 1 - self.eps) # limit outputs between {0, 1}

        # General form of Focal loss/Cross entropy
        loss = - ( (1 - logit)**self.gamma   * targets     * torch.log(logit)  \
                  + (logit)   **self.gamma   * (1-targets) * torch.log(1 - logit) \
                  )
        return loss.mean(dim=1).sum() # sum the loss results across all domains 
    
"""
Edge weighted Cross-entropy loss.
Date: 27/01/2020.

Designed this based on CASENet and DDS papers, which both use this kind of loss.
Suitable for semantic edge detection.  

Takes in full one-hot encoded tensors of both network output and targets (both size [class_num, 512, 512]).
    - outputs: in raw prediction form.
    - targets: in 1/0 form, so feed into function.
   
Computes the proportion of edge pixels (beta) in each image as they are read by the DataLoader().
Removes the need for loading weight maps each iteration (saves comp. time.)

"""
   
class WeightedMultiClassSigmoidCE_loss(nn.Module):
    def __init__(self, class_num):
        super(WeightedMultiClassSigmoidCE_loss, self).__init__()
        self.class_num = class_num
        
    def forward(self, outputs, targets):
        """
        Compute the edge pixel proportion, edge_weight, for every image.
        loss: compute the loss for each pixel i in |I|, 
                    sum all pixel loss values together, 
                      then average across all pixels. 
        
        Inputs:               
            outputs: the [B x C x H x W] network output: here, [1, k, 512, 512]
            targets: the [B x C x H x W] network output: here, [1, k, 512, 512]
        
        Result:
            loss: same size as inputs [1, k, 512, 512]
            
        """
        # Remove background class (i.e. outputs[:, 0, :, :] ) from both 
        # outputs and targets to give .size() = [1, 2, 512, 512] for 3-class (background, cyto, nuclei) problem
        if self.class_num != 1 :
            outputs = outputs[:, 1:self.class_num, :, :]
            targets = targets[:, 1:self.class_num, :, :]
        
        # Weight
        n_edge_pixels = torch.sum(targets[:,0,:,:]).float().data # total number of 1s in the target edges: compute for cytoplasms since nuclei arent edges anymore
        beta = n_edge_pixels.data / float(targets.size()[2] * targets.size()[3]) # beta: percentage of edge-pixels in the image (small)
        one_sigmoid_out = torch.sigmoid(outputs) # network predictions 
        zero_sigmoid_out = 1 - one_sigmoid_out

        loss = - ( (1-beta) * targets     * torch.log(one_sigmoid_out.clamp(min=1e-10))  \
                  + beta    * (1-targets) * torch.log(zero_sigmoid_out.clamp(min=1e-10)) \
                  )
               
        loss = loss.mean(dim=1).sum() # sum over the classes, and take the mean between them
        
        return loss
 
"""
================================================================
Other functions/older versions
================================================================

Cross-entropy loss.
Date: 14/01/2020.

Designed this myself since I have one-hot encoded tensors, which nn.CrossEntropyLoss or F.nll_loss() dont accept. 

Takes in full one-hot encoded tensors of both network output and targets (both size [262144, 4]).
    - output: already in log_softmax form, so just feed straight into function.
    - target: in 1/0 form, so feed into function.
   
Also takes in the distance-transform weight map [size = [262144,1] to emphasise the pixels of interest. 
(20/01/2020): added in the class-weighted map to diminish background pixels even further.

"""
class cross_entropy_loss(nn.Module):
    def __init__(self, class_num=3):
        super(cross_entropy_loss, self).__init__()
        self.class_num = class_num
        
    def forward(self, output, target, dist_map, class_map=None):
        ce_loss = Variable(torch.Tensor([0]).float())
        ce_loss = ce_loss.to(torch.device("cuda")) # need this line otherwise the loss wont be loaded to the GPU; spits out an error
        i = 0
#        ce_list = []
        for i in range(0, self.class_num):   
            if i == self.class_num:
                break                   # failsafe incase i goes to 4 again, which gives an error
            output_i = output[:, i]     # extract prediction map for ith class (log.softmax format)
            target_i = target[:, i]     # extract the OHE target for ith class 
            weightmap = dist_map  
            ce = -torch.sum(weightmap * target_i * torch.log(output_i), dim=0)  # compute CE for ith class, where CE = target * log(p), where output_i = softmax predictions
            ce = ce / 262144    # Mean loss across all pixels for the ith class: this reduces the loss by the number of samples (ie ce/262144)
            ce_loss += ce       # append total CE loss across all classes
            print(ce_loss)
#        ce_loss = torch.sum(ce_list)  # total CE loss across all classes
        
        return ce_loss



"""
Edge weighted Cross-entropy loss.
Date: 27/01/2020.

Designed this based on CASENet and DDS papers, which both use this kind of loss.
Suitable for semantic edge detection.  

Takes in full one-hot encoded tensors of both network output and targets (both size [class_num, 512, 512]).
    - outputs: in raw prediction form.
    - targets: in 1/0 form, so feed into function.
   
Computes the proportion of edge pixels (beta) in each image as they are read by the DataLoader().
Removes the need for loading weight maps each iteration (saves comp. time.)

Does it do multi-class? Test it -- YES

Updates:
    - 05.02.2020: (probs not, 06.02) added if statement to switch between class-agnostic and class-aware
                  loss calculation. 
                  If edge=True, then the function only considers the targets from the edge/non-edge 
                  targets, in addition to the edge/non-edge side outputs at each layer. 
                  Else, the function considers it a multi-class problem and so switches to 
                  summing the results over the one-hot encoded targets.
                  
""" 
class WeightedMultiClassSigmoidCE_loss2(nn.Module):
    def __init__(self, class_num):
        super(WeightedMultiClassSigmoidCE_loss2, self).__init__()
        self.class_num = class_num
        
    def forward(self, outputs, targets):
        """
        Compute the edge pixel proportion, edge_weight, for every image.
        loss: compute the loss for each pixel i in |I|, 
                    sum all pixel loss values together, 
                      then average across all pixels. 
        Inputs:               
            outputs: the [B x C x H x W] network output: here, [1, k, 512, 512]
            targets: the [B x C x H x W] network output: here, [1, k, 512, 512]
        
        Result:
            loss: same size as inputs [1, k, 512, 512]
            
        """
        CE_weighted = Variable(torch.Tensor([0]).float())
        CE_weighted = CE_weighted.to(torch.device("cuda"))
        
        for i in range(1, self.class_num):
            
            # For class i (only 2: cyto borders, and nuclei borders)
            output_i = outputs[:, i, :, :]
            target_i = targets[:, i, :, :]
            
            # Weight
            n_edge_pixels = torch.sum(target_i).float().data # total number of 1s in the target class
            beta = n_edge_pixels.data / float(target_i.size()[2] * target_i.size()[3]) # beta: percentage of edge-pixels in the image (small)
            one_sigmoid_out = torch.sigmoid(output_i) # network predictions for class i
            zero_sigmoid_out = 1 - one_sigmoid_out
    
            # Compute the edge-weighted cross-entropy loss for class i
            loss_i = - ( (1-beta) * target_i       * torch.log(one_sigmoid_out.clamp(min=1e-10))  \
                     +    beta    * (1-target_i)   * torch.log(zero_sigmoid_out.clamp(min=1e-10)) \
                      )
            loss_i = loss_i.sum()  # sum the loss across each pixel
            CE_weighted += loss_i  # sum the loss of each class 
            
        loss = CE_weighted / (self.class_num - 1)  # average loss over the two classes
        
        return loss


class FocalLoss_onesided(nn.Module):

    def __init__(self, class_num, gamma=2, alpha=None, eps=1e-7):
        super(FocalLoss_onesided, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        
    def forward(self, outputs, targets):
        
        # Remove background class (i.e. outputs[:, 0, :, :] ) from both 
        # outputs and targets to give .size() = [1, 2, 512, 512] for 3-class (background, cyto, nuclei) problem
        outputs = outputs[:, 1:self.class_num, :, :]
        targets = targets[:, 1:self.class_num, :, :]
        
        logit = torch.sigmoid(outputs) # compute probabilities from raw network outputs: use sigmoid as rec
        logit = logit.clamp(self.eps, 1 - self.eps) # limit outputs between {0, 1}

        # Compute one sided focal loss
        loss = - (1 - logit)**self.gamma   * targets     * torch.log(logit)
    
        return loss.mean(dim=1).sum() # take the mean between each class, and sum all of the probabilities

"""
Updates:
    
    04.03.2020:
        - adjusted BetaWeighted to account for when we use edge_losses where  
          self.class_num will = 1, meaning that outputs/targets will return an error.
          - added if statement to deal with this if class_num = 1.
    02.03.2020:
        - made beta tuned towards cytoplasm edges by having .sum(targets[:,0,:,:] (ie only considering cytoplasm class)
            - in this dataset, nuclei are no longer edges, so they will occupy a lot of the pixel count, thus lowering the 
              weight placed on the cytoplasm edges. 
    28.02.20:
        - added Focal loss: hopefully this will help Unet detect more of the harder
          overlapped pixels.
    24.02.2020:
        - updated WeightedLoss for doing the same as DiceLoss; calculating loss 
          for each class, add together, then take the average
"""

if __name__ == "__main__":
    
    """
    ----------------------------------------------
    Lets have a look at what "loss" is doing in the loss function.
    ----------------------------------------------
    """
    
    # Network outputs 
    outputs = torch.randn(1, 3, 512, 512)
    targets = torch.randn(1, 3, 512, 512)
    
    FL = FocalLoss(3)
    
    ce_loss, focal_loss = FL(outputs, targets) 
    
    
#    loss = loss.detach().cpu().numpy()












