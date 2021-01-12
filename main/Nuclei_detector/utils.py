
"""
Originally compiled by He Sun.

Adapted by Calum.

Latest update: 
    
    
"""

import numpy as np
import torch
import os
import os.path as osp
#import cv2
#import scipy.misc as misc
import shutil
#from skimage import measure
import math
import traceback
from sklearn import metrics
import zipfile
from torch.nn import init
import random

 

def split_dataset(dataset, val_percent, test_percent):
    dataset = list(dataset)
    length = len(dataset)
    n_val = int(length * val_percent)
    n_test = int(length * test_percent)
    n_train = length - n_val - n_test
#    random.shuffle(dataset) # not anymore: done it in matlab to make it easier! # need to shuffle so that the training, val, and test contain examples of all difficulties of image 
    return {'train': dataset[:-(n_val+n_test)], 'val': dataset[(n_train) : (length-n_test)], 'test': dataset[-n_test:]} 
    # That bit does the actual splitting.
#         Train_set: get all from dataset [:] except (-) end bits (n_val+n_test)
#         Val_set: get the examples after training [n_train:] but not the last bit [length-n_test]
#         Test_set: get only the n examples from the end of the dataset [-n_test:].
    
def split_train_val(dataset, val_percent):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
#    random.shuffle(dataset)  
    return {'train': dataset[:-n], 'val': dataset[-n:]} # train = all data minus number of validation examples
                                                        # val   = the remaining number of examples


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
#        if self.count==100:
#            print(self.count)
        
def save_checkpoint(state, is_best,checkpoint_path,filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_best.pth.tar'))



""""""""""""""""""""""""""""""""""""""""""
""" Compute dice coefficient for accuracy"""

def compute_average_stats(predicted, gt, class_num=3):
    dice_list = []
    precision_list = []
    recall_list = []
    tnr_list = []
    for i in range(1, class_num):     # only interested in cyto_borders and nuclei 
        predict_i = predicted.copy()
        gt_i = gt.copy()
        predict_i[predict_i != i] = 0
        gt_i[gt_i != i] = 0
           
        dice = compute_dice_score(predict_i, gt_i, foreground=i)
        dice_list.append(dice)
        
        precision, recall, tnr = precision_and_recall(predict_i, gt_i, foreground=i)
        precision_list.append(precision)
        recall_list.append(recall)
        tnr_list.append(tnr)
        
        
    return dice_list, precision_list, recall_list, tnr_list

def compute_dice_score(predict, gt, foreground): 
    score = 0
    count = 0
    assert(predict.shape == gt.shape)
    # have an error message here if these two arent the same: make it easier to debug if the sizes arent equal.
    overlap = 2.0 * ((predict == foreground)*(gt == foreground)).sum()
    #print('overlap:',overlap)
    
    return (overlap + 0.001) / ( ( (predict == foreground).sum() + (gt == foreground).sum() ) + 0.001 )


""""""""""""""""""""""""""""""""""""""""""""
""" (29.01.2020) Precision, recall, F-score metrics """
# Use these for edge detection results, since Dice score doesnt seem to capture
# the results properly.

""""""""""""""""""""""""""""""""""""""""""""
""" 
Note:
    y_pred and y_true are already flattened and loaded into numpy (shape = [262144, n_classes]) 
"""

def precision_and_recall(y_pred, y_true, foreground):
    assert len(y_pred) == len(y_true)
    
    tp = ( (y_true==foreground) * (y_pred==foreground) ).sum()
    tn = ( (1 - (y_true==foreground)) * (1 - (y_pred==foreground)) ).sum()
    fp = ( (1 - (y_true==foreground)) * (y_pred==foreground) ).sum()
    fn = ( (y_true==foreground) * (1 - (y_pred==foreground)) ).sum()
    
    eps = 1e-6
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)         # true negative rate, or the specificity: tells us how much overlap has been predicted/neglected
    
    f1 = 2* (precision*recall) / (precision + recall + eps)
    
    return precision, recall, tnr 














