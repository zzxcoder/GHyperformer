#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:51:34 2024

@author: zzx
"""

import numpy as np
import random
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_nii(path):
    nii_file = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return nii_file

def minmax(image, low_perc=1, high_perc=99):
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = torch.clip(image, low, high)
    min_ = torch.min(image)
    max_ = torch.max(image)
    image = (image - min_) / (max_ - min_)
    return image

def get_crop_slice(target_size, input_size):
    if input_size > target_size:
        crop_extent = input_size - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return (left, input_size - right)
    else:
        return (0, input_size)
    
def get_pad_slice(target_size, input_size):
    if input_size >= target_size:
        return [False]
    else:
        pad_extent = target_size - input_size
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right
    
def pad_image_and_label(image, label, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    pad_info = [get_pad_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    pad_list = [0, 0]   
    for pad in pad_info:
        if pad[0]:
            pad_list.insert(0, pad[1])
            pad_list.insert(0, pad[2])
        else:
            pad_list.insert(0, 0)
            pad_list.insert(0, 0)
            
    if np.sum(pad_list) != 0:
        image = F.pad(image, pad_list, "constant")
    
    if label is not None:
        if np.sum(pad_list) != 0:
            label = F.pad(label, pad_list, "constant")
            
    return image, label, pad_list
    

def pad_or_crop_image(image, label, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    crop_list = [z_slice, y_slice, x_slice]
    image = image[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    if label is not None:
        label = label[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    image, label, pad_list = pad_image_and_label(image, label)
    return image, label, pad_list, crop_list


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
    def _one_hot_encoder(self, inputs):
        tensor_list = []
        for i in range(self.num_classes):
            prob = inputs == i
            tensor_list.append(prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _dice_value(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(score * score)
        gt_sum = torch.sum(target * target)
        dice = (2 * intersect + smooth) / (y_sum + gt_sum + smooth)
        return dice
        
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.num_classes
        assert inputs.size() == target.size(), "predict {} & target {} shape do not match".format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(self.num_classes):
            dice = self._dice_value(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.num_classes
