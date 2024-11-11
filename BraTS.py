#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:52:58 2024

@author: zzx
"""

import os
import torch
from torch.utils.data import Dataset
from utils import load_nii, minmax, pad_or_crop_image, pad_image_and_label

class BraTS(Dataset):
    def __init__(self, 
        data_folder: str, 
        list_folder: str, 
        mode: str = "train"
    ):
        super().__init__()
        self.data_folder = data_folder
        self.mode = mode
        subject_ids = []
        if mode == "train":
            list_file = os.path.join(list_folder, "train_list.txt")
        elif mode == "train_val":
            list_file = os.path.join(list_folder, "validation_list.txt")
        elif mode == "test":
            list_file = os.path.join(list_folder, "test_list.txt")
        with open(list_file) as fid:
            for f in fid:
                subject_ids.append(f.strip('\n'))
                
        self.data_set = []
        pattens = ["_t1", "_t1ce", "_t2", "_flair", "_seg"]
        for pid in subject_ids:
            file_paths = [f"{pid}{patten}.nii.gz" for patten in pattens]
            subject = {
                "id": pid,
                "t1": file_paths[0],
                "t1ce": file_paths[1],
                "t2": file_paths[2],
                "flair": file_paths[3],
                "seg": file_paths[4]
            }
            self.data_set.append(subject)
    
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        subject = self.data_set[idx]
        subject_id = subject["id"]
        pad_list = []
        crop_list = []
        file_dir = os.path.join(self.data_folder, subject_id)
        subject_image = {key: torch.tensor(load_nii(os.path.join(file_dir, subject[key]))) for key in subject if key not in ["id", "seg"]}
        subject_image = torch.stack([subject_image[key] for key in subject_image])
        subject_label = torch.tensor(load_nii(os.path.join(file_dir, subject["seg"])), dtype=torch.int8)
        et = subject_label == 4
        tc = torch.logical_or(subject_label == 1, subject_label == 4)
        wt = torch.logical_or(subject_label == 2, tc)
        subject_label = torch.stack([et, tc, wt])
        
        nonzero_index = torch.nonzero(torch.sum(subject_image, dim=0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:,0], nonzero_index[:,1], nonzero_index[:,2]
        zmin, ymin, xmin = [max(0, int(torch.min(i) - 1)) for i in (z_indexes, y_indexes, x_indexes)]  # why -1?
        # zmin, ymin, xmin = [int(torch.min(i)) for i in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(torch.max(i) + 1) for i in (z_indexes, y_indexes, x_indexes)]
        subject_image = subject_image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()
        for i in range(subject_image.shape[0]):
            subject_image[i] = minmax(subject_image[i])
        subject_label = subject_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        
        if self.mode == "train" or self.mode =="train_val":
            subject_image, subject_label, pad_list, crop_list = pad_or_crop_image(subject_image, subject_label, target_size=(128, 128, 128))
        elif self.mode == "test":
            d, h, w = subject_image.shape[1:]
            pad_d = (128-d) if 128-d > 0 else 0
            pad_h = (128-h) if 128-h > 0 else 0
            pad_w = (128-w) if 128-w > 0 else 0
            subject_image, subject_label, pad_list = pad_image_and_label(subject_image, subject_label, target_size=(d+pad_d, h+pad_h, w+pad_w))
            
        return dict(
            subject_id=subject["id"],
            image=subject_image,
            label=subject_label,
            nonzero_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            box_slice=crop_list,
            pad_list=pad_list
        )   
