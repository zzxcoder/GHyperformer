#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:31:13 2024

@author: zzx
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import SimpleITK as sitk
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.inferers import sliding_window_inference
import pandas as pd
from config import ExperimentConfig
from BraTS import BraTS

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="name of model", default="GHyperFormer")
    parser.add_argument("--name", type=str, help="name of experiment", default="GHyperformer_Adam_CosineAnnealingLR")
    parser.add_argument("--dataset", type=str, help="root dir for dataset", default="/media/zzx/DATA/MedicalImage/BraTS/BraTS2021/BraTS2021_Training_Data")
    parser.add_argument("--listfolder", type=str, help="folder for list file", default="/media/zzx/DATA/GHyperformer/dataset")
    parser.add_argument("--experiment", type=str, help="root dir for experiment", default="/media/zzx/DATA/GHyperformer/experiments")
    parser.add_argument("--bestfolder", type=str, help="root dir for best model", default=None)
    parser.add_argument("--segmentation", type=str, help="root dir for segmentation", default="None")
    parser.add_argument("--csvfolder", type=str, help="root dir for csv file", default=None)
    parser.add_argument("--workers", type=int, default=1, help="The value of CPU's num_worker")
    args = parser.parse_args()
    return args

def init_work_folder(args):
    experiment_dir = os.path.join(args.experiment, args.name)
    if not os.path.exists(experiment_dir): 
        os.makedirs(experiment_dir, exist_ok=True)
    args.experiment = experiment_dir
    segmentation_folder = os.path.join(experiment_dir, "segmentations")
    if not os.path.exists(segmentation_folder):
        os.makedirs(segmentation_folder, exist_ok=True)
    args.segmentation = segmentation_folder
    best_model_dir = os.path.join(experiment_dir, "best_model")
    if not os.path.exists(best_model_dir): 
        os.makedirs(best_model_dir, exist_ok=True)
    args.bestfolder = best_model_dir
    csv_folder = os.path.join(experiment_dir, "csv")
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder, exist_ok=True)
    args.csvfolder = csv_folder

def get_test_cfg(args):
    cfg = ExperimentConfig[args.name]
    cfg["test"]["model_name"] = args.model
    cfg["test"]["dataset_folder"] = args.dataset
    cfg["test"]["list_folder"] = args.listfolder
    cfg["test"]["experiment_folder"] = args.experiment
    cfg["test"]["bestmodel_folder"] = args.bestfolder
    cfg["test"]["segmentation_folder"] = args.segmentation
    cfg["test"]["csv_folder"] = args.csvfolder
    cfg["test"]["workers"] = args.workers
    return cfg["test"]

def get_model(name, cfg):
    if name == "GHyperFormer":
        from model import GHyperFormer
        return GHyperFormer(cfg)

def save_test_label(args, patient_id, predict):
    ref_img = sitk.ReadImage(os.path.join(args["dataset_folder"], f"{patient_id}/{patient_id}_t1.nii.gz"))
    label_nii = sitk.GetImageFromArray(predict)
    label_nii.CopyInformation(ref_img)
    sitk.WriteImage(label_nii, os.path.join(args["segmentation_folder"], f"{patient_id}.nii.gz"))
    
def save_seg_csv(cfg, csv):
    try:
        val_metrics = pd.DataFrame.from_records(csv)
        columns = ['id', 'et_dice', 'tc_dice', 'wt_dice', 'et_hd', 'tc_hd', 'wt_hd', 'et_sens', 'tc_sens', 'wt_sens', 'et_spec', 'tc_spec', 'wt_spec']
        val_metrics.to_csv(f"{str(cfg['csv_folder'])}/metrics.csv", index=False, columns=columns)
    except KeyboardInterrupt:
        print("Save CSV File Error!")

def cal_confuse(preds, targets, patient):
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"
    labels = ["ET", "TC", "WT"]
    confuse_list = []
    for i, label in enumerate(labels):
        if torch.sum(targets[i]) == 0 and torch.sum(targets[i]==0):
            tp=tn=fp=fn=0
            sens=spec=1
        elif torch.sum(targets[i]) == 0:
            print(f'{patient} did not have {label}')
            sens = tp = fn = 0      
            tn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), torch.logical_not(targets[i])))
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i])))
            spec = tn / (tn + fp)
        else:
            tp = torch.sum(torch.logical_and(preds[i], targets[i]))
            tn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), torch.logical_not(targets[i])))
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i])))
            fn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
        confuse_list.append([sens, spec])
    return confuse_list

def cal_dice(predict, target, haussdor, dice):
    p_et = predict[0]
    p_tc = predict[1]
    p_wt = predict[2]
    t_et = target[0]
    t_tc = target[1]
    t_wt = target[2]
    p_et, p_tc, p_wt, t_et, t_tc, t_wt =  p_et.unsqueeze(0).unsqueeze(0), p_tc.unsqueeze(0).unsqueeze(0), p_wt.unsqueeze(0).unsqueeze(0), t_et.unsqueeze(0).unsqueeze(0), t_tc.unsqueeze(0).unsqueeze(0), t_wt.unsqueeze(0).unsqueeze(0)
    
    if torch.sum(p_et) != 0 and torch.sum(t_et) != 0:
        et_dice = np.mean(dice(p_et, t_et).cpu().numpy())
        et_hd = np.mean(haussdor(p_et, t_et).cpu().numpy())
    elif torch.sum(p_et) == 0 and torch.sum(t_et) == 0:
        et_dice =1
        et_hd = 0
    elif (torch.sum(p_et) == 0 and torch.sum(t_et) != 0) or (torch.sum(p_et) != 0 and torch.sum(t_et) == 0):
        et_dice =0
        et_hd = 347
    if torch.sum(p_tc) != 0 and torch.sum(t_tc) != 0:
        tc_dice = np.mean(dice(p_tc, t_tc).cpu().numpy())
        tc_hd = np.mean(haussdor(p_tc, t_tc).cpu().numpy())
    elif torch.sum(p_tc) == 0 and torch.sum(t_tc) == 0:
        tc_dice =1
        tc_hd = 0
    elif (torch.sum(p_tc) == 0 and torch.sum(t_tc) != 0) or (torch.sum(p_tc) != 0 and torch.sum(t_tc) == 0):
        tc_dice =0
        tc_hd = 347
    if torch.sum(p_wt) != 0 and torch.sum(t_wt) != 0:
        wt_dice = np.mean(dice(p_wt, t_wt).cpu().numpy())
        wt_hd = np.mean(haussdor(p_wt, t_wt).cpu().numpy())
    elif torch.sum(p_wt) == 0 and torch.sum(t_wt) == 0:
        wt_dice =1
        wt_hd = 0
    elif (torch.sum(p_wt) == 0 and torch.sum(t_wt) != 0) or (torch.sum(p_wt) != 0 and torch.sum(t_wt) == 0):
        wt_dice =0
        wt_hd = 347
    
    return [et_dice, tc_dice, wt_dice, et_hd, tc_hd, wt_hd]

def reconstruct_label(image):
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    image = (c3 > 0).astype(np.uint8)
    image[(c2 == False)*(c3 == True)] = 2
    image[(c1 == True)*(c3 == True)] = 4
    return image

def inference(model, dataloader, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()
    device_ids = list(range(device_count))
    model = nn.DataParallel(model, device_ids)
    
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)
    for i, data in enumerate(dataloader):
        subject_id = data['subject_id'][0]
        print(f"processing subject id {subject_id}.")
        image, label = data["image"].to(device), data["label"].to(device)
        # image, label = data["image"], data["label"]
        pad_list = data["pad_list"]
        nonzero_indexes = data["nonzero_indexes"]
        with torch.no_grad():
            predict = torch.sigmoid(sliding_window_inference(image, roi_size=(128, 128, 128), sw_batch_size=2, predictor=model, overlap=0.6))
        
        label = label[:, :, pad_list[-4]:label.shape[2]-pad_list[-3], pad_list[-6]:label.shape[3]-pad_list[-5], pad_list[-8]:label.shape[4]-pad_list[-7]]
        predict = predict[:, :, pad_list[-4]:predict.shape[2]-pad_list[-3], pad_list[-6]:predict.shape[3]-pad_list[-5], pad_list[-8]:predict.shape[4]-pad_list[-7]]
        predict = (predict>0.5).squeeze()
        label = label.squeeze()
        dice_metrics = cal_dice(predict, label, haussdor, meandice)
        confuse_metric = cal_confuse(predict, label, subject_id)
        et_dice, tc_dice, wt_dice = dice_metrics[0], dice_metrics[1], dice_metrics[2]
        et_hd, tc_hd, wt_hd = dice_metrics[3], dice_metrics[4], dice_metrics[5]
        et_sens, tc_sens, wt_sens = confuse_metric[0][0], confuse_metric[1][0], confuse_metric[2][0]
        et_spec, tc_spec, wt_spec = confuse_metric[0][1], confuse_metric[1][1], confuse_metric[2][1]
        metrics_dict.append(dict(
            id=subject_id,
            et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice,
            et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
            et_sens=et_sens, tc_sens=tc_sens, wt_sens=wt_sens,
            et_spec=et_spec, tc_spec=tc_spec, wt_spec=wt_spec
        ))
        full_predict = np.zeros((155, 240, 240))
        predict = reconstruct_label(predict)
        full_predict[slice(*nonzero_indexes[0]), slice(*nonzero_indexes[1]), slice(*nonzero_indexes[2])] = predict
        save_test_label(cfg, subject_id, full_predict)
    save_seg_csv(cfg, metrics_dict)

def test(model, cfg):
    db_test = BraTS(cfg["dataset_folder"], cfg["list_folder"], mode="test")
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=cfg["workers"])
    print("{} samples for testing".format(len(test_loader)))
    
    model.load_state_dict(torch.load(os.path.join(cfg["bestmodel_folder"], "best_model.pkl")))
    print("The model for testing load.")
    
    model.eval()
    inference(model, test_loader, cfg)
    print("Testing finished!")

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    args = get_parse_args()
    init_work_folder(args)
    net_cfg = ExperimentConfig[args.name]["net"]
    model = get_model(args.model, net_cfg)
    test_cfg = get_test_cfg(args)
    test(model, test_cfg)
    
    
