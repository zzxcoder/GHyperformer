#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:29:05 2024

@author: zzx
"""

import argparse
import os
import logging
import numpy as np
import random
from datetime import datetime
from monai.transforms.spatial.array import Zoom
from monai.transforms.intensity.array import RandGaussianNoise, GaussianSharpen, AdjustContrast
from monai.losses.dice import DiceLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import ExperimentConfig
from BraTS import BraTS


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="name of model", default="GHyperFormer")
    parser.add_argument("--name", type=str, help="name of experiment", default="GHyperformer_Adam_CosineAnnealingLR")
    parser.add_argument("--num_classes", type=int, help="number of classes", default=3)
    parser.add_argument("--dataset", type=str, help="root dir for data", default="/media/zzx/DATA/MedicalImage/BraTS/BraTS2021/BraTS2021_Training_Data")
    parser.add_argument("--listfolder", type=str, help="folder for list file", default="/media/zzx/DATA/GHyperformer/dataset")
    parser.add_argument("--experiment", type=str, help="root dir for experiment", default="/media/zzx/DATA/GHyperformer/experiments")
    parser.add_argument("--bestfolder", type=str, help="root dir for best model", default=None)
    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--workers", type=int, default=1, help="The value of CPU's num_worker")
    parser.add_argument("--end_epoch", type=int, default=500, help="Maximum epoch for training")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of samples for training")
    parser.add_argument("--learn_rate", type=float, default=1e-4, help="The learning rate for training")
    parser.add_argument("--resume", type=bool, default=False, help="Wheather continue train or not")
    parser.add_argument("--checkpoint_folder", type=str, help="The folder saving checkpoint file", default=None)
    parser.add_argument("--val", type=int, default=1, help="Validation frequency of the model")
    parser.add_argument("--log", type=str, help="Training log file", default=None)
    parser.add_argument("--writer", type=str, help="Summary writer folder", default=None)
    args = parser.parse_args()
    return args

def init_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def init_work_dir(args):
    experiment_dir = os.path.join(args.experiment, args.name)
    if not os.path.exists(experiment_dir): 
        os.makedirs(experiment_dir, exist_ok=True)
    args.experiment = experiment_dir
    best_model_dir = os.path.join(experiment_dir, "best_model")
    if not os.path.exists(best_model_dir): 
        os.makedirs(best_model_dir, exist_ok=True)
    args.bestfolder = best_model_dir
    checkpoint_folder = os.path.join(experiment_dir, "checkpoint")
    if not os.path.exists(checkpoint_folder): 
        os.makedirs(checkpoint_folder, exist_ok=True)
    args.checkpoint_folder = checkpoint_folder
    log_folder = os.path.join(experiment_dir, "log")
    if not os.path.exists(log_folder):
    	os.makedirs(log_folder, exist_ok=True)
    args.log = os.path.join(log_folder, f"{args.model}_{str(datetime.now())}.log")
    writer_folder = os.path.join(experiment_dir, "writer")
    if not os.path.exists(writer_folder):
        os.makedirs(writer_folder, exist_ok=True)
    args.writer = writer_folder
    
def get_train_cfg(args):
    cfg = ExperimentConfig[args.name]
    cfg["train"]["model_name"] = args.model
    cfg["train"]["num_classes"] = args.num_classes
    cfg["train"]["dataset_folder"] = args.dataset
    cfg["train"]["list_folder"] = args.listfolder
    cfg["train"]["experiment_folder"] = args.experiment
    cfg["train"]["bestmodel_folder"] = args.bestfolder
    cfg["train"]["seed"] = args.seed
    cfg["train"]["workers"] = args.workers
    cfg["train"]["end_epoch"] = args.end_epoch
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["learn_rate"] = args.learn_rate
    cfg["train"]["resume"] = args.resume
    cfg["train"]["checkpoint_folder"] = args.checkpoint_folder
    cfg["train"]["validation"] = args.val
    cfg["train"]["log"] = args.log
    cfg["train"]["writer_folder"] = args.writer
    return cfg["train"]

def get_model(name, cfg):
    if name == "GHyperFormer":
        from model import GHyperFormer
        return GHyperFormer(cfg)

def get_lr_scheduler(lr_scheduler: str, optimizer, **kwargs):
    if lr_scheduler == "multistep":
        end_epoch = kwargs["end_epoch"]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[end_epoch//8, end_epoch//4, end_epoch//2])
    elif lr_scheduler == "cosineanneal":
        end_epoch = kwargs["end_epoch"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=1e-5)
    elif lr_scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    return scheduler

def get_optimizer(name, model, base_lr):
    if name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001)
    elif name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5, amsgrad=True)
    return optimizer

def get_logger(logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class DataAugmenter(nn.Module):
    def __init__(self):
        super(DataAugmenter,self).__init__()
        self.flip_dim = []
        self.zoom_rate = random.uniform(0.7, 1.0)
        self.sigma_1 = random.uniform(0.5, 1.5)
        self.sigma_2 = random.uniform(0.5, 1.5)
        self.image_zoom = Zoom(zoom=self.zoom_rate, mode="trilinear", padding_mode="constant")
        self.label_zoom = Zoom(zoom=self.zoom_rate, mode="nearest", padding_mode="constant")
        self.noisy = RandGaussianNoise(prob=1, mean=0, std=random.uniform(0, 0.33))
        self.blur = GaussianSharpen(sigma1=self.sigma_1, sigma2=self.sigma_2)
        self.contrast = AdjustContrast(gamma=random.uniform(0.65, 1.5))
        
    def forward(self, images, lables):
        with torch.no_grad():
            for b in range(images.shape[0]):
                image = images[b].squeeze(0)
                lable = lables[b].squeeze(0)
                if random.random() < 0.15:
                    image = self.image_zoom(image)
                    lable = self.label_zoom(lable)
                if random.random() < 0.5:
                    image = torch.flip(image, dims=(1,))
                    lable = torch.flip(lable, dims=(1,))
                if random.random() < 0.5:
                    image = torch.flip(image, dims=(2,))
                    lable = torch.flip(lable, dims=(2,))
                if random.random() < 0.5:
                    image = torch.flip(image, dims=(3,))
                    lable = torch.flip(lable, dims=(3,))
                if random.random() < 0.15:
                    image = self.noisy(image)
                if random.random() < 0.15:
                    image = self.blur(image)
                if random.random() < 0.15:
                    image = self.contrast(image)
                images[b] = image.unsqueeze(0)
                lables[b] = lable.unsqueeze(0)
            return images, lables

class Trainer():
    def __init__(self, model, args, logger, writer):
        self.model = model
        self.train_cfg = args
        self.logger = logger
        self.writer = writer
        
    def _train(self,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch: int
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_count = torch.cuda.device_count()
        device_ids = list(range(device_count))
        model = nn.DataParallel(model, device_ids)
        model.to(device)
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        loss_value = 0.
        for i, data in enumerate(data_loader):
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            data_aug = DataAugmenter().to(device)
            image_batch, label_batch = data["image"].to(device), data["label"].to(device)
            
            assert next(model.parameters()).device == image_batch.device, "Model and data must be on the same device"
            
            image_batch, label_batch = data_aug(image_batch, label_batch)
            pred = model(image_batch)
            loss = criterion(pred, label_batch)
            loss_value += loss.item()
            loss.backward()
            optimizer.step()
                
        scheduler.step()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        loss_value = loss_value / len(data_loader)
        self.writer.add_scalar("loss/train", loss_value, epoch)
        return loss_value
    
    def _train_val(self,
        data_loader,
        model,
        criterion,
        epoch
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_count = torch.cuda.device_count()
        device_ids = list(range(device_count))
        model = nn.DataParallel(model, device_ids)
        model.to(device)
        
        loss_value = 0.
        for i, data in enumerate(data_loader):
            image_batch, label_batch = data["image"].to(device), data["label"].to(device)
            pred = model(image_batch)
            loss = criterion(pred, label_batch)
            loss_value += loss.item()
        loss_value = loss_value / len(data_loader)
        self.writer.add_scalar("loss/train_val", loss_value, epoch)
        return loss_value
        
    def run(self):
        db_train = BraTS(self.train_cfg["dataset_folder"], self.train_cfg["list_folder"], mode="train")
        db_train_val = BraTS(self.train_cfg["dataset_folder"], self.train_cfg["list_folder"], mode="train_val")
        print("The length of train set is: {}".format(len(db_train)))
        
        base_lr = self.train_cfg["learn_rate"]
        num_workers = self.train_cfg["workers"]
        batch_size = self.train_cfg["batch_size"]
        start_epoch = 0
        end_epoch = self.train_cfg["end_epoch"]
        
        def worker_init_fn(worker_id):
            random.seed(self.train_cfg["seed"] + worker_id)
        
        train_loader = DataLoader(
            dataset=db_train, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            worker_init_fn=worker_init_fn
        )
        train_val_loader = DataLoader(
            dataset=db_train_val,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
        )
        
        dice_loss = DiceLoss(sigmoid=True)
        optimizer = get_optimizer(self.train_cfg["optimizer"], self.model, base_lr)
        scheduler = get_lr_scheduler(self.train_cfg["lr_scheduler"], optimizer, end_epoch=end_epoch)
        
        best_loss = np.inf
        if self.train_cfg["resume"]:
            checkpoint = torch.load(os.path.join(self.train_cfg["checkpoint_folder"], "checkpoint.pth.tar"))
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            
        
        self.logger.info(f"Start train from epoch = {start_epoch}")
        for epoch in range(start_epoch, end_epoch):
            self.model.train() 
            self.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            train_loss = self._train(train_loader, self.model, dice_loss, optimizer, scheduler, epoch)
            if (epoch + 1) % self.train_cfg["validation"] == 0:
                self.model.eval()
                with torch.no_grad():
                    train_val_loss = self._train_val(train_val_loader, self.model, dice_loss, epoch)
                    if train_val_loss < best_loss:
                        best_loss = train_val_loss
                        torch.save(self.model.state_dict(), os.path.join(self.train_cfg["bestmodel_folder"], "best_model.pkl"))
            checkpoint = dict(epoch=epoch, model=self.model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(checkpoint, os.path.join(self.train_cfg["checkpoint_folder"], "checkpoint.pth.tar"))
            self.logger.info(f"epoch {epoch}, train_loss = {train_loss}, train_val_loss = {train_val_loss}, lr={optimizer.param_groups[0]['lr']}.")
        
        self.writer.close()
        self.logger.info("Training finished!")        


if __name__ =="__main__":
    args = get_parse_args()
    init_random(args.seed)
    init_work_dir(args)
    net_cfg = ExperimentConfig[args.name]["net"]
    train_cfg = get_train_cfg(args)
    logger = get_logger(train_cfg["log"])
    writer = SummaryWriter(train_cfg["writer_folder"])
    model = get_model(args.model, net_cfg)
    trainer = Trainer(model, train_cfg, logger, writer)
    trainer.run()
