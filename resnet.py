#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:10:49 2024

@author: zzx
"""
import torch.nn as nn
import torch.nn.functional as F  

class BasicBlock(nn.Module):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels)
        )
        
        if in_channels != out_channels or stride != 2:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.skip = nn.Identity()
            
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = self.skip(x)
        x = self.block(x)
        x = self.relu(x + residual)
        return x

inplanes = [64, 128, 256, 512]
        
class ResNet18_3D(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                inplanes[0], 
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.BatchNorm3d(inplanes[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        
    def _make_layer(self, in_channels, out_channels, n_blocks, stride=1):
        layer_list = []
        layer_list.append(BasicBlock(in_channels, out_channels, stride))
        for i in range(1, n_blocks):
            layer_list.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layer_list)
    
    def forward(self, x):
        features = []
        x = self.stem(x)        # 16 x 64 x 64 x 64
        features.append(x)
        x = self.layer1(x)      # 64 x 64 x 64 x 64
        x = self.layer2(x)      # 128 x 32 x 32 x 32
        features.append(x)
        x = self.layer3(x)      # 256 x 16 x 16 x 16
        features.append(x)
        x = self.layer4(x)      # 512 x 8 x 8 x 8
        # x = self.pool(x)
        return x, features
       
