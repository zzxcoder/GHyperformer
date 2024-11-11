#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:00:58 2024

@author: zzx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from functools import reduce
import math
import operator

from resnet import ResNet18_3D

class PatchEmbed3D(nn.Module):
    def __init__(self,
        img_size: Tuple[int],
        patch_size: Tuple[int],
        num_patches: int,
        in_channels: int,
        embed_dim: int,
        norm_layer: nn.Module = None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()
        
    def forward(self, x):
        _, _, D, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        
        # flatten: [B, C, PD, PH, PW] -> [B, C, PD*PH*PW], PD*PH*PW = number of patches
        # transpose: [B, C, PD*PH*PW] -> [B, PD*PH*PW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm_layer(x)
        return x


class Embedding3D(nn.Module):
    def __init__(self,
        img_size: Tuple[int],
        patch_size: Tuple[int],
        in_channels: int,
        embed_dim: int = None,
        drop_ratio: float = 0.,
        norm_layer: nn.Module = None
    ):
        super().__init__()
        embed_dim = reduce(operator.mul, img_size) * in_channels if embed_dim is None else embed_dim
        self.embed_dim = embed_dim
        num_patches = reduce(operator.mul, tuple(a // b for a, b in zip(img_size, patch_size)))
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            num_patches=num_patches,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.position_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))      
        self.dropout= nn.Dropout(p=drop_ratio)
        
    def forward(self, x):
        patch_embed = self.patch_embed(x)
        position_embed = self.position_embed
        embeddings = patch_embed + position_embed  
        embeddings = self.dropout(embeddings)
        return embeddings
        

class SEWeightBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=in_channels//reduction, 
            kernel_size=1, 
            padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(
            in_channels=in_channels//reduction, 
            out_channels=in_channels, 
            kernel_size=1, 
            padding=0
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight


class MSCA(nn.Module):
    """
        Multi-scale cross-attetion
    """
    def __init__(self,
        in_dim: int,
        reduction_dim: int,
        dilation_rates: List[int] = [1, 2, 3],
        conv_groups: List[int] = [1, 4, 8],
        stride: int = 1
    ):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_dim, 
                    out_channels=reduction_dim, 
                    kernel_size=1, 
                    bias=False
                ),
                nn.BatchNorm3d(reduction_dim),
                nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=reduction_dim, 
                    out_channels=reduction_dim//4, 
                    kernel_size=1, 
                    bias=False
                ),
                nn.BatchNorm3d(reduction_dim//4),
                nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
                nn.Conv3d(
                    in_channels=reduction_dim, 
                    out_channels=reduction_dim//4, 
                    kernel_size=3, 
                    stride=stride, 
                    padding=dilation_rates[0], 
                    dilation=dilation_rates[0], 
                    groups=conv_groups[0], 
                    bias=False
                ),
                nn.BatchNorm3d(reduction_dim//4),
                nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
                nn.Conv3d(
                    in_channels=reduction_dim, 
                    out_channels=reduction_dim//4, 
                    kernel_size=3,
                    stride=stride,
                    padding=dilation_rates[1],
                    dilation=dilation_rates[1],
                    groups=conv_groups[1],
                    bias=False
                ),
                nn.BatchNorm3d(reduction_dim//4),
                nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
                nn.Conv3d(
                    in_channels=reduction_dim, 
                    out_channels=reduction_dim//4, 
                    kernel_size=3,
                    stride=stride,
                    padding=dilation_rates[2],
                    dilation=dilation_rates[2],
                    groups=conv_groups[2],
                    bias=False
                ),
                nn.BatchNorm3d(reduction_dim//4),
                nn.ReLU(inplace=True)
        )
        
        self.se = SEWeightBlock3D(reduction_dim//4)
        
        self.weight_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_1.data.fill_(1.0)
        self.weight_2.data.fill_(1.0)
        self.weight_3.data.fill_(1.0)
        self.weight_4.data.fill_(1.0)
        self.softmax = nn.Softmax(dim=1)
        self.split_channel = reduction_dim // 4
        
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv1(self.features(x))
        x2 = self.conv2(self.features(x))
        x3 = self.conv3(self.features(x))
        x4 = self.conv4(self.features(x))
        
        
        x1_se = self.weight_1 * self.se(x1)
        x2_se = self.weight_2 * self.se(x2)
        x3_se = self.weight_3 * self.se(x3)
        x4_se = self.weight_4 * self.se(x4)
        
        features = torch.cat([x1, x2, x3, x4], dim=1)
        x_se = torch.cat([x1_se, x2_se, x3_se, x4_se], dim=1)
        
        features = features.view(batch_size, 4, self.split_channel, features.shape[2], features.shape[3], features.shape[4])
        attn_vecs = x_se.view(batch_size, 4, self.split_channel, 1, 1, 1)
        attn_vecs = self.softmax(attn_vecs)
        features_weight = features * attn_vecs
        for i in range(4):
            x_se_weight_fp = features_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat([x_se_weight_fp, out], dim=1)
        return out


class SelfAttension(nn.Module):
    def __init__(self,
        embed_dim: int,
        head_dim: int,
        qkv_bias: bool = False,
        qkv_scale: float = None,
        attn_drop_ratio: float = 0.,
        proj_drop_ratio: float = 0.
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.qkv_scale = qkv_scale if qkv_scale is not None else head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, head_dim * 3, qkv_bias)   
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(head_dim, head_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        
    def forward(self, x):
        # x: [B, N, D], B: batch size; N: number of patches; D: dimention of embedding.
        # qkv: [B, N, 3D]
        qkv = self.qkv(x)
        q = qkv[:, :, :self.head_dim]
        k = qkv[:, :, self.head_dim : self.head_dim * 2]
        v = qkv[:, :, self.head_dim * 2:]
        
        # transpose: [B, N, D] -> [B, D, N]
        # @: matrix multiply -> [B, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.qkv_scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # @: matrix multiply -> [B, N, D]
        x = attn @ v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        

class MultiHeadAttention(nn.Module):
    def __init__(self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        attn_list = [SelfAttension(embed_dim, head_dim) for _ in range(num_heads)]
        self.multi_head_attn = nn.ModuleList(attn_list)
        self.proj = nn.Linear(num_heads * head_dim, embed_dim)
        
    def forward(self, x):
        attn_scores = [attn(x) for attn in self.multi_head_attn]
        attn_scores = torch.cat(attn_scores, dim=-1)
        attn_scores = self.proj(attn_scores)
        return attn_scores



class Mlp(nn.Module):
    def __init__(self,
        in_features: int,
        hidden_dim: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop_ratio: float = 0.
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else in_features
        out_features = out_features if out_features is not None else in_features
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_features)
        self.dropout = nn.Dropout(drop_ratio)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class TransformerBlock3D(nn.Module):
    def __init__(self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        drop_ratio: float,
        attn_drop_ratio: float,
        drop_path_ratio: float,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.MSA = MultiHeadAttention(embed_dim, num_heads)
        self.MLP = Mlp(embed_dim, mlp_dim, drop_ratio=drop_ratio)
        self.norm2 = norm_layer(embed_dim)
        self.drop_path = nn.Dropout(drop_path_ratio)
    
    def forward(self, x):
        x = x + self.drop_path(self.MSA(self.norm1(self.drop_path(x))))
        x = x + self.drop_path(self.MLP(self.norm2(self.drop_path(x))))
        return x
    

class SingleConvBlock3D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size-1)//2
        )
        
    def forward(self, x):
        return self.conv(x)


class SingleDeconvBlock3D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
    
    def forward(self, x):
        return self.deconv(x)


class ConvBlock3D(nn.Sequential):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        conv = SingleConvBlock3D(in_channels, out_channels, kernel_size, stride, padding)
        bn = nn.BatchNorm3d(out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)     


class DeconvBlock3D(nn.Sequential):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3
    ):
        deconv = SingleDeconvBlock3D(in_channels, out_channels)
        conv = SingleConvBlock3D(out_channels, out_channels, kernel_size)
        bn = nn.BatchNorm3d(out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(deconv, conv, bn, relu)


class GDHA(nn.Module):
    def __init__(self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        in_channels: int,
        mlp_dim: int,
        num_heads: List[int],
        depths: List[int],       # Depth of each Transformer block
        drop_ratio: float = 0.,
        attn_drop_ratio: float = 0.,
        drop_path_ratio: float = 0.1,     # Stochastic depth rate. Default: 0.1
        norm_layer: nn.Module = nn.LayerNorm,
        with_msca: bool = True
    ):
        super().__init__()
        num_blocks = len(depths)
        self.img_size = (img_size, ) * 3
        self.patch_size = (patch_size, ) * 3
        self.embed_dim = embed_dim
        
        out_channels = in_channels #512
        self.msca = MSCA(in_channels, out_channels) if with_msca else None
        
        # split image into non-overlapping patches
        self.embedding = Embedding3D(self.img_size, self.patch_size, in_channels, embed_dim)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]
        self.transformer = nn.Sequential(*[
            TransformerBlock3D(embed_dim=embed_dim,
                  num_heads=num_heads[i],
                  mlp_dim=mlp_dim,
                  drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i]
            ) for i in range(num_blocks)
        ])
        self.norm_layer = norm_layer(embed_dim)
        
        self.align_layer = nn.Sequential(
            nn.Upsample(scale_factor=patch_size),
            nn.Conv3d(embed_dim, out_channels, kernel_size=1, padding=0, stride=1)
        )
        
        self.conv_gru = ConvGRU3D(in_channels, out_channels)
        
    
    def forward(self, x):
        a1 = self.msca(x) if self.msca is not None else x
        embedding = self.embedding(x)
        a2 = self.transformer(embedding)
        a2 = self.norm_layer(a2)
        B, n_patch, C = a2.size()
        D, H, W = (math.ceil(n_patch ** (1/3)), ) * 3
        a2 = a2.transpose(1, 2).contiguous().view(-1, C, D, H, W)
        a2 = self.align_layer(a2)
        y = self.conv_gru(a1, a2)
        return y
    
    
class ConvGRU3D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.conv_x_z = ConvBlock3D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_h_z = ConvBlock3D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_x_r = ConvBlock3D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_h_r = ConvBlock3D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_x_h = ConvBlock3D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_h_h = ConvBlock3D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = ConvBlock3D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, h):
        z_t = torch.sigmoid(self.conv_x_z(x) + self.conv_h_z(h))
        r_t = torch.sigmoid(self.conv_x_r(x) + self.conv_h_r(h))
        h_hat = torch.tanh(self.conv_x_h(x) + self.conv_h_h(torch.mul(r_t, h)))
        h_t = torch.mul(z_t, h) + torch.mul(1 - z_t, h_hat)
        y = self.conv_out(h_t)
        return y
    
    
class DecoderBlock3D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        with_gru: bool = True
    ):
        super().__init__()
        self.deconv = DeconvBlock3D(in_channels=in_channels, out_channels=out_channels)
        self.conv_gru = ConvGRU3D(out_channels, out_channels)
        self.conv1 = ConvBlock3D(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = ConvBlock3D(
            in_channels=out_channels, 
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.with_gru = with_gru
    
    def forward(self, x, h=None):
        x = self.deconv(x)
        if h is not None:
            if self.with_gru:
                x = self.conv_gru(x, h)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
        in_channels: int
    ):
        super().__init__()
        self.resnet = ResNet18_3D(in_channels)
        
    def forward(self, x):
        features = self.resnet(x)
        return features

    
class Decoder(nn.Module):
    def __init__(self,
        hidden_channels: List[int],
        head_channels: int
    ):
        super().__init__()
        num_blocks = len(hidden_channels)
        decoder_channels = hidden_channels
        in_channels = [head_channels] + decoder_channels[:-1]
        out_channels = decoder_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock3D(
                in_channels=in_ch,
                out_channels=out_ch
            ) for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        self.convgru_blocks = nn.ModuleList([
            ConvGRU3D(ch, ch) for ch in decoder_channels
        ])
        
        self.conv_cats = nn.ModuleList([
            ConvBlock3D(in_channels=in_ch, out_channels=out_ch, kernel_size=1) for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        self.deconv_fc = DeconvBlock3D(in_channels=out_channels[-1], out_channels=out_channels[-1])
    
    def forward(self, x, f):
        for decoder_block, convgru_block, conv_cat, fmap in zip(self.decoder_blocks, self.convgru_blocks, self.conv_cats, f): 
            x_decode = decoder_block(x)
            x_convgru = convgru_block(x_decode, fmap)
            x = torch.cat([x_convgru, x_decode], dim=1)
            x = conv_cat(x)
        x = self.deconv_fc(x)
        return x
    
    
class SegmentationHead(nn.Sequential):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int
    ):
        hidden_channels = [in_channels//2, in_channels//4, in_channels//8]
        conv3d_1 = ConvBlock3D(in_channels, hidden_channels[0], kernel_size, padding=1)
        conv3d_2 = ConvBlock3D(hidden_channels[0], hidden_channels[1], kernel_size, padding=1)
        conv3d_3 = ConvBlock3D(hidden_channels[1], hidden_channels[2], kernel_size, padding=1)
        fc = SingleConvBlock3D(hidden_channels[2], out_channels, kernel_size=3, padding=1)
        super().__init__(conv3d_1, conv3d_2, conv3d_3, fc)


class GHyperFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_cfg = config["encoder"]
        gdha_cfg = config["gdha"]
        decoder_cfg = config["decoder"]
        head_cfg = config["head"]
        self.encoder = Encoder(in_channels=encoder_cfg["in_channels"])
        self.gdha = GDHA(
            img_size=gdha_cfg["img_size"], 
            patch_size=gdha_cfg["patch_size"], 
            embed_dim=gdha_cfg["embed_dim"], 
            in_channels=gdha_cfg["in_channels"], 
            mlp_dim=gdha_cfg["mlp_dim"], 
            num_heads=gdha_cfg["num_heads"], 
            depths=gdha_cfg["depths"], 
            drop_ratio=gdha_cfg["drop_ratio"], 
            attn_drop_ratio=gdha_cfg["attn_drop_ratio"], 
            drop_path_ratio=gdha_cfg["drop_path_ratio"],
            with_msca=gdha_cfg["with_msca"]
        )
        self.decoder = Decoder(
            hidden_channels=decoder_cfg["hidden_channels"],
            head_channels=decoder_cfg["head_channels"]
        )
        self.head = SegmentationHead(
            in_channels=head_cfg["input_channels"],
            out_channels=head_cfg["output_channels"],
            kernel_size=head_cfg["kernel_size"]
        )
    
    def forward(self, x):
        x_, fmaps = self.encoder(x)
        trans_fmap = self.gdha(x_)
        features = self.decoder(trans_fmap, fmaps[::-1])
        y = self.head(features)
        return y
