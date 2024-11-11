#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:53:13 2024

@author: zzx
"""

def get_ghperformer_config():
    return {
        "encoder": {
            "in_channels": 4	# number of modalities
        },
        "gdha": {
            "img_size": 8,
            "patch_size": 4,
            "embed_dim": 128,
            "in_channels": 512,
            "mlp_dim": 256,
            "num_heads": [2, 4, 8],
            "depths": [2, 2, 2],
            "drop_ratio": 1.0,
            "attn_drop_ratio": 1.0,
            "drop_path_ratio": 1.0,
            "with_msca": True
        },
        "decoder": {
            "hidden_channels": [256, 128, 64],
            "head_channels": 512
        },
        "head": {
            "input_channels": 64,
            "output_channels": 3,   # number of classes
            "kernel_size": 3
        }
    }
    

def get_train_config():
    return {
        "model_name": None,
        "dataset_folder": None,
        "list_folder": None,
        "experiment_folder": None,
        "bestmodel_folder": None,
        "seed": 1,
        "workers": 1,
        "end_epoch": 500,
        "batch_size": 1,
        "learn_rate": 1e-4,
        "resume": False,
        "checkpoint_folder": None,
        "validation": 1
    }


def get_test_config():
    return {
        "model_name": None,
        "dataset_folder": None,
        "list_folder": None,
        "experiment_folder": None,
        "bestmodel_folder": None,
        "segmentation_folder": None
    }

ExperimentConfig = {
    "GHyperformer_Adam_CosineAnnealingLR": {
        "name": "GHyperformer",
        "net": get_ghperformer_config(),
        "train": get_train_config() | dict(optimizer="Adam", lr_scheduler="cosineanneal"),
        "test": get_test_config()
    }
}
