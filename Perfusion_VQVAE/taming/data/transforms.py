import numpy as np
import torch
from monai import transforms

def get_transforms_DSC():
    val_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["dsc_signal"], channel_dim=0),
            transforms.ToTensord(keys=["dsc_signal", "dsc_signal_maxv"], dtype=torch.float32),
        ]
    )
    return val_transform

