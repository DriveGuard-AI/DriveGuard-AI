"""Frozen ResNet18 with explicit local weights and whole-ROI preprocessing.

This compact implementation expresses the ResNet18 architecture of He et al.
(https://arxiv.org/abs/1512.03385), using torchvision's parameter names for strict
ImageNet V1 state-dict compatibility. No torchvision code or dependency is used.
Reference layout: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
"""
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch import nn

from inference.experts.assets import file_sha256
from training.common.device import select_device

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def encoder_metadata(weights_sha256):
    return dict(architecture='resnet18', weights='IMAGENET1K_V1', weights_sha256=weights_sha256,
                output_dim=512, preprocessing=dict(version='driver-roi-letterbox-v1', size=[224,224],
                    color='RGB', interpolation='bilinear', padding=[0,0,0],
                    mean=[.485,.456,.406], std=[.229,.224,.225]),
                sampling=dict(hz=25, offsets=[12,24], max_age_ms=80))


def preprocess(bgr, driver_roi):
    """Include every ROI pixel (floor/ceil bounds), bilinear resize, black pad."""
    if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
            or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1):
        raise ValueError('Expected uint8 HWC BGR image')
    if (not isinstance(driver_roi, (list,tuple)) or len(driver_roi) != 4
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in driver_roi)):
        raise ValueError('Invalid driver ROI')
    x0,y0,x1,y1 = driver_roi
    if not (0 <= x0 < x1 <= bgr.shape[1] and 0 <= y0 < y1 <= bgr.shape[0]):
        raise ValueError('Driver ROI outside image')
    crop = bgr[math.floor(y0):math.ceil(y1), math.floor(x0):math.ceil(x1), ::-1]
    height,width = crop.shape[:2]
    scale = 224/max(height,width)
    size = (max(1,round(width*scale)), max(1,round(height*scale)))
    rgb = Image.fromarray(np.ascontiguousarray(crop)).resize(size, Image.Resampling.BILINEAR)
    canvas = np.zeros((224,224,3),np.uint8)
    left,top = (224-size[0])//2,(224-size[1])//2
    canvas[top:top+size[1],left:left+size[0]] = np.asarray(rgb)
    return np.ascontiguousarray(((canvas.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1))


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming,outgoing,3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU()
        self.downsample = (nn.Sequential(nn.Conv2d(incoming,outgoing,1,stride=stride,bias=False),
                                        nn.BatchNorm2d(outgoing))
                           if incoming != outgoing or stride != 1 else None)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        branch = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return self.relu(branch+residual)


class ResNet18(nn.Module):
    """Full 1000-class weight layout; forward returns the 512-D pooled features."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3,stride=2,padding=1)
        incoming = 64
        for index,outgoing in enumerate((64,128,256,512),1):
            stride = 1 if index == 1 else 2
            setattr(self,'layer'+str(index),nn.Sequential(ResidualBlock(incoming,outgoing,stride),
                                                        ResidualBlock(outgoing,outgoing,1)))
            incoming = outgoing
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512,1000)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for index in range(1,5):
            x = getattr(self,'layer'+str(index))(x)
        return self.avgpool(x).flatten(1)


class FrozenEncoder:
    def __init__(self, weights, weights_sha256, backend='cpu', device_index=0):
        path = Path(weights).resolve(strict=True)
        if not isinstance(weights_sha256,str) or re.fullmatch('[0-9a-f]{64}',weights_sha256) is None:
            raise ValueError('Expected lowercase SHA256')
        stamp = path.stat()
        if file_sha256(path) != weights_sha256:
            raise ValueError('Weights SHA256 mismatch')
        state = torch.load(path,map_location='cpu',weights_only=True)
        after = path.stat()
        if (stamp.st_size,stamp.st_mtime_ns,stamp.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Weights changed during verification/loading')
        self.model = ResNet18()
        self.model.load_state_dict(state,strict=True)
        del state
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = select_device(backend,device_index)
        self.model.to(self.device)
        self.metadata = encoder_metadata(weights_sha256)

    def encode(self, batch):
        if not isinstance(batch,np.ndarray) or batch.dtype != np.float32 or batch.shape[1:] != (3,224,224):
            raise ValueError('Expected normalized float32 [B,3,224,224] batch')
        with torch.inference_mode():
            return self.model(torch.from_numpy(batch).to(self.device)).cpu().numpy().astype(np.float32,copy=False)
