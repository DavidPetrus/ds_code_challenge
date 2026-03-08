import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from absl import flags

FLAGS = flags.FLAGS

class PoolClassifier(nn.Module):
    def __init__(self):
        super(PoolClassifier, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2), nn.SiLU(), # 417
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2), nn.InstanceNorm2d(32, affine=True), nn.SiLU(), # 209
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.SiLU(), # 105
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.InstanceNorm2d(128, affine=True), nn.SiLU(), # 53
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), nn.SiLU(), # 27
            nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=2), nn.InstanceNorm2d(512, affine=True), nn.SiLU(), # 14
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2), nn.SiLU(), # 7
            nn.Conv2d(in_channels=256, out_channels=128, padding=1, kernel_size=3, stride=2), nn.InstanceNorm2d(128, affine=True), nn.SiLU(), # 4
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2), # 2
        )

        self.head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.cnn(x) # bs, 3, 417, 417 -> bs, 64, 1, 1
        result = self.head(features.squeeze())

        return result.squeeze() # bs