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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2), nn.SiLU(), # 208
            nn.ReflectionPad2d((1,0,1,0)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2), nn.InstanceNorm2d(32, affine=True), nn.SiLU(), # 104
            nn.ReflectionPad2d((1,0,1,0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.SiLU(), # 52
            nn.ReflectionPad2d((1,0,1,0)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.InstanceNorm2d(128, affine=True), nn.SiLU(), # 26
            nn.ReflectionPad2d((1,0,1,0)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), nn.SiLU(), # 13
            nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=2), nn.InstanceNorm2d(512, affine=True), nn.SiLU(), # 6
            nn.ReflectionPad2d((1,0,1,0)),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=2), nn.SiLU(), # 3
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3), # 1
        )

        self.head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.cnn(x) # bs, 3, 417, 417 -> bs, 64, 1, 1
        result = self.head(features.squeeze())

        return result.squeeze() # bs


class RequestPredictor(nn.Module):
    def __init__(self):
        super(RequestPredictor, self).__init__()

        self.add_neighbors = False
        if self.add_neighbors:
            self.net = nn.Sequential(nn.Linear(26*7, 512), nn.SiLU(), nn.Linear(512, 64), nn.SiLU(), nn.Linear(64, 2))
        else:
            self.net = nn.Sequential(nn.Linear(26, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)