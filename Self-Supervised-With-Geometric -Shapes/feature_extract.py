import numpy as np
import torch
import torch.nn as nn

class extract_features(nn.Module):

    def __init__(self, in_channels, output_channels):
        super(extract_features, self).__init__()
        self.in_channels = in_channels
        self.output_channels=output_channels
        self.dp = nn.Dropout(0.5)

        expansion = 1
        self.max=torch.max
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * expansion, 64, 1, 1),
            nn.GroupNorm(64, 64), nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 1, 1),
            nn.GroupNorm(128, 128), nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 1, 1),
            nn.GroupNorm(256, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 1, 1),
            nn.GroupNorm(256, 256),
            nn.ReLU(inplace=True), 

            nn.Conv2d(256, 512, 1, 1),nn.ReLU(inplace=True)
            )

    def forward(self, all_objects_cube):
        #print('all_objects_cube',all_objects_cube.shape)

        out = self.conv1(all_objects_cube)
        out=self.dp(out)
        out=self.max(out,2).values

        return out

