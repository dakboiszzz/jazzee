import torch
import torch.nn as nn

# -----BUILDING THE DISCRIMINATOR------

# Create a Ck Block, with k represents the out_channel
# Consiting of:
# - A conv layer, filter 4x4, stride 2, padding 1, bias = False, padding_mode 'reflect'
# - BatchNorm
# - Leaky ReLU (slope 0.2)

class DBlock(nn.Module):
    def __init__(self,in_channels, out_channels, bn = True, stride = 2):
        super().__init__()
        sequence = [
            nn.Conv2d(in_channels, out_channels, (4,4), stride, 1, bias = False, padding_mode = 'reflect'),
        ]
        if bn:
            sequence.append(
                nn.BatchNorm2d(out_channels)
            )
        sequence.append(nn.LeakyReLU(0.2))
        
        self.block = nn.Sequential(*sequence)
    def forward(self,x):
        return self.block(x)
        
# Discriminator -> C64 - C128 - C256 - C512 
# Few things to note here:
# - First layer: No BatchNorm
# - First layer: Doubling the channel for stacking the target with the input
# - Last layer: stride 1 (the author failed to mention this in the paper, but I've figured it out)
# - After the last layer: Another layer with out_channel = 1, stride 1 + Sigmoid
class Discriminator(nn.Module):
    def __init__(self,in_feat = 3,features = [64,128,256,512]):
        super().__init__()
        layers = []
        for feature in features:
            if feature == features[0]:
                # Double bc Discriminator take both input & target
                layers.append(DBlock(in_feat * 2, feature, bn = False))
            elif feature == features[-1]:
                # Change the stride for the C512 layer
                layers.append(DBlock(in_feat,feature, stride = 1))
            else:
                layers.append(DBlock(in_feat,feature))
            in_feat = feature
        # For the layer that comes after last layer
        layers.extend([
            nn.Conv2d(features[-1], 1, (4,4), 1, 1, bias = False, padding_mode= 'reflect'),
            nn.Sigmoid(),
        ])
        self.model = nn.Sequential(*layers)
        
    def forward(self,x,y):
        # Concat along the channel dim
        out = torch.cat([x,y], 1)
        return self.model(out)
    