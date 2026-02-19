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



# ------BUILDING THE GENERATOR ------- 

# Create the C block for Generator (bascially the same with the Discriminator)
# Few things to note:
# If it's upscaling : Use ConvTranspose2d rather than Conv2d
# Downward: Leaky ReLU | Upward: ReLU
# Some layers in the upward path has Dropout (0.5)
class GBlock(nn.Module):
    def __init__(self,in_channels, out_channels, bn = True, stride = 2, en = True, drop = True):
        super().__init__()
        sequence = [
            nn.Conv2d(in_channels, out_channels, (4,4), stride, 1, bias = False, padding_mode = 'reflect')
            if en
            else nn.ConvTranspose2d(in_channels, out_channels, (4,4), stride, 1, bias = False)
        ]
        if bn:
            sequence.append(
                nn.BatchNorm2d(out_channels)
            )
        if en:
            sequence.append(nn.LeakyReLU(0.2))
        else: 
            if drop:
                sequence.append(nn.Dropout(0.5))
            sequence.append(nn.ReLU())
        
        self.block = nn.Sequential(*sequence)
    def forward(self,x):
        return self.block(x)

# Create the Generator: Resembling the U-Net architecture
class Generator(nn.Module):
    def __init__(self,in_channels = 3, feature = 64):
        super().__init__()
        # --- ENCODER (Downsampling) ---
        # Down: C64-C128-C256-C512-C512-C512-C512-C512
        # I have to define them individually for skip connection
        self.down1 = GBlock(in_channels, feature, bn=False)             # Out: 64
        self.down2 = GBlock(feature, feature * 2)                       # Out: 128
        self.down3 = GBlock(feature * 2, feature * 4)                   # Out: 256
        self.down4 = GBlock(feature * 4, feature * 8)                   # Out: 512
        self.down5 = GBlock(feature * 8, feature * 8)                   # Out: 512
        self.down6 = GBlock(feature * 8, feature * 8)                   # Out: 512
        self.down7 = GBlock(feature * 8, feature * 8)                   # Out: 512
        
        # A bottleneck layer (No BatchNorm)
        self.bottleneck = GBlock(feature * 8, feature * 8, bn=False)    # Out: 512
        
        # --- DECODER (Upsampling) ---
        # Up (U-Net version): CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        # Notice the inputs: After down7 is concatenated, the input becomes 1024!
        self.up1 = GBlock(feature * 8, feature * 8, en=False, drop=True)          # In: 512 -> Out: 512
        self.up2 = GBlock(feature * 8 * 2, feature * 8, en=False, drop=True)      # In: 1024 -> Out: 512
        self.up3 = GBlock(feature * 8 * 2, feature * 8, en=False, drop=True)      # In: 1024 -> Out: 512
        self.up4 = GBlock(feature * 8 * 2, feature * 8, en=False)                 # In: 1024 -> Out: 512
        self.up5 = GBlock(feature * 8 * 2, feature * 4, en=False)                 # In: 1024 -> Out: 256
        self.up6 = GBlock(feature * 4 * 2, feature * 2, en=False)                 # In: 512 -> Out: 128
        self.up7 = GBlock(feature * 2 * 2, feature, en=False)                     # In: 256 -> Out: 64
        
        # Final Layer (Maps back to 3 RGB channels)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(feature * 2, 3, (4,4), 2, 1, bias=False), # In: 128 (64+64) -> Out: 3
            nn.Tanh()
        )
    def forward(self, x):
        # --- PASS DOWN (Save the skip connections!) ---
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        bottleneck = self.bottleneck(d7)
        
        # --- PASS UP & CONCATENATE ---
        # Concat along the channel dim
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d7], dim=1) 
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)
        
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)
        
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)
        
        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)
        
        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)
        
        return self.final_up(u7)