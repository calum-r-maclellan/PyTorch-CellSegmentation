
"""
Different ideas for the edge correcting/post-processing network for cell 
segmentation.


Started: 30/03/2020.

Latest updates: 
    
    
author: calmac
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



"""
----------------------------------------
Edge detection/segmentation network (Unet_Seg).

This is for taking an input image and through supervised learning will segment the 
edges in the image using the given ground truth edge maps.

Output is a binary edge map indicating pixels net thinks are cytoplasm edges.
        
----------------------------------------
"""
class Unet_Seg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet_Seg, self).__init__()
        
        """ Layer 4 at 256 FMs: 3.36M params 
            Layer 5 at 512Fms:  13.3M params: OURS  
        """
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_channels, ch_out=64)         # Out1: Dim=512x512
        self.Conv2 = conv_block(64,  128)                       # Out3: Dim=256x256
        self.Conv3 = conv_block(128, 256)                       # Out5: Dim=128x128
        self.Conv4 = conv_block(256, 512)                       # Out7: Dim=64x64
        self.Conv5 = conv_block(512, 512)
#        
        self.Up5 = up(512, 512)
        self.Up_conv4 = conv_block(1024, 256)
         
        self.Up4 = up(256,  256)                            
        self.Up_conv3 = conv_block(512,  128)
        
        self.Up3 = up(128,  128)
        self.Up_conv2 = conv_block(256,  64)
        
        self.Up2 = up(64,  64)
        self.Up_conv1 = conv_block(128,  64)

        self.Out = out_conv(64, out_channels)

    def forward(self, x):
        
        """ Contracting path """
        # only thing changing dimensions is max pooling since  
        # convolution blocks have padding=1.
        
        x1 = self.Conv1(x)             # 1st layer: 1 -> 64Fms 
                               
        x2 = self.Maxpool(x1)           
        x2 = self.Conv2(x2)            # 2nd layer: 64 -> 128Fms 
                               
        x3 = self.Maxpool(x2)           
        x3 = self.Conv3(x3)            # 3rd layer: 128 -> 256Fms

        x4 = self.Maxpool(x3)           
        x4 = self.Conv4(x4)            # 4th layer: 256 -> 256Fms/512Fms for 5 layers
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)            # 5th layer: 512 -> 512Fms
#        
        """ Expanding path """
        
        u4 = self.Up5(x5)                 # only dimensions doubled, Fms still 256 
        cat4 = torch.cat((x4,u4), dim=1)   # concat 256 with 256 to give 512Fms
        d4 = self.Up_conv4(cat4)           # 3rd layer: 512 -> 128Fms

        u3 = self.Up4(d4)              # only dimensions doubled, Fms still 256 
        cat3 = torch.cat((x3,u3), dim=1)   # concat 256 with 256 to give 512Fms
        d3 = self.Up_conv3(cat3)           # 3rd layer: 512 -> 128Fms

        u2 = self.Up3(d3)                
        cat2 = torch.cat((x2,u2), dim=1)   # concat 128 with 128 to give 256Fms
        d2 = self.Up_conv2(cat2)           # 2nd layer: 256 -> 64Fms

        u1 = self.Up2(d2)                
        cat1 = torch.cat((x1,u1), dim=1)   # concat 64 with 64 to give 128Fms
        d1 = self.Up_conv1(cat1)           # 1st layer: 128 -> 64Fms

        net_out = self.Out(d1)              # Output: 64 -> 1Fm 

        return net_out         # raw scores 

"""
----------------------------------------
Correction Unet (Unet_Cor): Post-processing 

This is for taking the output binary edge map produced from Unet_Seg, which will
have broken edges, and learns to correct these edges to form a map similar to the 
GT edge map. 

No need for 5 levels of encoder-decoders: 
    - try 4 first: n. params = roughly 3.5M, 10M less than with 5 levels
----------------------------------------
"""
class Unet_Corr(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet_Corr, self).__init__()
        
        """ Layer 4 at 256 FMs: 3.36M params """
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_channels, ch_out=64)         # Out1: Dim=512x512
        self.Conv2 = conv_block(64,  128)                       # Out3: Dim=256x256
        self.Conv3 = conv_block(128, 256)                       # Out5: Dim=128x128
        self.Conv4 = conv_block(256, 256)                       # Out7: Dim=64x64
#        
        self.Up4 = up(256,  256)                            
        self.Up_conv3 = conv_block(512,  128)
        
        self.Up3 = up(128,  128)
        self.Up_conv2 = conv_block(256,  64)
        
        self.Up2 = up(64,  64)
        self.Up_conv1 = conv_block(128,  64)

        self.Out = out_conv(64, out_channels)

    def forward(self, x):
        
        """ Contracting path """
        x1 = self.Conv1(x)             # 1st layer: 1 -> 64Fms 
                               
        x2 = self.Maxpool(x1)           
        x2 = self.Conv2(x2)            # 2nd layer: 64 -> 128Fms 
                               
        x3 = self.Maxpool(x2)           
        x3 = self.Conv3(x3)            # 3rd layer: 128 -> 256Fms

        x4 = self.Maxpool(x3)           
        x4 = self.Conv4(x4)            # 4th layer: 256 -> 256Fms
        
       
        """ Expanding path """
        u3 = self.Up4(x4)                   # only dimensions doubled, Fms still 256 
        cat3 = torch.cat((x3,u3), dim=1)    # concat 256 with 256 to give 512Fms
        d3 = self.Up_conv3(cat3)            # 3rd layer: 512 -> 128Fms

        u2 = self.Up3(d3)                
        cat2 = torch.cat((x2,u2), dim=1)    # concat 128 with 128 to give 256Fms
        d2 = self.Up_conv2(cat2)            # 2nd layer: 256 -> 64Fms

        u1 = self.Up2(d2)                
        cat1 = torch.cat((x1,u1), dim=1)    # concat 64 with 64 to give 128Fms
        d1 = self.Up_conv1(cat1)            # 1st layer: 128 -> 64Fms

        net_out = self.Out(d1)              # Output: 64 -> 1Fm 

        return net_out         # raw scores 


"""
----------------------------------------
EncoderBlock:
    Describe
----------------------------------------
"""
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

"""
----------------------------------------
Upsample + Conv:
    Describe
    This bit:
        - doubles the dimension of the FMs (bilinear interp.),
        - halves the amount of FMs (2x2 conv)
        
Two alterative versions:
    - up_conv(): includes learnable parameters, by halving the FMs upon upsampling
    - up(): basic version, for reducing parameter number. 
----------------------------------------
"""
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.GroupNorm(32, ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class up(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up,self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,x):
        x = self.up(x)
        return x
    
"""
----------------------------------------
Single convolution for the output segmentation map:
    Describe
    
----------------------------------------
"""
class out_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(out_conv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self,x):
        x = self.conv(x)
        return x

"""
----------------------------------------
Weight initialisation schemes:
    Describe
    
----------------------------------------
"""

def init_weights(net, init_type='normal', gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('GroupNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


