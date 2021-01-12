
"""
U-net, Attention U-net, and Edge Att. U-Net

Written for the task of medical image segmentation.

Started: 29/11/19

Latest updates: 
    
    26.02.2020:
        - increased the number of F_int in Edge_AG_1 from 32 to 256
        - same for Edge_AG_2 but from 64 to 128
        - this is an attempt to retain more intermediate edge feature maps
    
    25.02.20:
        - added if statements for deciding how many FMs to include in the Edge_AGs 
            depending on which level they are at
    24.02.20:        
        - built the edge block for extracting more features from low-level FMs
            - used for Edge_Attention_gate() 
            - for enhancing the learning of the overlap edges
    
    
author: Cal Mac
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

"""
----------------------------------------
Vanilla U-net architecture:
    Describe.
        This Unet has the learnable upsampling conv. removed to save GPU memory. 
        To do this, I ensured that the bottom conv block (self.Conv4, where the FMs go to 256)
        I've kept the FMs at 256 (ie conv_block(256, 256)) rather than go to 512 (256, 512).
        This way, the 256Fms from the 4th layer can now be combined with the 256Fms from the 
        3rd layer, so that we now have 512Fms in the 3rd layer, which is then double
        convolved to produce 128 Fms (line  )
    NAME: Unet
    
Notes: 
    - 5 layers deep = 13M parameters (13 ,394,242)
    - 4 layers deep = 3M parameters (3,361,090) 
        -- 10M more parameters for one layer deeper???
    - add learned upsampling: = 8,562,946
    
    TOTAL PARAMETERS:
        13.4M (for 5 levels)
----------------------------------------
"""
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        
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
Vanilla Attention U-net:
    Describe:
        - has 2 semantic edge outputs:
                -:: cytoplasm boundaries
                -:: nuclei boundaries
        - unlearned upsampling (using up)
        - uses deep supervision through the intermediate layers/skip connections
            -:: transforms FMs at each skip connection into a 1x1 - 1 edge map 
                for comparing with class-agnostic edgemap.
    
    TOTAL PARAMETERS:   
        13.75M for 5 levels (only 350k more than vanilla U-Net)
        
    Training time: 
        ~1hr for 500 images
----------------------------------------
"""
class Att_Unet(nn.Module):
    def __init__(self, img_ch=1, output_ch=3):
        super(Att_Unet,self).__init__()
        
        """ Encoder functions """        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(img_ch, 64)
        self.Conv2 = conv_block(64,  128)
        self.Conv3 = conv_block(128, 256)
        self.Conv4 = conv_block(256, 512)
        self.Conv5 = conv_block(512, 512)

        """ Decoder functions """
        self.Up5 = up(512, 512)
        self.Att4 = Attention_gate(512, 512, 256)
        self.Up_conv4 = conv_block(1024, 256)

        self.Up4 = up(256, 256)
        self.Att3 = Attention_gate(256, 256, 128)
        self.Up_conv3 = conv_block(512,  128)
        
        self.Up3 = up(128, 128)
        self.Att2 = Attention_gate(128, 128, 64)
        self.Up_conv2 = conv_block(256, 64)
        
        self.Up2 = up(64, 64)
        self.Att1 = Attention_gate(64, 64, 32)
        self.Up_conv1 = conv_block(128, 64)

        """ Output """
        self.Out = out_conv(64, output_ch)
        

    def forward(self, x):

        """ Encoder functions """
        x1 = self.Conv1(x)          # output from layer 1: 64Fms
        x2 = self.Maxpool(x1)  
        
        x2 = self.Conv2(x2)         # output from layer 2: 128Fms
        x3 = self.Maxpool(x2)
        
        x3 = self.Conv3(x3)         # output from layer 3: 256Fms
        x4 = self.Maxpool(x3)
        
        x4 = self.Conv4(x4)         # output from layer 4: 512Fms
        x5 = self.Maxpool(x4)
        
        x5 = self.Conv5(x5)

        """ Decoder functions """
        x_hat4 = self.Att4(x5, x4)
        u4 = self.Up5(x5)
        cat4 = torch.cat((x_hat4, u4),dim=1)   
        d4 = self.Up_conv4(cat4)

        x_hat3 = self.Att3(d4, x3)          
        u3 = self.Up4(d4)                        
        cat3 = torch.cat((x_hat3, u3),dim=1)     
        d3 = self.Up_conv3(cat3)                  

        x_hat2 = self.Att2(d3, x2)
        u2 = self.Up3(d3)
        cat2 = torch.cat((x_hat2, u2),dim=1)
        d2 = self.Up_conv2(cat2)

        x_hat1 = self.Att1(d2, x1) 
        u1 = self.Up2(d2)
        cat1 = torch.cat((x_hat1, u1),dim=1)
        d1 = self.Up_conv1(cat1)

        net_out = self.Out(d1)

        return net_out

"""
----------------------------------------
New Attention U-net:
    Describe:
        - same as Att Unet except that greater emphasis is placed on low-level features
        - this way, the AGs will be more tuned towards the cytoplasm/nuclei boundaries.
        - added edge_convs to deal with this: same as HED, CASENet, and DDS
    To preserve low-level features:
        - use encoder outputs (x) as gating signals instead of decoder features (d)
        - upsample deeper encoder feature maps to accomadate spatial dimensions
        
    Variable names are labelled according to their layer number.
    
    Date modified: 22.02.20
----------------------------------------
"""
class Edge_Att_Unet(nn.Module):
    def __init__(self, img_ch=1, output_ch=3):
        super(Edge_Att_Unet, self).__init__()
        
        """ Encoder functions """
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, 64)
        self.Conv2 = conv_block(64,  128)
        self.Conv3 = conv_block(128, 256)
        self.Conv4 = conv_block(256, 512)
        self.Conv5 = conv_block(512, 512)

        """ Decoder functions """
        self.Up5 = up(512, 512)
        self.AG_4 = Attention_gate(512, 512, 256)
        self.Up_conv4 = conv_block(1024, 256)

        self.Up4 = up(256, 256)
        self.AG_3 = Attention_gate(256, 256, 128)  
        self.Up_conv3 = conv_block(512, 128)
        
        # Only have Edge_AG's on levels 1 and 2, since deeper levels wont carry as much edge info
        # (also to save no. params!)
        self.Up3 = up(128, 128)
        self.Edge_AG_2 = Edge_Attention_gate(256, 128, 64, idx=2)  # x3 will carry 2x FMs as x2 (hence (256, 128) )
        self.Up_conv2 = conv_block(256, 64)
        
        self.Up2 = up(64, 64)
        self.Edge_AG_1 = Edge_Attention_gate(128, 64, 32, idx=1)   # and again with x2 and x1:
        self.Up_conv1 = conv_block(128, 64)

        """ Outputs """
        self.Out = out_conv(64, output_ch)


    def forward(self, x):

        """ Encoder path"""
        x1 = self.Conv1(x)          # output from layer 1: 64Fms
        x2 = self.Maxpool(x1)  
        
        x2 = self.Conv2(x2)         # output from layer 2: 128Fms
        x3 = self.Maxpool(x2)
        
        x3 = self.Conv3(x3)         # output from layer 3: 256Fms
        x4 = self.Maxpool(x3)
        
        x4 = self.Conv4(x4)         # output from layer 4: 512Fms
        x5 = self.Maxpool(x4)
       
        x5 = self.Conv5(x5)         # output from layer 5: 512Fms

        """ Decoder path """
        x_hat4 = self.AG_4(x5, x4)
        u4 = self.Up5(x5)
        cat4 = torch.cat((x_hat4, u4),dim=1)        
        d4 = self.Up_conv4(cat4)

        x_hat3 = self.AG_3(d4, x3)
        u3 = self.Up4(d4)                         
        cat3 = torch.cat((x_hat3, u3),dim=1)
        d3 = self.Up_conv3(cat3)

        x_hat2 = self.Edge_AG_2(x3, x2)
        u2 = self.Up3(d3)
        cat2 = torch.cat((x_hat2, u2),dim=1)
        d2 = self.Up_conv2(cat2)

        x_hat1 = self.Edge_AG_1(x2, x1) 
        u1 = self.Up2(d2)
        cat1 = torch.cat((x_hat1, u1),dim=1)
        d1 = self.Up_conv1(cat1)

        """ Output """
        net_out = self.Out(d1)

        return net_out



"""
----------------------------------------
Attention Block:
    Describe
    - adapted from Oktay et al code
    - added if statement to decide between AG for edge_net or baseline
    - edge_net incorporates low-level (ie xf) into gate for emphasising edges, 
      so upsample deeper xf to size of shallower xf
----------------------------------------
"""
class Attention_gate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super(Attention_gate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  
            nn.GroupNorm(32, F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.GroupNorm(32, F_int)
        )
        
        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1,1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        
    
    def forward(self, g, xf): 
        
        gl = self.W_g(g)
        xl = self.W_x(xf)
        g_height, g_width = g.size()[2], g.size()[3] # get dimensions of gating signal g for downsampling xl
        xl = F.interpolate(xl, size=(g_height, g_width), mode='bilinear', align_corners=True) # downsample xl to the size of gl: regular attention gate
       
        sum_block = torch.add(xl, gl)
        
        pre_psi = self.relu(sum_block)
        q_att = self.psi(pre_psi)
        q_att = self.sigmoid(q_att)
        alpha = self.up(q_att)  # upsample attention coefficients to size of xf
        
        x_hat = torch.mul(alpha, xf)

        return x_hat

  
"""
----------------------------------------
Edge_Attention Block:
    Describe
    - added residual type block to 
----------------------------------------
"""
class Edge_Attention_gate(nn.Module):
    def __init__(self, F_g, F_x, F_int, idx):
        super(Edge_Attention_gate, self).__init__()
        
        self.F_x = F_x
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  
            nn.GroupNorm(32, F_int)
            )
        
        """ A Residual block for learning the low-level feature maps in greater detail """
        # Takes the feature maps from the same level as the AG, sends them through 3 Conv-GN-Relu blocks
        # (added 25.02) if statement for determining how many FMs to have at a particular level (idx)
        if idx == 1:
            self.F_x = F_x * 4   # we want n* (64*n) the amount of features learned in the first level
            
        elif idx == 2: 
            self.F_x = F_x * 1    # we're happy with 128FMs passing through the edge_block in the second level
            
      
        self.edge_block = nn.Sequential(
                nn.Conv2d(F_x, self.F_x, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(32, self.F_x),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.F_x, self.F_x, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(32, self.F_x),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.F_x, F_int, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(32, F_int),
                nn.ReLU(inplace=True)
                )
        
        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1,1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        
    def forward(self, xn, xf): # replace g with xn, for the nth level down
        
        """ 
        gating signal           (xn): fine detail feature maps from next level down (l+1) 
        low-level feature maps  (xf): fine detail feature maps from level l
        """
        # Gating signal
        gl = self.W_g(xn)  # perform 1x1 conv to reduce g to F_int feature maps
        gl = self.up(gl)  # upsample deeper low-level features to the size of xl
        
        # Low-level feature maps
        xf_edge = self.edge_block(xf)     # send through edge block
        
        # Combine the two, then perform sequence of AG functions 
        sum_block = torch.add(xf_edge, gl)
        pre_psi = self.relu(sum_block)
        q_att = self.psi(pre_psi)
        alpha = self.sigmoid(q_att)
#        alpha = self.up(q_att)  # no upsampling required: we upsampled gating signal to the size of x

        # Multiply the coefficients with the original feature maps  
        x_hat = torch.mul(alpha, xf) # filtered feature maps 

        return x_hat

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


