
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .swin_transformer import *
import copy
import random

class SwinEncoder(nn.Module):
    def __init__(self,mode="swin_tiny",pretrained=True):
        super().__init__() 

        if mode=="swin_tiny":
            pretrain_img_size=224
            embed_dim=96
            depths=[2, 2, 6, 2]
            num_heads=[ 3, 6, 12, 24 ]
            window_size=7
            drop_path_rate=0.2
            num_ch_enc = [96, 96, 192, 384, 768]
        elif mode=="swin_small":
            pretrain_img_size=224
            embed_dim=96
            depths=[2, 2, 18, 2]
            num_heads=[ 3, 6, 12, 24 ]
            window_size=7
            drop_path_rate=0.2
            num_ch_enc = [96, 96, 192, 384, 768]
        elif mode=="swin_base":
            pretrain_img_size=384#224
            embed_dim=128
            depths=[2, 2, 18, 2]
            num_heads=[ 4, 8, 16, 32]
            window_size=7
            drop_path_rate=0.2
            num_ch_enc = [128,128,256,512,1024]
        self.mode = mode
        self.encoder = SwinTransformer(pretrain_img_size=pretrain_img_size,
                                        embed_dim=embed_dim, 
                                        depths=depths, 
                                        num_heads=num_heads,
                                        window_size=window_size,
                                        drop_path_rate=drop_path_rate,)
        self.num_ch_enc = np.array(num_ch_enc)
        if pretrain_img_size==224 and pretrained:
            self.encoder.init_weights("models/%s_patch4_window7_224_22k.pth"%(mode))
        elif pretrain_img_size==384 and pretrained: 
            self.encoder.init_weights("models/%s_patch4_window12_384_22k.pth"%(mode))  
    
    def forward(self, x):                  
        return  self.encoder(x)