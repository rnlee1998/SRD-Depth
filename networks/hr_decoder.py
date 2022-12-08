from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .hr_layers import *


class DepthDecoder(nn.Module):
    def __init__(self, ch_enc = [64,128,216,288,288], scales=range(4),num_ch_enc = [ 64, 64, 128, 256, 512 ], num_output_channels=1):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        
        # decoder
        self.convs = nn.ModuleDict()
        
        # feature fusion
        self.convs["f4"] = Attention_Module(self.ch_enc[4]  , num_ch_enc[4])
        self.convs["f3"] = Attention_Module(self.ch_enc[3]  , num_ch_enc[3])
        self.convs["f2"] = Attention_Module(self.ch_enc[2]  , num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.ch_enc[1]  , num_ch_enc[1])
        


        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                        self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        for i in range(4):
            if i<3:
                self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i]+1, self.num_output_channels)
            else:
                self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
            self.convs["uncerconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(16+32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24), nn.ELU(), Conv3x3(24,2,bias=False))
        self.delta_gen1[3].conv.weight.data.zero_()

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(32+64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48), nn.ELU(), Conv3x3(48,2,bias=False))
        self.delta_gen2[3].conv.weight.data.zero_()

        self.delta_gen3 = nn.Sequential(
            nn.Conv2d(64+128, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96), nn.ELU(), Conv3x3(96,2,bias=False))
        self.delta_gen3[3].conv.weight.data.zero_()

        self.delta = [self.delta_gen1, self.delta_gen2, self.delta_gen3]

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        feat={}
        feat[4] = self.convs["f4"](input_features[4])
        feat[3] = self.convs["f3"](input_features[3])
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = input_features[0]

        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
       
        feat_list = [x,features["X_04"],features["X_13"],features["X_22"]]
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))
        for i in range(3,0,-1):
            h, w = feat_list[i-1].shape[2:]
            high_stage = F.interpolate(input=feat_list[i],
                                   size=(h, w),
                                   mode='bilinear',
                                   align_corners=True)
            concat = torch.cat((high_stage,feat_list[i-1]),1)   #128+64 64+32 32+16
            delta = self.delta[i-1](concat) # 2 1 0

            _disp = self.bilinear_interpolate_torch_gridsample(outputs[("disp", i)], (h, w), delta)
            concat_feat = torch.cat((_disp,feat_list[i-1]),1)
            outputs[("disp", i-1)] = self.sigmoid(self.convs["dispconv{}".format(i-1)](concat_feat))
            outputs[("delta", i-1)] = delta

        # outputs[("disp", 0)] = self.sigmoid(self.convs["dispconv0"](x))                 #[12,1,192,640]
        # outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](features["X_04"]))  #[12,1,96,320]
        # outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv2"](features["X_13"]))  #[12,1,48,160]
        # outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))  #[12,1,24,80]

        # for i in self.scales:
        #     outputs[("uncer", 0)] = self.sigmoid(self.convs["uncerconv0"](x))
        #     outputs[("uncer", 1)] = self.sigmoid(self.convs["uncerconv1"](features["X_04"]))
        #     outputs[("uncer", 2)] = self.sigmoid(self.convs["uncerconv2"](features["X_13"]))
        #     outputs[("uncer", 3)] = self.sigmoid(self.convs["uncerconv3"](features["X_22"]))            
        return outputs

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w - 1) / s, (out_h - 1) / s]]]
                             ]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class Conv3x3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=1,
                 padding=1,
                 bias=True,
                 use_refl=True,
                 name=None):
        super().__init__()
        self.name = name
        if use_refl:
            # self.pad = nn.ReplicationPad2d(padding)
            self.pad = nn.ReflectionPad2d(padding)
        else:
            self.pad = nn.ZeroPad2d(padding)
        conv = nn.Conv2d(int(in_channels),
                         int(out_channels),
                         3,
                         dilation=dilation,
                         bias=bias)
        if self.name:
            setattr(self, self.name, conv)
        else:
            self.conv = conv

    def forward(self, x):
        out = self.pad(x)
        if self.name:
            use_conv = getattr(self, self.name)
        else:
            use_conv = self.conv
        out = use_conv(out)
        return out