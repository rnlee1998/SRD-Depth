from networks import resnet_encoder,pose_decoder,mpvit,swin_encoder,mobilenet_encoder
from torchvision.models import resnet50,resnet18,resnet101,densenet161,resnext101_32x8d,resnext50_32x4d
from thop import profile
import torch 
import networks
from torchsummaryX import summary

model = resnet101()
input = torch.randn(1, 3, 352, 1120)
summary(model, input)
# flops, params = profile(model, inputs=(input, ))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')