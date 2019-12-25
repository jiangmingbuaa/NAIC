# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.resnext_ibn_a import resnext101_ibn_a
from .backbones.densenet_ibn_a import densenet121_ibn_a, densenet169_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.aognet.aognet_singlescale import aognet_singlescale
from .backbones.hacnn import resnet50_ibn_a_ha

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class GeM(nn.Module):
    def __init__(self, p=2, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p, requires_grad=True)
        # self.p = torch.ones(1)*p
        self.eps = eps

    def forward(self, x):
        # print(self.p)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        elif model_name == 'resnext101_ibn_a':
            self.base = resnext101_ibn_a(4,32)
        elif model_name == 'densenet121_ibn_a':
            self.base = densenet121_ibn_a()
        elif model_name == 'densenet169_ibn_a':
            self.base = densenet169_ibn_a()
        elif model_name == 'se_resnet101_ibn_a':
            self.base = se_resnet101_ibn_a()
        elif model_name == 'aognet':
            self.base = aognet_singlescale()
        elif model_name == 'resnet50_ibn_a_ha':
            self.base = resnet50_ibn_a_ha(last_stride)
        
        print('Loading ' + model_name + ' model......')
        print("model_path:" + model_path)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = GeM() 
        #self.gap = nn.Conv2d(2048, 2048, kernel_size=(16,8), stride=1, padding=0, bias=False, groups = 2048)
        #self.gap.apply(weights_init_classifier)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.use_oim = True
        # self.neck='no'
        if self.neck == 'no':
            if not self.use_oim:
                self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            # self.bottleneck = nn.BatchNorm1d(1664)  # densenet169
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

            # self.bottleneck_local = nn.BatchNorm1d(self.in_planes)
            # self.bottleneck_local.bias.requires_grad_(False)  # no shift
            # self.bottleneck_local.apply(weights_init_kaiming)

            if not self.use_oim:
                self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.classifier.apply(weights_init_classifier)
        
        # self.dropout = nn.Dropout(p=0.1)
        ###  fix param
        # print('fix some params')
        # for name, module in self.base._modules.items():
        #     if name in ["conv1"]: 
        #     #if name in ["conv1", "bn1", "relu", "maxpool","layer1", "layer2","layer3","layer4"]:
        #         for param in module.parameters():
        #             param.requires_grad = False
        ###

    def forward(self, x):
        
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        # global_feat, local_feat = self.base(x)
        # global_feat = self.gap(global_feat)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            # feat_local = self.bottleneck_local(local_feat)

        # feat = self.dropout(feat) 
        # feat = 1. * feat / (torch.norm(feat, 2, -1, keepdim=True).expand_as(feat) + 1e-12)

        if self.training:
            if self.use_oim:
                return feat, global_feat
            else:
                cls_score = self.classifier(feat)
                return cls_score, global_feat  # global feature for triplet loss
        else: 
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                # feat = 1. * feat / (torch.norm(feat, 2, -1, keepdim=True).expand_as(feat) + 1e-12)
                # feat_local = 1. * feat_local / (torch.norm(feat_local, 2, -1, keepdim=True).expand_as(feat_local) + 1e-12)
                # return torch.cat([feat, feat_local], dim=1)
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        # import pdb
        # pdb.set_trace()
        param_dict = torch.load(trained_path)
        for k, _ in param_dict.state_dict().items():
            # print(i)
            if 'classifier' in k:
                # print(i[0])
                continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])
        # for i in param_dict:
        #     if 'classifier' in i:
        #         continue
        #     self.state_dict()[i].copy_(param_dict[i])
