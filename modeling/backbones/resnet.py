# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None,sub_sample=False,bn_layer=True):
        super(NonLocalBlock, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
                
        self.g = nn.Conv2d(in_channels=self.in_channels,out_channels=self.inter_channels,
                            kernel_size=1,stride=1,padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels,out_channels=self.in_channels,
                            kernel_size=1,stride=1,padding=0),
                    nn.BatchNorm2d(self.in_channels))
            nn.init.constant_(self.W[1].weight,0)
            nn.init.constant_(self.W[1].bias,0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels,out_channels=self.in_channels,
                            kernel_size=1,stride=1,padding=0)
            nn.init.constant_(self.W.weight,0)
            nn.init.constant_(self.W.bias,0)
        
        self.theta = nn.Conv2d(in_channels=self.in_channels,out_channels=self.inter_channels,
                            kernel_size=1,stride=1,padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels,out_channels=self.inter_channels,
                            kernel_size=1,stride=1,padding=0)
                            
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2,2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2,2)))
        #self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        N = x.size(0)
        gx = self.g(x).view(N,self.inter_channels,-1)
        gx = gx.permute(0,2,1)
        thetax = self.theta(x).view(N,self.inter_channels,-1)
        thetax = thetax.permute(0,2,1)
        phix = self.phi(x).view(N,self.inter_channels,-1)
        f = torch.matmul(thetax,phix)
        f_div_C = F.softmax(f,dim=-1)
        
        y = torch.matmul(f_div_C,gx)
        y = y.permute(0,2,1).contiguous()
        y = y.view(N,self.inter_channels,*x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class _BatchAttNorm(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_BatchAttNorm, self).__init__(num_features, eps, momentum, affine)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight_readjust = nn.Parameter(torch.Tensor(num_features))
        self.bias_readjust = nn.Parameter(torch.Tensor(num_features))
        self.weight_readjust.data.fill_(0)
        self.bias_readjust.data.fill_(-1)
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        attention = self.sigmoid(self.avg(input) * self.weight_readjust + self.bias_readjust)
        bn_w = self.weight * attention

        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training, self.momentum, self.eps)
        out_bn = out_bn * bn_w + self.bias

        return out_bn

class BAN2d(_BatchAttNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = BAN2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = BAN2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = BAN2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = BAN2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # self.bn3 = BAN2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1 = BAN2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

        ### non-local
        # self.non_local = NonLocalBlock(in_channels=512,inter_channels=256)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                # BAN2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.non_local(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.non_local(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            # print(i,param_dict[i].size())
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

