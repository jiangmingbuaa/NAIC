import torch
import torch.nn as nn
from torch.nn import functional as F
import math


__all__ = ['ResNet_IBN_HA', 'resnet50_ibn_a_ha', 'resnet101_ibn_a_ha',
           'resnet152_ibn_a_ha']

################ ha block
class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class InceptionB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(in_channels, mid_channels * 2, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y



class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        # scaling conv
        x = self.conv2(x)
        return x

class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(
            torch.tensor(
                [0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float
            )
        )

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels, learn_region):
        super(HarmAttn, self).__init__()
        self.learn_region = learn_region
        self.soft_attn = SoftAttn(in_channels)
        if self.learn_region:
            self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = None
        if self.learn_region:
            theta = self.hard_attn(x)
        return y_soft_attn, theta 
################

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet_IBN_HA(nn.Module):

    def __init__(self, last_stride, block, layers, learn_region=False, use_gpu=True):
        self.learn_region = learn_region
        self.use_gpu = use_gpu
        scale = 64
        self.inplanes = scale
        super(ResNet_IBN_HA, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale*8, layers[3], stride=last_stride)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        ########## ha
        self.ha1 = HarmAttn(512, self.learn_region)
        self.ha2 = HarmAttn(1024, self.learn_region)
        self.ha3 = HarmAttn(2048, self.learn_region)
        if self.learn_region:
            self.init_scale_factors()
            self.local_conv1 = InceptionB(256, 512)
            self.local_conv2 = InceptionB(512, 1024)
            self.local_conv3 = InceptionB(1024, 2048)
            self.fc_local = nn.Linear(2048*4, 2048)
            # self.reduction1 = nn.Linear(2048, 512)
            # self.reduction2 = nn.Linear(2048, 512)
            # self.reduction3 = nn.Linear(2048, 512)
            # self.reduction4 = nn.Linear(2048, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_scale_factors(self):
        self.scale_factors = []
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
    
    def stn(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform_theta(self, theta_i, region_idx):
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,:,:2] = scale_factors
        theta[:,:,-1] = theta_i
        if self.use_gpu:
            theta = theta.cuda()
        return theta

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)

        x1 = self.layer2(x)
        x1_attn, x1_theta = self.ha1(x1)
        x1_out = x1*x1_attn
        if self.learn_region:
            x1_local_list = []
            for region_idx in range(4):
                x1_theta_i = x1_theta[:, region_idx, :]
                x1_theta_i = self.transform_theta(x1_theta_i, region_idx)
                x1_trans_i = self.stn(x, x1_theta_i)
                x1_trans_i = F.upsample(x1_trans_i, (24, 28), mode='bilinear', align_corners=True)
                x1_local_i = self.local_conv1(x1_trans_i)
                x1_local_list.append(x1_local_i)

        x2 = self.layer3(x1_out)
        x2_attn, x2_theta = self.ha2(x2)
        x2_out = x2*x2_attn
        if self.learn_region:
            x2_local_list = []
            for region_idx in range(4):
                x2_theta_i = x2_theta[:, region_idx, :]
                x2_theta_i = self.transform_theta(x2_theta_i, region_idx)
                x2_trans_i = self.stn(x1_out, x2_theta_i)
                x2_trans_i = F.upsample(x2_trans_i, (12, 14), mode='bilinear', align_corners=True)
                x2_local_i = x2_trans_i + x1_local_list[region_idx]
                x2_local_i = self.local_conv2(x2_local_i)
                x2_local_list.append(x2_local_i)

        x3 = self.layer4(x2_out)
        x3_attn, x3_theta = self.ha3(x3)
        x3_out = x3*x3_attn
        if self.learn_region:
            x3_local_list = []
            for region_idx in range(4):
                x3_theta_i = x3_theta[:, region_idx, :]
                x3_theta_i = self.transform_theta(x3_theta_i, region_idx)
                x3_trans_i = self.stn(x2_out, x3_theta_i)
                x3_trans_i = F.upsample(x3_trans_i, (6, 7), mode='bilinear', align_corners=True)
                x3_local_i = x3_trans_i + x2_local_list[region_idx]
                x3_local_i = self.local_conv3(x3_local_i)
                x3_local_list.append(x3_local_i)

        if self.learn_region:
            x_local_list = []
            for region_idx in range(4):
                x_local_i = x3_local_list[region_idx]
                x_local_i = F.avg_pool2d(x_local_i, x_local_i.size()[2:]).view(x_local_i.size(0), -1)
                # if region_idx==0:
                #     x_local_i = self.reduction1(x_local_i)
                # elif region_idx==1:
                #     x_local_i = self.reduction2(x_local_i)
                # elif region_idx==2:
                #     x_local_i = self.reduction3(x_local_i)
                # else:
                #     x_local_i = self.reduction4(x_local_i)
                x_local_list.append(x_local_i)
            x_local = torch.cat(x_local_list, 1)
            x_local = self.fc_local(x_local)
            return x3_out, x_local
        else:
            return x3_out
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def resnet50_ibn_a_ha(last_stride, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN_HA(last_stride, Bottleneck_IBN, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_ibn_a_ha(last_stride, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN_HA(last_stride, Bottleneck_IBN, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_ibn_a_ha(last_stride, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN_HA(last_stride, Bottleneck_IBN, [3, 8, 36, 3], **kwargs)
    return model