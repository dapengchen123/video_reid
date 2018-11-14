from __future__ import absolute_import
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
import torchvision

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, dropout=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrain) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)

        conv0 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.kaiming_normal(conv0.weight, mode='fan_out')

        self.conv0 = conv0
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.dropout = dropout
            self.has_embedding = num_features > 0

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                self.num_features = out_planes

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, imgs, motions, mode):

        img_size = imgs.size()
        motion_size = motions.size()
        batch_sz = img_size[0]
        seq_len = img_size[1]
        imgs = imgs.view(-1, img_size[2], img_size[3], img_size[4])
        motions = motions.view(-1, motion_size[2], motion_size[3], motion_size[4])
        motions = motions[:, 1:3]

        for name, module in self.base._modules.items():

            if name == 'conv1':
                x = module(imgs) + self.conv0(motions)
                continue

            if name == 'avgpool':
                break

            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if mode == 'cnn_rnn':
            raw = x.view(batch_sz, seq_len, -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)

        if self.dropout > 0:
            x = self.drop(x)


        if mode == 'cnn_rnn':
            x = x / x.norm(2, 1).expand_as(x)
            x = x.view(batch_sz, seq_len, -1)
            return x, raw
        elif mode == 'cnn':
            x = x / x.norm(2, 1).expand_as(x)
            x = x.view(batch_sz, seq_len, -1)
            x = torch.squeeze(torch.mean(x, 1), 1)
            return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

def resnet18(**kwargs):
    return ResNet(18, **kwargs)

def resnet34(**kwargs):
    return ResNet(34, **kwargs)

def resnet50(**kwargs):
    return ResNet(50, **kwargs)

def resnet101(**kwargs):
    return ResNet(101, **kwargs)

def resnet152(**kwargs):
    return ResNet(152, **kwargs)



