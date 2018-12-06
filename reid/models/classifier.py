from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init
import numpy as np


class Classifier(nn.Module):

    def __init__(self, feat_num, class_num, drop=0):
        super(Classifier, self).__init__()
        self.feat_num = feat_num
        self.class_num = class_num
        self.drop = drop

        # BN layer
        self.classifierBN = nn.BatchNorm1d(self.feat_num)
        # feat classifeir
        self.classifierlinear = nn.Linear(self.feat_num, self.class_num)
        # dropout_layer
        self.drop = drop
        if self.drop > 0:
            self.droplayer = nn.Dropout(drop)

        init.constant_(self.classifierBN.weight, 1)
        init.constant_(self.classifierBN.bias, 0)

        init.normal_(self.classifierlinear.weight, std=0.001)
        init.constant_(self.classifierlinear.bias, 0)

    def forward(self, probe, gallery):
        S_gallery = gallery.size()
        N_probe = S_gallery[0]
        N_gallery = S_gallery[1]
        feat_num = S_gallery[2]

        probe = probe.unsqueeze(1)
        probe = probe.expand(N_probe, N_gallery, feat_num)
        diff = torch.pow(probe - gallery, 2)
        diff = diff.view(N_probe * N_gallery, -1)
        diff = diff.contiguous()

        slice = 50000
        if N_probe * N_gallery < slice:
            diff = self.classifierBN(diff)
            if self.drop > 0:
                diff = self.droplayer(diff)

            cls_encode = self.classifierlinear(diff)
            cls_encode = cls_encode.view(N_probe, N_gallery, -1)

        else:

            Iter_time = int(np.floor(N_probe * N_gallery / slice))
            cls_encode = 0
            for i in range(0, Iter_time):
                before_index = i * slice
                after_index = (i + 1) * slice

                diff_i = diff[before_index:after_index, :]
                diff_i = self.classifierBN(diff_i)

                if self.drop > 0:
                    diff_i = self.droplayer(diff_i)

                cls_encode_i = self.classifierlinear(diff_i)

                if i == 0:
                    cls_encode = cls_encode_i
                else:
                    cls_encode = torch.cat((cls_encode, cls_encode_i), 0)

            before_index = Iter_time * slice
            after_index = N_probe * N_gallery
            if after_index > before_index:
                diff_i = diff[before_index:after_index, :]
                diff_i = self.classifierBN(diff_i)
                if self.drop > 0:
                    diff_i = self.droplayer(diff_i)

                cls_encode_i = self.classifierlinear(diff_i)
                cls_encode = torch.cat((cls_encode, cls_encode_i), 0)

            cls_encode = cls_encode.view(N_probe, N_gallery, self.class_num)

        return cls_encode
