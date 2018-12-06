from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init


class CrossPoolingDir(nn.Module):

    def __init__(self, input_num, output_num):
        super(CrossPoolingDir, self).__init__()
        self.input_num = input_num
        self.output_num = output_num

        self.featK = nn.Linear(self.input_num, self.output_num)
        self.featK_bn = nn.BatchNorm1d(self.output_num)

        # Softmax
        self.softmax = nn.Softmax()

        init.kaiming_uniform_(self.featK.weight, mode='fan_out')
        init.constant_(self.featK.bias, 0)
        init.constant_(self.featK_bn.weight, 1)
        init.constant_(self.featK_bn.bias, 0)

    def forward(self,  gallery_value, gallery_base, querys):

        gal_size = gallery_value.size()
        gal_batch = gal_size[0]
        gal_len = gal_size[1]

        # Linear self-transorfmation
        Q_size = querys.size()
        pro_batch = Q_size[0]
        Q_featnum = Q_size[1]

        K = gallery_base.view(gal_batch * gal_len, -1)
        K = self.featK(K)
        K = self.featK_bn(K)
        K = K.view(gal_batch, gal_len, -1)
        #  K: gal_batch x gal_len x H_featnum
        #  query: pro_batch x H_featnum

        Q = querys.unsqueeze(1)
        Q = Q.unsqueeze(1)
        K = K.unsqueeze(0)

        #  Q: pro_batch x 1 x 1 x Q_featnum
        #  K: 1 x gal_batch x gal_len x Q_featnum

        Q = Q.expand(pro_batch, gal_batch, gal_len, Q_featnum)
        K = K.expand(pro_batch, gal_batch, gal_len, Q_featnum)

        QK = Q * K
        QK = QK.permute(0, 1, 3, 2)

        # pro_batch x gal_batch x Q_featnum x gal_len
        QK = QK.contiguous()
        QK = QK.view(-1, gal_len)
        weights = self.softmax(QK)
        weights = weights.view(pro_batch,  gal_batch, Q_featnum, gal_len)

        # gallery : gal_batch x gal_len x Q_featnum
        gallery_value = gallery_value.permute(0, 2, 1)
        # gallery : gal_batch x Q_featnum x gal_len
        gallery_value = gallery_value.contiguous()
        gallery_value = gallery_value.unsqueeze(0)
        # gallery : 1 x gal_batch x Q_featnum x gal_len
        gallery_value = gallery_value.expand(pro_batch, gal_batch, Q_featnum, gal_len)
        # gallery : pro_batch x gal_batch x Q_featnum x gal_len
        pool_gallery = (weights * gallery_value).sum(3)
        # pool_gallery = pool_gallery.squeeze(3)

        return pool_gallery
