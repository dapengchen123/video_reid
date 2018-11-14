from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init

class SelfPoolingDir(nn.Module):
    def __init__(self, input_num, output_num):
        super(SelfPoolingDir, self).__init__()
        self.input_num = input_num
        self.output_num = output_num

        ## Linear K
        self.featK = nn.Linear(self.input_num, self.output_num)
        self.featK_bn = nn.BatchNorm1d(self.output_num)

        ## Linear_Q
        self.featQ = nn.Linear(self.input_num, self.output_num)
        self.featQ_bn = nn.BatchNorm1d(self.output_num)


        ## Softmax
        self.softmax = nn.Softmax()

        init.kaiming_normal(self.featK.weight, mode='fan_out')
        init.constant(self.featK.bias, 0)
        init.constant(self.featK_bn.weight, 1)
        init.constant(self.featK_bn.bias, 0)

        init.kaiming_normal(self.featQ.weight, mode='fan_out')
        init.constant(self.featQ.bias, 0)
        init.constant(self.featQ_bn.weight, 1)
        init.constant(self.featQ_bn.bias, 0)



    def forward(self, probe_value, probe_base):
#        pro_size = probe_value.size()
#        pro_batch = pro_size[0]
#        pro_len = pro_size[1]
#
#        ## generating Querys
#        Qs = probe_base.view(pro_batch * pro_len, -1)
#        Qs = self.featQ(Qs)
#        Qs = self.featQ_bn(Qs)
#        Qs = Qs.view(pro_batch, pro_len, -1)
#
#        Qmean = torch.mean(Qs, 1)
#        Qs = Qmean.squeeze(1)
#        Hs = Qmean.expand(pro_batch, pro_len, self.output_num)
#
#        ## generating Keys
#        K = probe_base.view(pro_batch * pro_len, -1)
#        K = self.featK(K)
#        K = self.featK_bn(K)
#        K = K.view(pro_batch, pro_len, -1)
#
#
#        weights = Hs * K
#        weights = weights.permute(0, 2, 1)
#        weights = weights.contiguous()
#        weights = weights.view(-1, pro_len)
#        weights = self.softmax(weights)
#        weights = weights.view(pro_batch, self.output_num, pro_len)
#        weights = weights.permute(0, 2, 1)
#
#        pool_probe = probe_value * weights
#        pool_probe = pool_probe.sum(1)
#        pool_probe = pool_probe.squeeze(1)

        pool_probe = torch.mean(probe_value,1)
        pool_probe = pool_probe.squeeze(1)


        # pool_probe  Batch x featnum
        # Hs  Batch x hidden_num

        return pool_probe, pool_probe
