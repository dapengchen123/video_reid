from __future__ import absolute_import

import torch
from torch import nn

from reid.evaluator import accuracy


class PairLoss(nn.Module):
    def __init__(self, sampling_rate=3):
        super(PairLoss, self).__init__()
        self.sampling_rate = sampling_rate
        # self.sigmod = nn.Sigmoid()
        self.BCE = nn.BCELoss()
        self.BCE.size_average = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, score, tar_probe, tar_gallery):
        cls_Size = score.size()
        N_probe = cls_Size[0]
        N_gallery = cls_Size[1]

        tar_gallery = tar_gallery.unsqueeze(1)
        tar_probe = tar_probe.unsqueeze(0)
        mask = tar_probe.expand(N_probe, N_gallery).eq(tar_gallery.expand(N_probe, N_gallery))
        mask = mask.view(-1).cpu().numpy().tolist()

        score = score.contiguous()
        samplers = score.view(-1)

        # samplers = self.sigmod(samplers)
        # labels = Variable(torch.Tensor(mask).cuda())
        labels = torch.Tensor(mask).to(self.device)

        positivelabel = torch.Tensor(mask)
        negativelabel = 1 - positivelabel
        positiveweightsum = torch.sum(positivelabel)
        negativeweightsum = torch.sum(negativelabel)
        neg_relativeweight = positiveweightsum / negativeweightsum * self.sampling_rate
        weights = (positivelabel + negativelabel * neg_relativeweight)
        weights = weights / torch.sum(weights) / 10

        self.BCE.weight = weights.to(self.device)
        loss = self.BCE(samplers, labels)

        samplers_data = samplers.data
        samplers_neg = 1 - samplers_data
        samplerdata = torch.cat((samplers_neg.unsqueeze(1), samplers_data.unsqueeze(1)), 1)

        labeldata = torch.LongTensor(mask).to(self.device)
        prec, = accuracy(samplerdata, labeldata)

        return loss, prec
