from __future__ import absolute_import
from torch import nn
from reid.models import SelfPoolingDir
from reid.models import CrossPoolingDir


class AttModuleDir(nn.Module):

    def __init__(self, input_num, output_num):
        super(AttModuleDir, self).__init__()

        self.input_num = input_num
        self.output_num = output_num

        # attention modules

        self.selfpooling_model = SelfPoolingDir(self.input_num, self.output_num)
        self.crosspooling_model = CrossPoolingDir(self.input_num, self.output_num)

    def forward(self, x, input):
        xsize = x.size()
        sample_num = xsize[0]

        if sample_num % 2 != 0:
            raise RuntimeError("the batch size should be even number!")

        seq_len = x.size()[1]
        x = x.view(int(sample_num/2), 2, seq_len, -1)
        input = input.view(int(sample_num/2), 2, seq_len, -1)
        probe_x = x[:, 0, :, :]
        probe_x = probe_x.contiguous()
        gallery_x = x[:, 1, :, :]
        gallery_x = gallery_x.contiguous()

        probe_input = input[:, 0, :, :]
        probe_input = probe_input.contiguous()
        gallery_input = input[:, 1, :, :]
        gallery_input = gallery_input.contiguous()

        # self-pooling
        pooled_probe, hidden_probe = self.selfpooling_model(probe_x, probe_input)
        pooled_gallery, hidden_gallery = self.selfpooling_model(gallery_x, gallery_input)

        probesize = pooled_probe.size()
        gallerysize = pooled_gallery.size()
        probe_batch = probesize[0]
        gallery_batch = gallerysize[0]
        gallery_num = gallerysize[1]
        pooled_gallery.unsqueeze(0)
        pooled_gallery = pooled_gallery.expand(probe_batch, gallery_batch, gallery_num)

        # cross-pooling

        return pooled_probe, pooled_gallery
