from __future__ import print_function, absolute_import
import time
import torch
from torch.autograd import Variable
from reid.evaluator import accuracy
from utils.meters import AverageMeter
import torch.nn.functional as F


# mode decide how to train the model


class BaseTrainer(object):

    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer1, optimizer2):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        precisions1 = AverageMeter()
        precisions2 = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)


            loss, prec_oim, prec_score, prec_finalscore = self._forward(inputs, targets)
            losses.update(loss.data[0], targets.size(0))

            precisions.update(prec_oim, targets.size(0))
            precisions1.update(prec_score, targets.size(0))
            precisions2.update(prec_finalscore, targets.size(0))

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            batch_time.update(time.time() - end)
            end = time.time()
            print_freq = 50
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'prec_oim {:.2%} ({:.2%})\t'
                      'prec_score {:.2%} ({:.2%})\t'
                      'prec_finalscore(total) {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              precisions1.val, precisions1.avg,
                              precisions2.val, precisions2.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class SEQTrainer(BaseTrainer):

    def __init__(self, cnn_model, att_model, classifier_model, criterion_veri, criterion_oim, mode, rate):
        super(SEQTrainer, self).__init__(cnn_model, criterion_veri)
        self.att_model = att_model
        self.classifier_model = classifier_model
        self.regular_criterion = criterion_oim
        self.mode = mode
        self.rate = rate

    def _parse_data(self, inputs):
        imgs, flows, pids, _ = inputs
        inputs = [Variable(imgs), Variable(flows)]
        targets = Variable(pids).cuda()
        return inputs, targets

    def _forward(self, inputs, targets):

        if self.mode == 'cnn':
            out_feat = self.model(inputs[0], inputs[1], self.mode)

            loss, outputs = self.regular_criterion(out_feat, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]

            return loss, prec, 0, 0

        elif self.mode == 'cnn_rnn':


            feat, feat_raw  = self.model(inputs[0], inputs[1], self.mode)
            featsize = feat.size()
            featbatch = featsize[0]
            seqlen = featsize[1]

            ## expand the target label ID loss
            featX = feat.view(featbatch * seqlen, -1)

            targetX = targets.unsqueeze(1)
            targetX = targetX.expand(featbatch, seqlen)
            targetX = targetX.contiguous()
            targetX = targetX.view(featbatch * seqlen, -1)
            targetX = targetX.squeeze(1)
            loss_id, outputs_id = self.regular_criterion(featX, targetX)

            prec_id, = accuracy(outputs_id.data, targetX.data)
            prec_id = prec_id[0]

            ## verification label

            featsize = feat.size()
            sample_num = featsize[0]
            targets = targets.data
            targets = targets.view(int(sample_num / 2), -1)
            tar_probe = targets[:, 0]
            tar_gallery = targets[:, 1]

            pooled_probe, pooled_gallery = self.att_model(feat, feat_raw)

            encode_scores = self.classifier_model(pooled_probe, pooled_gallery)

            encode_size = encode_scores.size()
            encodemat = encode_scores.view(-1, 2)
            encodemat = F.softmax(encodemat)
            encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
            encodemat = encodemat[:, :, 1]

            loss_ver, prec_ver = self.criterion(encodemat, tar_probe, tar_gallery)


            loss = loss_id*self.rate + 100*loss_ver

            return loss, prec_id, prec_ver, 0
        else:
            raise ValueError("Unsupported loss:", self.criterion)

    def train(self, epoch, data_loader, optimizer1, optimizer2, rate):
        self.att_model.train()
        self.classifier_model.train()
        self.rate = rate
        super(SEQTrainer, self).train(epoch, data_loader, optimizer1, optimizer2)
