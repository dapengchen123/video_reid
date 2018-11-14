from __future__ import print_function, absolute_import
import time
import torch
from torch.autograd import Variable
from utils.meters import AverageMeter
from utils import to_numpy
from .eva_functions import cmc, mean_ap
import numpy as np
import torch.nn.functional as F



def evaluate_seq(distmat, query_pids, query_camids, gallery_pids, gallery_camids, cmc_topk=(1, 5, 10)):
    query_ids = np.array(query_pids)
    gallery_ids = np.array(gallery_pids)
    query_cams = np.array(query_camids)
    gallery_cams = np.array(gallery_camids)

    ##
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return mAP



class ATTEvaluator(object):

    def __init__(self, cnn_model, att_model, classifier_model,mode):
        super(ATTEvaluator, self).__init__()
        self.cnn_model = cnn_model
        self.att_model = att_model
        self.classifier_model = classifier_model
        self.mode = mode

    def extract_feature(self, data_loader):
        print_freq = 50
        self.cnn_model.eval()
        self.att_model.eval()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        allfeatures = 0
        allfeatures_raw = 0

        for i, (imgs, flows, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs = Variable(imgs, volatile=True)
            flows = Variable(flows, volatile=True)

            if i == 0:
                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)
                out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)
                allfeatures = out_feat
                allfeatures_raw = out_raw
                preimgs = imgs
                preflows = flows
            elif imgs.size(0) < data_loader.batch_size:
                flaw_batchsize = imgs.size(0)
                cat_batchsize = data_loader.batch_size - flaw_batchsize
                imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                flows = torch.cat((flows, preflows[0:cat_batchsize]), 0)

                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)
                out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)

                out_feat = out_feat[0:flaw_batchsize]
                out_raw = out_feat[0:flaw_batchsize]

                allfeatures = torch.cat((allfeatures, out_feat), 0)
                allfeatures_raw = torch.cat((allfeatures_raw, out_raw), 0)
            else:
                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)
                out_feat, out_raw = self.att_model.selfpooling_model(out_feat, out_raw)

                allfeatures = torch.cat((allfeatures, out_feat), 0)
                allfeatures_raw = torch.cat((allfeatures_raw, out_raw), 0)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        return allfeatures, allfeatures_raw

    def evaluate(self, query_loader, gallery_loader, queryinfo, galleryinfo):


        self.cnn_model.eval()
        self.att_model.eval()
        self.classifier_model.eval()


        querypid = queryinfo.pid
        querycamid = queryinfo.camid
        querytranum = queryinfo.tranum

        gallerypid = galleryinfo.pid
        gallerycamid = galleryinfo.camid
        gallerytranum = galleryinfo.tranum

        pooled_probe, hidden_probe = self.extract_feature(query_loader)

        querylen = len(querypid)
        gallerylen = len(gallerypid)

        # online gallery extraction
        single_distmat = np.zeros((querylen, gallerylen))
        gallery_resize = 0
        gallery_popindex = 0
        gallery_popsize = gallerytranum[gallery_popindex]

        gallery_resfeatures = 0
        gallery_resraw = 0

        gallery_empty = True
        preimgs = 0
        preflows = 0

        # time
        gallery_time = AverageMeter()
        end = time.time()


        for i, (imgs, flows, _, _ ) in enumerate(gallery_loader):

            imgs = Variable(imgs, volatile=True)
            flows = Variable(flows, volatile=True)
            seqnum = imgs.size(0)

            if i == 0:
                preimgs = imgs
                preflows = flows

            if gallery_empty:
                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)

                gallery_resfeatures = out_feat
                gallery_resraw = out_raw

                gallery_empty = False

            elif imgs.size(0) < gallery_loader.batch_size:
                flaw_batchsize = imgs.size(0)
                cat_batchsize = gallery_loader.batch_size - flaw_batchsize
                imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                flows = torch.cat((flows, preflows[0:cat_batchsize]), 0)
                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)


                out_feat = out_feat[0:flaw_batchsize]
                out_raw  = out_raw[0:flaw_batchsize]

                gallery_resfeatures = torch.cat((gallery_resfeatures, out_feat), 0)
                gallery_resraw = torch.cat((gallery_resraw, out_raw), 0)

            else:
                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)

                gallery_resfeatures = torch.cat((gallery_resfeatures, out_feat), 0)
                gallery_resraw = torch.cat((gallery_resraw, out_raw), 0)


            gallery_resize = gallery_resize + seqnum

            while gallery_popsize <= gallery_resize:

                if (gallery_popindex + 1) % 50 == 0:
                    print('gallery--{:04d}'.format(gallery_popindex))
                gallery_popfeatures = gallery_resfeatures[0:gallery_popsize, :]
                gallery_popraw = gallery_resraw[0:gallery_popsize, :]

                if gallery_popsize < gallery_resize:
                    gallery_resfeatures = gallery_resfeatures[gallery_popsize:gallery_resize, :]
                    gallery_resraw = gallery_resraw[gallery_popsize:gallery_resize, :]
                else:
                    gallery_resfeatures = 0
                    gallery_resraw = 0
                    gallery_empty = True

                gallery_resize = gallery_resize - gallery_popsize

                pooled_gallery, pooled_raw = self.att_model.selfpooling_model(gallery_popfeatures, gallery_popraw)
                probesize = pooled_probe.size()
                gallerysize = pooled_gallery.size()
                probe_batch = probesize[0]
                gallery_batch = gallerysize[0]
                gallery_num = gallerysize[1]
                pooled_gallery.unsqueeze(0)
                pooled_gallery = pooled_gallery.expand(probe_batch, gallery_batch, gallery_num)

                encode_scores = self.classifier_model(pooled_probe, pooled_gallery)

                encode_size = encode_scores.size()
                encodemat = encode_scores.view(-1, 2)
                encodemat = F.softmax(encodemat)
                encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
                distmat_qall_g = encodemat[:, :, 0]

                q_start = 0
                for qind, qnum in enumerate(querytranum):
                    distmat_qg = distmat_qall_g[q_start:q_start + qnum, :]
                    distmat_qg = distmat_qg.data.cpu().numpy()
                    percile = np.percentile(distmat_qg, 20)

                    if distmat_qg[distmat_qg <= percile] is not None:
                        distmean = np.mean(distmat_qg[distmat_qg <= percile])
                    else:
                        distmean = np.mean(distmat_qg)

                    single_distmat[qind, gallery_popindex] = distmean
                    q_start = q_start + qnum

                gallery_popindex = gallery_popindex + 1

                if gallery_popindex < gallerylen:

                    gallery_popsize = gallerytranum[gallery_popindex]
                gallery_time.update(time.time() - end)
                end = time.time()

        return evaluate_seq(single_distmat, querypid, querycamid, gallerypid, gallerycamid)

