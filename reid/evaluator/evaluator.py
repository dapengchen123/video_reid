from __future__ import print_function, absolute_import
import time
import torch
from torch.autograd import Variable
from utils.meters import AverageMeter
from utils import to_numpy
from .eva_functions import cmc, mean_ap
import numpy as np
from utils import to_torch


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


def pairwise_distance_tensor(query_x, gallery_x):

    # query_n = query_x.size(0)
    # gallery_n = gallery_x.size(0)
    # query_squ = torch.pow(query_x, 2).sum(1)  # .squeeze(1)
    # gallery_squ = torch.pow(gallery_x, 2).sum(1)  # .squeeze(1)
    # query_squ = query_squ.unsqueeze(1)
    # gallery_squ = gallery_squ.unsqueeze(0)
    # query_squ = query_squ.expand(query_n, gallery_n)
    # gallery_squ = gallery_squ.expand(query_n, gallery_n)
    #
    # query_gallery_squ = query_squ + gallery_squ
    # query_gallery_pro = torch.mm(query_x, gallery_x.t())
    # dist = query_gallery_squ - 2*query_gallery_pro
    m, n = query_x.size(0), gallery_x.size(0)
    x = query_x.view(m, -1)
    y = gallery_x.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) +\
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())

    return dist


class CNNEvaluator(object):

    def __init__(self, cnn_model, mode):
        super(CNNEvaluator, self).__init__()
        self.cnn_model = cnn_model
        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def extract_feature(self, cnn_model, data_loader):
        print_freq = 50
        cnn_model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        allfeatures = 0

        for i, (imgs, flows, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs = to_torch(imgs)
            flows = to_torch(flows)
            imgs = imgs.to(self.device)
            flows = flows.to(self.device)

            with torch.no_grad():
                if i == 0:
                    out_feat = self.cnn_model(imgs, flows, self.mode)
                    allfeatures = out_feat.data
                    preimgs = imgs
                    preflows = flows
                elif imgs.size(0) < data_loader.batch_size:
                    flaw_batchsize = imgs.size(0)
                    cat_batchsize = data_loader.batch_size - flaw_batchsize
                    imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                    flows = torch.cat((flows, preflows[0:cat_batchsize]), 0)
                    out_feat = self.cnn_model(imgs, flows, self.mode)
                    out_feat = out_feat[0:flaw_batchsize]
                    allfeatures = torch.cat((allfeatures, out_feat.data), 0)
                else:
                    out_feat = cnn_model(imgs, flows, self.mode)
                    allfeatures = torch.cat((allfeatures, out_feat.data), 0)

                batch_time.update(time.time() - end)
                end = time.time()

                if (i+1) % print_freq == 0:
                    print('Extract Features: [{}/{}]\t''Time {:.3f} ({:.3f})\t''Data {:.3f} ({:.3f})\t'.format(i + 1, len(data_loader),
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg))

        return allfeatures

    def evaluate(self, query_loader, gallery_loader, queryinfo, galleryinfo):

        self.cnn_model.eval()

        querypid = queryinfo.pid
        querycamid = queryinfo.camid
        querytranum = queryinfo.tranum

        gallerypid = galleryinfo.pid
        gallerycamid = galleryinfo.camid
        gallerytranum = galleryinfo.tranum

        query_features = self.extract_feature(self.cnn_model, query_loader)

        querylen = len(querypid)
        gallerylen = len(gallerypid)

        # online gallery extraction
        single_distmat = np.zeros((querylen, gallerylen))
        gallery_resize = 0
        gallery_popindex = 0
        gallery_popsize = gallerytranum[gallery_popindex]
        gallery_resfeatures = 0
        gallery_empty = True
        preimgs = 0
        preflows = 0

        # time
        gallery_time = AverageMeter()
        end = time.time()
        
        for i, (imgs, flows, _, _) in enumerate(gallery_loader):
            imgs = to_torch(imgs)
            flows = to_torch(flows)
            imgs = imgs.to(self.device)
            flows = flows.to(self.device)
            with torch.no_grad():
                seqnum = imgs.size(0)
                ##############
                if i == 0:
                    preimgs = imgs
                    preflows = flows

                if gallery_empty:
                    out_feat = self.cnn_model(imgs, flows, self.mode)

                    gallery_resfeatures = out_feat.data
                    gallery_empty = False

                elif imgs.size(0) < gallery_loader.batch_size:
                    flaw_batchsize = imgs.size(0)
                    cat_batchsize = gallery_loader.batch_size - flaw_batchsize
                    imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                    flows = torch.cat((flows, preflows[0:cat_batchsize]), 0)
                    out_feat = self.cnn_model(imgs, flows, self.mode)
                    out_feat = out_feat[0:flaw_batchsize]
                    gallery_resfeatures = torch.cat((gallery_resfeatures, out_feat.data), 0)
                else:
                    out_feat = self.cnn_model(imgs, flows, self.mode)
                    gallery_resfeatures = torch.cat((gallery_resfeatures, out_feat.data), 0)

            gallery_resize = gallery_resize + seqnum

            while gallery_popsize <= gallery_resize:
                if (gallery_popindex + 1) % 50 == 0:
                    print('gallery--{:04d}'.format(gallery_popindex))
                if gallery_popsize == 1:
                    gallery_popfeatures = gallery_resfeatures
                else:
                    gallery_popfeatures = gallery_resfeatures[0:gallery_popsize, :]
                if gallery_popsize < gallery_resize:
                    gallery_resfeatures = gallery_resfeatures[gallery_popsize:gallery_resize, :]
                else:
                    gallery_resfeatures = 0
                    gallery_empty = True
                gallery_resize = gallery_resize - gallery_popsize
                distmat_qall_g = pairwise_distance_tensor(query_features, gallery_popfeatures)

                q_start = 0
                for qind, qnum in enumerate(querytranum):
                    distmat_qg = distmat_qall_g[q_start:q_start + qnum, :]
                    distmat_qg = distmat_qg.cpu().numpy()
                    percile = np.percentile(distmat_qg, 20)

                    if distmat_qg[distmat_qg < percile] is not None:
                        distmean = np.mean(distmat_qg[distmat_qg < percile])
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
