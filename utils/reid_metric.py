# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric
import json

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []

    def update(self, output):
        feat, pid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()

        #import pdb
        #pdb.set_trace()
        #if type(q_pids[0]) is not str:
        if type(q_pids[0]) is np.int64:
            #print('Only use the first 200 results\n')
            #distmat = distmat[:,:200]  # 11.1
            cmc, mAP = eval_func(distmat, q_pids, g_pids)
            # import pdb
            # pdb.set_trace()
            print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))
            return cmc, mAP
        else:
            submission = {}
            dis_sort = np.argsort(distmat, axis=1)
            for i in range(m):
                result200 = []
                for rank in range(200):
                    result200.append(g_pids[dis_sort[i][rank]])
                submission[q_pids[i]] = result200
            with open("NAIC_baseline_submission.json",'w',encoding='utf-8') as json_file:
                json.dump(submission,json_file)


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []

    def update(self, output):
        feat, pid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        #distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        distmat = re_ranking(qf, gf, k1=20, k2=3, lambda_value=0.6)


        #if type(q_pids[0]) is not str:
        # max_result=-1
        # max_value=(0,0,0)
        # import pdb
        # pdb.set_trace()
        model_name = './ensemble_b/distmat_resnext101_ibn_a_60'
        np.save(model_name,distmat)
        np.save('./ensemble_b/g_pids.npy',g_pids)
        np.save('./ensemble_b/q_pids.npy',q_pids)

        if type(q_pids[0]) is np.int64:
            #print('Only use the first 200 results\n')
            #distmat = distmat[:,:200]  # 11.1
            # for k1 in range(2,21):
            #     for k2 in range(1,k1):
            #         for value in range(11):
            #             distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=value/10.0)
            #             cmc, mAP = eval_func(distmat, q_pids, g_pids)
            #             print(k1,k2,value/10.0)
            #             print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))
            #             if 0.5*cmc[0]+0.5*mAP>max_result:
            #                 max_result = 0.5*cmc[0]+0.5*mAP
            #                 max_value = (k1,k2,value/10.0)
            # print(max_result)
            # print(max_value)
            cmc, mAP = eval_func(distmat, q_pids, g_pids)
            print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))
            return cmc, mAP
        else:
            submission = {}
            dis_sort = np.argsort(distmat, axis=1)
            for i in range(qf.shape[0]):
                result200 = []
                for rank in range(200):
                    result200.append(g_pids[dis_sort[i][rank]])
                submission[q_pids[i]] = result200
            with open("NAIC_baseline_submission_b.json",'w',encoding='utf-8') as json_file:
                json.dump(submission,json_file)