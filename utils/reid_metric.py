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
from .ecn import ECN

def query_expansion(qf, gf, ranks, m, n, QEk=10, alpha=10.0/2):
    QE_weight = torch.pow((torch.arange(QEk,0,-1, dtype=torch.float32) / QEk).reshape(QEk, 1), alpha).cuda()
    ranks_split = ranks[:, :QEk]
    top_k_vecs = gf[ranks_split, :]
    qvecs_temp = torch.matmul(top_k_vecs.permute(0, 2, 1), QE_weight).squeeze()
    qvecs_temp += qf
    qvecs = qvecs_temp / (torch.norm(qvecs_temp, 2, 1, keepdim=True)+1e-6)

    return qvecs
    # distmat = torch.pow(qvecs, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #         torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qvecs, gf.t())
    # distmat = distmat.cpu().numpy()
    # return distmat

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

        # ranks = np.argsort(distmat, axis=1)
        # distmat = query_expansion(qf, gf, ranks, m, n, 10, 5)

        # import pdb
        # pdb.set_trace()
        #if type(q_pids[0]) is not str:
        if type(q_pids[0]) is np.int64:
            #print('Only use the first 200 results\n')
            #distmat = distmat[:,:200]  # 11.1
            cmc, mAP = eval_func(distmat, q_pids, g_pids)
            print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))
            # import pdb
            # pdb.set_trace()
            ############################ query_expansion
            # QEk=10
            # alpha=10.0/2
            # QE_weight = torch.pow((torch.arange(QEk,0,-1, dtype=torch.float32) / QEk).reshape(QEk, 1), alpha).cuda()
            # ranks = np.argsort(distmat, axis=1)
            # ranks_split = ranks[:, :QEk]
            # top_k_vecs = gf[ranks_split, :]
            # qvecs_temp = torch.matmul(top_k_vecs.permute(0, 2, 1), QE_weight).squeeze()
            # qvecs = qvecs_temp / (torch.norm(qvecs_temp, 2, 1, keepdim=True)+1e-6)

            # distmat = torch.pow(qvecs, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            #       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            # distmat.addmm_(1, -2, qvecs, gf.t())
            # distmat = distmat.cpu().numpy()

            # max_result=-1
            # ranks = np.argsort(distmat, axis=1)
            # for k in range(2,25,2):
            #     alpha=k//2
            #     distmat = self.query_expansion(gf, ranks, m, n, k, alpha)
            #     cmc, mAP = eval_func(distmat, q_pids, g_pids)
            #     print(k,alpha)
            #     print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))

            return cmc, mAP
        else:
            submission = {}
            dis_sort = np.argsort(distmat, axis=1)
            for i in range(m):
                result200 = []
                for rank in range(200):
                    result200.append(g_pids[dis_sort[i][rank]])
                submission[q_pids[i]] = result200
            with open("NAIC_baseline_submission_2a.json",'w',encoding='utf-8') as json_file:
                json.dump(submission,json_file)
            return 0, 0


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
        # ranks = np.argsort(distmat, axis=1)
        # qf = query_expansion(qf, gf, ranks, m, n)

        print("Enter reranking")
        #distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        import time
        start = time.time()
        # distmat = re_ranking(qf, gf, k1=20, k2=3, lambda_value=0.6)
        distmat1 = re_ranking(qf[:,:1024], gf[:,:1024], k1=20, k2=3, lambda_value=0.6)
        distmat2 = re_ranking(qf[:,1024:], gf[:,1024:], k1=20, k2=3, lambda_value=0.6)
        distmat = distmat1 + distmat2
        end = time.time()
        print('Running time: %s Seconds'%(end-start))

        #if type(q_pids[0]) is not str:
        # max_result=-1
        # max_value=(0,0,0)
        # import pdb
        # pdb.set_trace()

        # model_name = './ensemble_2a/distmat_resnet50_ibn_a_80' # change with different models (e.g. resnext101_ibn_a, densenet169_ibn_a)
        # np.save(model_name,distmat)
        # np.save('./ensemble_2a/g_pids.npy',g_pids)
        # np.save('./ensemble_2a/q_pids.npy',q_pids)

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

            # start = time.time()
            # distmat = ECN(qf, gf, k=25,t=1,q=3,method='rankdist')
            # end = time.time()
            # print('Running time: %s Seconds'%(end-start))
            # cmc, mAP = eval_func(distmat, q_pids, g_pids)
            # print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))
            # for t in range(10, 15):
            #     for q in range(1, t+2):
            #         distmat = ECN(qf.cpu().numpy(), gf.cpu().numpy(), k=25,t=t,q=q,method='rankdist')
            #         cmc, mAP = eval_func(distmat, q_pids, g_pids)
            #         # if 0.5*cmc[0]+0.5*mAP>0.94:
            #         print(t,q)
            #         print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))

            return cmc, mAP
        else:
            submission = {}
            dis_sort = np.argsort(distmat, axis=1)
            for i in range(qf.shape[0]):
                result200 = []
                for rank in range(200):
                    result200.append(g_pids[dis_sort[i][rank]])
                submission[q_pids[i]] = result200
            with open("NAIC_baseline_submission_2a.json",'w',encoding='utf-8') as json_file:
                json.dump(submission,json_file)
            return 0, 0