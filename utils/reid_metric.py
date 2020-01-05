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
from .diffusion import Diffusion
from sklearn import preprocessing

def compute_distmat(qf, gf, MemorySave=True, Minibatch=1000):
    m, n = qf.shape[0], gf.shape[0]
    if MemorySave or n>50000:
        Minibatch = min(Minibatch, m)
        print('MemorySave Mode')
        original_dist = np.zeros(shape=[m,n],dtype=np.float16)
        # original_dist = torch.zeros((all_num,all_num),dtype=torch.float16).cuda()
        i = 0
        yy = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, Minibatch).t()
        while True:
            it = i + Minibatch
            if it < np.shape(qf)[0]:
                xx = torch.pow(qf[i:it,:], 2).sum(dim=1, keepdim=True).expand(Minibatch,n)
                distmat = xx + yy
                distmat.addmm_(1,-2,qf[i:it,:],gf.t())
                original_dist[i:it,:] = distmat.cpu().numpy()
                # original_dist[i:it] = distmat
            else:
                xx = torch.pow(qf[-Minibatch:,:], 2).sum(dim=1, keepdim=True).expand(Minibatch,n)
                distmat = xx + yy
                distmat.addmm_(1,-2,qf[-Minibatch:,:],gf.t())
                original_dist[-Minibatch:,:] = distmat.cpu().numpy()
                # original_dist[-Minibatch:,:] = distmat
                break
            i = it
        del xx
        del yy
    else:
        distmat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(m, n).t()
        distmat.addmm_(1,-2,qf,gf.t())
        # original_dist = distmat
        original_dist = distmat.cpu().numpy()
        # original_dist = distmat.numpy()
    return original_dist

def query_expansion(qf, gf, QEk=10, alpha=10.0/2, MemorySave=True, Minibatch=1000):
    
    # gc.collect()
    QE_weight = torch.pow((torch.arange(QEk,0,-1, dtype=torch.float32) / QEk).reshape(QEk, 1), alpha).cuda()
    distmat = compute_distmat(qf, gf)
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #             torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.cpu().numpy()
    ranks = np.argsort(distmat, axis=1)
    ranks_split = ranks[:, :QEk]
    # ranks_split = np.zeros(shape=[original_dist.shape[0], QEk+1], dtype=np.int32)
    # for i in range(all_num):
    #     ranks_split[i,:]=np.argpartition(original_dist[i,:], range(1,k1+1))[:k1+1].astype(np.int32)

    top_k_vecs = gf[ranks_split, :]
    qvecs_temp = torch.matmul(top_k_vecs.permute(0, 2, 1), QE_weight).squeeze()
    qvecs = qvecs_temp / (torch.norm(qvecs_temp, 2, 1, keepdim=True)+1e-6)

    # return qvecs
    # distmat = torch.pow(qvecs, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #         torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qvecs, gf.t())
    # distmat = distmat.cpu().numpy()
    distmat = compute_distmat(qvecs, gf)
    return distmat

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', test_model=''):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.test_model = test_model

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
        # import pdb
        # pdb.set_trace()

        # distmat = query_expansion(qf[:,:1024],gf[:,:1024], 10, 10.0/2)
        # distmat2 = query_expansion(qf[:,1024:],gf[:,1024:], 10, 10.0/2)
        # distmat += distmat2
        # distmat = distmat / 2
        # qf_qe = query_expansion(qf, gf, 10, 10.0/2)
        # np.save('/root/data/reid/diffusion/data/qf_qe_32.npy', qf_qe.cpu().numpy())
        # np.save('/root/data/reid/diffusion/data/gf_32.npy', gf.cpu().numpy())
        # np.save('/root/data/reid/diffusion/data/qf_32.npy', qf.cpu().numpy())
        # scores = features[:n_query] @ features[n_query:].T
        
        # cmc, mAP = eval_func(distmat, q_pids, g_pids)
        # print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))

        # import pdb
        # pdb.set_trace()
        #if type(q_pids[0]) is not str:
        if type(q_pids[0]) is np.int64:
            #print('Only use the first 200 results\n')
            #distmat = distmat[:,:200]  # 11.1
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()

            cmc, mAP = eval_func(distmat, q_pids, g_pids)
            print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))
            # import pdb
            # pdb.set_trace()
            ############################ query_expansion
            print('Using QE:')
            print('1 part:')
            distmat = query_expansion(qf,gf, 10, 10.0/2)
            cmc, mAP = eval_func(distmat, q_pids, g_pids)
            print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))

            # print('2 part:')
            # distmat1 = query_expansion(qf[:,:1024],gf[:,:1024], 10, 10.0/2)
            # distmat2 = query_expansion(qf[:,1024:],gf[:,1024:], 10, 10.0/2)
            # distmat1 += distmat2
            # distmat1 = distmat1 / 2
            # cmc, mAP = eval_func(distmat1, q_pids, g_pids)
            # print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))

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
            # distmat = query_expansion(qf,gf, 10, 10.0/2)
            # cmc, mAP = eval_func(distmat, q_pids, g_pids)
            # print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))

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
            # submission = {}
            # dis_sort = np.argsort(distmat, axis=1)
            # for i in range(m):
            #     result200 = []
            #     for rank in range(200):
            #         result200.append(g_pids[dis_sort[i][rank]])
            #     submission[q_pids[i]] = result200
            # with open("NAIC_baseline_submission_2b.json",'w',encoding='utf-8') as json_file:
            #     json.dump(submission,json_file)

            # model_num = 38 
            # np.save('./ensemble_2b/g_pids.npy',g_pids)
            # np.save('./ensemble_2b/q_pids.npy',q_pids)
            np.save('/tmp/data/model/g_pids.npy',g_pids)
            np.save('/tmp/data/model/q_pids.npy',q_pids)
            np.save('/tmp/data/model/qf.npy',qf.cpu().numpy())
            
            print('Using QE:')
            distmat = query_expansion(qf,gf, 10, 10.0/2)
            # model_name = './ensemble_2b/distmat_'+str(model_num)+'_80_qe' # change with different models (e.g. resnext101_ibn_a, densenet169_ibn_a)
            model_name = '/tmp/data/model/'+self.test_model+'_qe.npy'
            np.save(model_name,distmat)

            submission = {}
            dis_sort = np.argsort(distmat, axis=1)
            for i in range(m):
                result200 = []
                for rank in range(200):
                    result200.append(g_pids[dis_sort[i][rank]])
                submission[q_pids[i]] = result200
            with open("/tmp/data/model/NAIC_baseline_submission_2b.json",'w',encoding='utf-8') as json_file:
                json.dump(submission,json_file)
            
            torch.cuda.empty_cache()
            print('Using Diffusion')
            truncation_size = 750
            kd = 20
            qf = qf.cpu().numpy()
            gf = gf.cpu().numpy()
            diffusion = Diffusion(np.vstack([qf, gf]), './temp')
            offline = diffusion.get_offline_results(truncation_size, kd)
            features = preprocessing.normalize(offline, norm="l2", axis=1)
            scores = features[:m] @ features[m:].T
            distmat = 2 - 2*scores.toarray()
            # model_name = './ensemble_2b/distmat_'+str(model_num)+'_80_diffusion' # change with different models (e.g. resnext101_ibn_a, densenet169_ibn_a)
            model_name = '/tmp/data/model/'+self.test_model+'_diffusion.npy'
            np.save(model_name,distmat)

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

        # np.save('/root/data/reid/diffusion/data/gf.npy', gf.cpu().numpy())
        # np.save('/root/data/reid/diffusion/data/qf.npy', qf.cpu().numpy())
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        # ranks = np.argsort(distmat, axis=1)
        # qf = query_expansion(qf, gf, ranks, m, n)
        # import pdb
        # pdb.set_trace()
        # np.save('./cluster/g_pids_2.npy',g_pids)
        # np.save('./cluster/q_pids_2.npy',q_pids)
        # np.save('./cluster/feat_query_2.npy',qf.cpu().numpy())
        # np.save('./cluster/feat_gallery_2.npy',gf.cpu().numpy())

        import time
        MemorySave = False
        if gf.shape[0]>100000:
            MemorySave = True
        
        if not MemorySave:
            print("Enter reranking")
            #distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            start = time.time()
            distmat = re_ranking(qf, gf, k1=20, k2=3, lambda_value=0.6)
            # distmat = re_ranking(qf[:,:1024], gf[:,:1024], k1=20, k2=3, lambda_value=0.6)
            # distmat2 = re_ranking(qf[:,1024:], gf[:,1024:], k1=20, k2=3, lambda_value=0.6)
            # distmat = distmat + distmat2
            # distmat = distmat / 2
            
            end = time.time()
            print('Running time: %s Seconds'%(end-start))

        #if type(q_pids[0]) is not str:
        # max_result=-1
        # max_value=(0,0,0)
        # import pdb
        # pdb.set_trace()

        # model_name = './ensemble_2a/distmat_resnet50_ibn_a_35_90_1part' # change with different models (e.g. resnext101_ibn_a, densenet169_ibn_a)
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
            if not MemorySave:
                dis_sort = np.argsort(distmat, axis=1)
                for i in range(qf.shape[0]):
                    result200 = []
                    for rank in range(200):
                        result200.append(g_pids[dis_sort[i][rank]])
                    submission[q_pids[i]] = result200
            else:
                batch = 1000
                m = qf.shape[0]
                print("Enter reranking")
                start_time = time.time()
                for start in range(0,m,batch):
                    print('re-ranking:', start)
                    batch_end = min(start+batch,m)
                    distmat = re_ranking(qf[start:batch_end,:1024], gf[:,:1024], k1=20, k2=3, lambda_value=0.6)
                    distmat2 = re_ranking(qf[start:batch_end,1024:], gf[:,1024:], k1=20, k2=3, lambda_value=0.6)
                    distmat = distmat + distmat2
                    distmat = distmat / 2

                    model_name = './cluster/distmat/distmat_se_resnet101_ibn_a_25_80_2part_'+str(start) # change with different models (e.g. resnext101_ibn_a, densenet169_ibn_a)
                    np.save(model_name,distmat)

                    dis_sort = np.argsort(distmat, axis=1)
                    for i in range(start,batch_end):
                        result200 = []
                        for rank in range(200):
                            result200.append(g_pids[dis_sort[i][rank]])
                        submission[q_pids[i]] = result200
                end_time = time.time()
                print('Running time: %s Seconds'%(end_time-start_time))
            
            with open("NAIC_baseline_submission_2b.json",'w',encoding='utf-8') as json_file:
                    json.dump(submission,json_file)
            return 0, 0