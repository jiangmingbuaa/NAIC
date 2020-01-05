#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri, 25 May 2018 20:29:09

@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (torch tensor)
probFea: all feature vectors of the gallery set (torch tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""

'''
import numpy as np
import torch
import gc

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(probFea, galFea, k1=20, k2=6, lambda_value=0.3, MemorySave = True, Minibatch = 5000):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    # q_q_dist = torch.mm(probFea, probFea.t()).cpu().numpy()
    # q_g_dist = torch.mm(probFea, galFea.t()).cpu().numpy()
    # g_g_dist = torch.mm(galFea, galFea.t()).cpu().numpy()

    # original_dist = np.concatenate(
    #   [np.concatenate([q_q_dist, q_g_dist], axis=1),
    #    np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
    #   axis=0)
    # original_dist = 2. - 2 * original_dist   # change the cosine similarity metric to euclidean similarity metric
    # original_dist = np.power(original_dist, 2).astype(np.float32)
    # original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    feat = torch.cat([probFea,galFea])
    print('using GPU to compute original distance')
    if MemorySave or all_num>50000:
        print('MemorySave Mode')
        original_dist = np.zeros(shape=[all_num,all_num],dtype=np.float32)
        i = 0
        yy = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, Minibatch).t()
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                xx = torch.pow(feat[i:it,:], 2).sum(dim=1, keepdim=True).expand(Minibatch,all_num)
                distmat = xx + yy
                distmat.addmm_(1,-2,feat[i:it,:],feat.t())
                original_dist[i:it,:] = distmat.cpu().numpy()
            else:
                xx = torch.pow(feat[-Minibatch:,:], 2).sum(dim=1, keepdim=True).expand(Minibatch,all_num)
                distmat = xx + yy
                distmat.addmm_(1,-2,feat[-Minibatch:,:],feat.t())
                original_dist[-Minibatch:,:] = distmat.cpu().numpy()
                break
            i = it
        del xx
        del yy
        del distmat
    else:
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
            torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        # original_dist = distmat.numpy()
    del feat
    gc.collect()

    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    # query_num = q_g_dist.shape[0]
    # all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
        gc.collect()

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    gc.collect()
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

'''
import numpy as np
import torch
import gc

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False, MemorySave = True, Minibatch = 1000):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        # feat = torch.cat([probFea.cpu(),galFea.cpu()])
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        if MemorySave or all_num>50000:
            Minibatch = min(Minibatch, query_num)
            print('MemorySave Mode')
            original_dist = np.zeros(shape=[all_num,all_num],dtype=np.float16)
            # original_dist = torch.zeros((all_num,all_num),dtype=torch.float16).cuda()
            i = 0
            yy = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, Minibatch).t()
            while True:
                it = i + Minibatch
                if it < np.shape(feat)[0]:
                    xx = torch.pow(feat[i:it,:], 2).sum(dim=1, keepdim=True).expand(Minibatch,all_num)
                    distmat = xx + yy
                    distmat.addmm_(1,-2,feat[i:it,:],feat.t())
                    original_dist[i:it,:] = distmat.cpu().numpy()
                    # original_dist[i:it] = distmat
                else:
                    xx = torch.pow(feat[-Minibatch:,:], 2).sum(dim=1, keepdim=True).expand(Minibatch,all_num)
                    distmat = xx + yy
                    distmat.addmm_(1,-2,feat[-Minibatch:,:],feat.t())
                    original_dist[-Minibatch:,:] = distmat.cpu().numpy()
                    # original_dist[-Minibatch:,:] = distmat
                    break
                i = it
            del xx
            del yy
        else:
            distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
            distmat.addmm_(1,-2,feat,feat.t())
            # original_dist = distmat
            original_dist = distmat.cpu().numpy()
            # original_dist = distmat.numpy()
        del distmat
        del feat
        gc.collect()
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    # original_dist = torch.transpose(original_dist / torch.max(original_dist, dim=0), 0, 1)
    # V = torch.zeros_like(original_dist)
    # import pdb
    # pdb.set_trace()
    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # initial_rank = np.zeros_like(original_dist).astype(np.int32)
    # for i in range(all_num):
    #     initial_rank[i,:]=np.argsort(original_dist[i,:]).astype(np.int32)
    # initial_rank = np.argpartition(original_dist, range(1,k1+1))
    initial_rank = np.zeros(shape=[original_dist.shape[0], k1+1], dtype=np.int32)
    for i in range(all_num):
        initial_rank[i,:]=np.argpartition(original_dist[i,:], range(1,k1+1))[:k1+1].astype(np.int32)
    # _, initial_rank = torch.topk(original_dist, k1+1, dim=1)
    import pdb
    pdb.set_trace()
    V = np.zeros_like(original_dist).astype(np.float16)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
