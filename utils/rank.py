#!/usr/bin/env python
# -*- coding: utf-8 -*-

" rank module "

import os
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from knn import KNN
from diffusion import Diffusion
from sklearn import preprocessing


def search():
    # import pdb
    # pdb.set_trace()
    n_query = len(queries)
    result=[]
    for truncation_size in range(750,950,50):
        for kd in range(15,26,5):
            diffusion = Diffusion(np.vstack([queries, gallery]), args.cache_dir)
            offline = diffusion.get_offline_results(truncation_size, kd)
            features = preprocessing.normalize(offline, norm="l2", axis=1)
            scores = features[:n_query] @ features[n_query:].T
            cmc, mAP = evaluate(-scores.toarray(), q_pids, g_pids)
            print(truncation_size,kd)
            print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))
            result.append((truncation_size,kd,cmc[0],mAP,0.5*cmc[0]+0.5*mAP))
    print(result)
    import pdb
    pdb.set_trace()
    # diffusion = Diffusion(np.vstack([queries, gallery]), args.cache_dir)
    # offline = diffusion.get_offline_results(args.truncation_size, args.kd)
    # features = preprocessing.normalize(offline, norm="l2", axis=1)
    # scores = features[:n_query] @ features[n_query:].T
    # cmc, mAP = evaluate(-scores.toarray(), q_pids, g_pids)
    # print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))


def search_old(gamma=3):
    diffusion = Diffusion(gallery, args.cache_dir)
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)

    time0 = time.time()
    print('[search] 1) k-NN search')
    sims, ids = diffusion.knn.search(queries, args.kq)
    sims = sims ** gamma
    qr_num = ids.shape[0]

    print('[search] 2) linear combination')
    all_scores = np.empty((qr_num, args.truncation_size), dtype=np.float32)
    all_ranks = np.empty((qr_num, args.truncation_size), dtype=np.int)
    for i in tqdm(range(qr_num), desc='[search] query'):
        scores = sims[i] @ offline[ids[i]]
        parts = np.argpartition(-scores, args.truncation_size)[:args.truncation_size]
        ranks = np.argsort(-scores[parts])
        all_scores[i] = scores[parts][ranks]
        all_ranks[i] = parts[ranks]
    print('[search] search costs {:.2f}s'.format(time.time() - time0))

    # 3) evaluation
    evaluate(all_ranks)


def evaluate(distmat, q_pids, g_pids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    # import pdb
    # pdb.set_trace()
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        # remove = (g_pids[order] == q_pid)
        remove = [False]*len(order)  # 11.1
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    # assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',
                        type=str,
                        default='./cache',
                        help="""
                        Directory to cache
                        """)
    # parser.add_argument('--dataset_name',
    #                     type=str,
    #                     required=True,
    #                     help="""
    #                     Name of the dataset
    #                     """)
    # parser.add_argument('--query_path',
    #                     type=str,
    #                     required=True,
    #                     help="""
    #                     Path to query features
    #                     """)
    # parser.add_argument('--gallery_path',
    #                     type=str,
    #                     required=True,
    #                     help="""
    #                     Path to gallery features
    #                     """)
    # parser.add_argument('--gnd_path',
    #                     type=str,
    #                     help="""
    #                     Path to ground-truth
    #                     """)
    parser.add_argument('-n', '--truncation_size',
                        type=int,
                        default=1000,
                        help="""
                        Number of images in the truncated gallery
                        """)
    args = parser.parse_args()
    args.kq, args.kd = 10, 50
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir)
    # dataset = Dataset(args.query_path, args.gallery_path)
    # queries, gallery = dataset.queries, dataset.gallery
    queries = np.load('./data/qf.npy')
    gallery = np.load('./data/gf.npy')
    q_pids = np.load('./data/q_pids.npy')
    g_pids = np.load('./data/g_pids.npy')
    search()