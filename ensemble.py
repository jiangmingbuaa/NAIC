import numpy as np
import json
from data.datasets.eval_reid import eval_func


g_pids = np.load('./ensemble_b/g_pids.npy')
q_pids = np.load('./ensemble_b/q_pids.npy')

# model_list = ['resnet50_ibn_a_80', 'densenet169_ibn_a_80', 'resnext101_ibn_a_80', 'se_resnet101_ibn_a_100', 'resnet50_ibn_a_maxpool_60']
model_list = ['resnet50_ibn_a_80', 'densenet169_ibn_a_80', 'resnext101_ibn_a_80', 'se_resnet101_ibn_a_80']

dist_list = []
for name in model_list:
    dist_list.append(np.load('./ensemble_b/distmat_'+name+'.npy'))

# weight = [1.0/3, 1.0/3, 1.0/3]
weight = [0.4, 0.25, 0.25, 0.1]
# weight = [0.35, 0.3, 0.3, 0.05]
# weight = [0.25, 0.25, 0.25, 0.25]

distmat = weight[0] * dist_list[0]
for i in range(1,len(model_list)):
    distmat += weight[i] * dist_list[i]

dis_sort = np.argsort(distmat, axis=1)
if type(q_pids[0]) is np.int64:
    cmc, mAP = eval_func(distmat, q_pids, g_pids)
    print(('rank 1: {:.3%} mAP: {:.3%}, result: {:.3%}'.format(cmc[0], mAP, 0.5*cmc[0]+0.5*mAP)))
else:
    dis_sort = np.argsort(distmat, axis=1)
    submission = {}
    for i in range(q_pids.shape[0]):
        result200 = []
        for rank in range(200):
            result200.append(g_pids[dis_sort[i][rank]])
        submission[q_pids[i]] = result200
    with open("NAIC_ensemble_submission_b.json",'w',encoding='utf-8') as json_file:
        json.dump(submission,json_file)