import numpy as np
import json
from data.datasets.eval_reid import eval_func


g_pids = np.load('/tmp/data/model/q_pids.npy')
q_pids = np.load('/tmp/data/model/g_pids.npy')

# model_list = ['resnet50_ibn_a_80', 'densenet169_ibn_a_80', 'resnext101_ibn_a_80', 'se_resnet101_ibn_a_100', 'resnet50_ibn_a_maxpool_60']
# model_list = ['resnet50_ibn_a_80', 'densenet169_ibn_a_80', 'resnext101_ibn_a_80', 'se_resnet101_ibn_a_80']
# model_list_qe = ['25_80', '34_80', '36_80', '37_80', '38_80']
# model_list_di = ['25_80', '34_80', '36_80', '37_80', '38_80']
model_list = ['stage_1_se_resnet101_80', 'stage_1_se_resnet101_90', 'stage_2_resnet50_80', 'stage_2_resnet50_90', 'stage_3_se_resnet101_20', 'stage_3_resnet50_20']


# print('Loading model qe:')
# dist_list_qe = []
# for name in model_list_qe:
#     dist_list_qe.append(np.load('./ensemble_2b/temp/distmat_'+name+'_qe.npy'))

distmat = np.load('/tmp/data/model/stage_1_se_resnet101_80_qe.npy')
for i,name in enumerate(model_list):
    if i==0:
        continue
    distmat += np.load('/tmp/data/model/'+name+'_qe.npy')

for i,name in enumerate(model_list):
    distmat += np.load('/tmp/data/model/'+name+'_diffusion.npy')
# print('Loading model fi:')
# dist_list_di = []
# for name in model_list:
#     dist_list_di.append(np.load('./ensemble_2b/temp/distmat_'+name+'_diffusion.npy'))

# weight = [1.0/3, 1.0/3, 1.0/3]
# weight = [0.4, 0.25, 0.25]
# weight = [0.35, 0.3, 0.3, 0.05]
# weight = [1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6]
# weight = [0.2, 0.2, 0.2, 0.2, 0.2]

# distmat = weight[0] * (dist_list_qe[0])
# for i in range(1,len(model_list_qe)):
#     distmat += weight[i] * (dist_list_qe[i])


print('Result:')
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
    with open("/tmp/data/answer/NAIC_ensemble_submission_2b.json",'w',encoding='utf-8') as json_file:
        json.dump(submission,json_file)