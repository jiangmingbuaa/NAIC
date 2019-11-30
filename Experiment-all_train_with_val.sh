# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss

# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0,3')" MODEL.NAME "('se_resnext101')" \ 
# MODEL.PRETRAIN_PATH "('/home1/lizihan/pre_models/se_resnext101_32x4d-3b2fe3d8.pth')" \ 
# OUTPUT_DIR "('/home1/lizihan/reID/model/naic/Experiment-se_resnext101-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')"

# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('4,5')" MODEL.NAME "('resnet50_ibn_a')" \
# MODEL.PRETRAIN_PATH "('/home1/lizihan/pre_models/r50_ibn_a.pth')" \ 
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50_ibn_a-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')"

# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('4,5')" MODEL.NAME "('se_resnet101')" \ 
# MODEL.PRETRAIN_PATH "('/home1/lizihan/pre_models/se_resnet101-7e38fcc6.pth')" \
# OUTPUT_DIR "('/home1/lizihan/reID/model/naic/Experiment-se_resnet101-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')"

# triplet loss + center loss + softmax / oim
# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('4,5')" MODEL.NAME "('resnet50')" DATASETS.NAMES "('NAIC_val')" \
# MODEL.PRETRAIN_PATH "('/home/jiangming/.torch/models/resnet50-19c8e357.pth')" \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-val')" \
# MODEL.METRIC_LOSS_TYPE "('triplet_center_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss 
# #MODEL.METRIC_LOSS_TYPE "('triplet_center')"  MODEL.IF_LABELSMOOTH "('on')"


# triplet loss + softmax / oim
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0,1')" MODEL.NAME "('resnet50')" DATASETS.NAMES "('NAIC_val')" \
MODEL.PRETRAIN_PATH "('/home/jiangming/.torch/models/resnet50-19c8e357.pth')" \
OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-val')" \
MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"
#MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss 
#MODEL.METRIC_LOSS_TYPE "('triplet_center')"  MODEL.IF_LABELSMOOTH "('on')"
