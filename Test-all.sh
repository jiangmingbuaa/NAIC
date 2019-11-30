# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# without re-ranking
# python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('4')" DATASETS.NAMES "('NAIC')" DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home1/lizihan/reID/model/naic/Experiment-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnet50_model_120.pth')"

#python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('4')" MODEL.NAME "('resnet50_ibn_a')" DATASETS.NAMES "('NAIC')" DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home1/lizihan/reID/model/naic/Experiment-resnet50_ibn_a-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnet50_ibn_a_model_120.pth')"
# python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('4')" MODEL.NAME "('resnet50')" DATASETS.NAMES "('NAIC')"  \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
# DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnet50_model_120.pth')" \
# #TEST.RE_RANKING "('yes')" # re-ranking


# triplet + oim: resnext 101
# python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('7')" MODEL.NAME "('resnext101_ibn_a')" DATASETS.NAMES "('NAIC')"  \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
# DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnext101_ibn_a_model_60_a.pth')" \
# TEST.RE_RANKING "('yes')" # re-ranking

# triplet + oim: se_resnet101
# python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" DATASETS.NAMES "('NAIC')"  \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
# DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/se_resnet101_ibn_a_model_80_a.pth')" \
# TEST.RE_RANKING "('yes')" # re-ranking

# triplet + oim:
# python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('7')" MODEL.NAME "('densenet169_ibn_a')" DATASETS.NAMES "('NAIC')"  \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
# DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/densenet169_ibn_a_model_80_a.pth')" \
# TEST.RE_RANKING "('yes')" # re-ranking

# triplet + oim:
python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('7')" MODEL.NAME "('resnet50_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnet50_ibn_a_model_13_80.pth')" \
TEST.RE_RANKING "('yes')" # re-ranking