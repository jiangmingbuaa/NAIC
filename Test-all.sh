# triplet + oim: resnext101_ibn_a
# python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnext101_ibn_a')" DATASETS.NAMES "('NAIC')"  \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
# DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnext101_ibn_a_model_80.pth')" \
# TEST.RE_RANKING "('yes')" # re-ranking

# triplet + oim: se_resnet101_ibn_a
# python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" DATASETS.NAMES "('NAIC')"  \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
# DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/se_resnet101_ibn_a_model_80.pth')" \
# TEST.RE_RANKING "('yes')" # re-ranking

# triplet + oim: densenet169_ibn_a
# python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('densenet169_ibn_a')" DATASETS.NAMES "('NAIC')"  \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
# DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/densenet169_ibn_a_model_80.pth')" \
# TEST.RE_RANKING "('yes')" # re-ranking

# triplet + oim: resnet50_ibn_a
python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005/resnet50_ibn_a_model_80.pth')" \
TEST.RE_RANKING "('yes')" # re-ranking