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

# python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('4')" MODEL.NAME "('resnet50')" DATASETS.NAMES "('NAIC_val')"  \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/')" \
# DATASETS.ROOT_DIR "('/home1/lizihan/reID/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-val/resnet50_model_120.pth')" \
#TEST.RE_RANKING "('yes')" # re-ranking


# triplet loss + softmax loss / oim loss
python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('6')" MODEL.NAME "('resnet50_ibn_a')" DATASETS.NAMES "('NAIC_val')"  \
OUTPUT_DIR "('/root/data/reid/NAIC/output/')" \
DATASETS.ROOT_DIR "('/root/share/dataset/reid')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/root/data/reid/NAIC/output/Experiment-resnet50_ibn_a_val_2/resnet50_ibn_a_model_80.pth')" \
# TEST.RE_RANKING "('yes')" # re-ranking