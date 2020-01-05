# triplet + center + oim
# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0,1')" MODEL.NAME "('resnet50')" \
# MODEL.PRETRAIN_PATH "('/home/jiangming/.torch/models/resnet50-19c8e357.pth')" \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')" \
# MODEL.METRIC_LOSS_TYPE "('triplet_center_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss

# # triplet + oim, resnet50
# python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0,1')" MODEL.NAME "('resnet50')" \
# MODEL.PRETRAIN_PATH "('/home/jiangming/.torch/models/resnet50-19c8e357.pth')" \
# OUTPUT_DIR "('/home/jiangming/re-id/NAIC/reid_strong_baseline/output/Experiment-resnet50-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005')" \
# MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss

# triplet + oim, resnet50_ibn_a
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/root/share/pretrained_models/r50_ibn_a.pth')" \
OUTPUT_DIR "('/root/data/reid/NAIC/output/Experiment-resnet50_ibn_a_3')" \
MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss

# # triplet + oim, resnext101_ibn_a
# python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0,1,2')" MODEL.NAME "('resnext101_ibn_a')" \
# MODEL.PRETRAIN_PATH "('/root/share/pretrained_models/resnext101_ibn_a.pth.tar')" \
# OUTPUT_DIR "('/root/data/reid/NAIC/output/Experiment-resnext101_ibn_a')" \
# MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss

# triplet + oim, densenet169_ibn_a
# python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0,1')" MODEL.NAME "('densenet169_ibn_a')" \
# MODEL.PRETRAIN_PATH "('/root/share/pretrained_models/densenet169_ibn_a.pth.tar')" \
# OUTPUT_DIR "('/root/data/reid/NAIC/output/Experiment-densenet169_ibn_a')" \
# MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss

# triplet + oim, densenet169_ibn_a
# python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0,1,2')" MODEL.NAME "('densenet121_ibn_a')" \
# MODEL.PRETRAIN_PATH "('/root/share/pretrained_models/densenet121_ibn_a.pth.tar')" \
# OUTPUT_DIR "('/root/data/reid/NAIC/output/Experiment-densenet121_ibn_a')" \
# MODEL.METRIC_LOSS_TYPE "('oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss

# triplet + oim, se_resnet101_ibn_a
# python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0,1,2')" MODEL.NAME "('se_resnet101_ibn_a')" \
# MODEL.PRETRAIN_PATH "('/root/share/pretrained_models/se_resnet101_ibn_a.pth.tar')" \
# OUTPUT_DIR "('/root/data/reid/NAIC/output/Experiment-se_resnet101_ibn_a')" \
# MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss

# triplet + oim, aognet
# python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('aognet')" \
# MODEL.PRETRAIN_PATH "('/root/share/pretrained_models/aognet-40m/model_best.pth.tar')" \
# OUTPUT_DIR "('/root/data/reid/NAIC/output/Experiment-aognet')" \
# MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  ### oim loss