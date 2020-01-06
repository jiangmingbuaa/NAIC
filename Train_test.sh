# # oim, se_resnet101_ibn_a
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" \
MODEL.PRETRAIN_PATH "('/root/share/pretrained_models/se_resnet101_ibn_a.pth.tar')"  SOLVER.IMS_PER_BATCH "(60)" \
OUTPUT_DIR "('/root/data/reid/NAIC/output/Experiment-se_resnet101_ibn_a_stage_1')" \
MODEL.METRIC_LOSS_TYPE "('oim')"  MODEL.IF_LABELSMOOTH "('off')"  SOLVER.OIM_MARGIN "(0.2)" 
