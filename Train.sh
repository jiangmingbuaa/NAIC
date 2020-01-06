# # oim, se_resnet101_ibn_a
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" \
MODEL.PRETRAIN_PATH "('/tmp/data/pretrain_model/se_resnet101_ibn_a.pth')"  SOLVER.IMS_PER_BATCH "(360)" \
OUTPUT_DIR "('/tmp/data/model/Experiment-se_resnet101_ibn_a_stage_1')" \
MODEL.METRIC_LOSS_TYPE "('oim')"  MODEL.IF_LABELSMOOTH "('off')"  SOLVER.OIM_MARGIN "(0.2)" 

python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/tmp/data/model/')" DATASETS.QUERY_DIR "('query_A')" DATASETS.GALLERY_DIR "('gallery_A')" \
DATASETS.ROOT_DIR "('/tmp/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/tmp/data/model/Experiment-se_resnet101_ibn_a_stage_1/se_resnet101_ibn_a_model_80.pth')"  TEST.TEST_MODEL "stage_1_se_resnet101_80" 

# triplet + oim, resnet50_ibn_a
# python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" \
# MODEL.PRETRAIN_PATH "('/tmp/data/pretrain_model/resnet50_ibn_a.pth')"  SOLVER.IMS_PER_BATCH "('336')" \
# OUTPUT_DIR "('/tmp/data/model/Experiment-resnet50_ibn_a_stage_1')" \
# MODEL.METRIC_LOSS_TYPE "('oim')"  MODEL.IF_LABELSMOOTH "('off')"  SOLVER.OIM_MARGIN "('0.2')"

# python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" DATASETS.NAMES "('NAIC')"  \
# OUTPUT_DIR "('/tmp/data/model/')" DATASETS.QUERY_DIR "('query_A')" DATASETS.GALLERY_DIR "('gallery_A')" \
# DATASETS.ROOT_DIR "('/tmp/data')" MODEL.PRETRAIN_CHOICE "('self')" \
# TEST.WEIGHT "('/tmp/data/model/Experiment-resnet50_ibn_a_stage_1/resnet50_ibn_a_model_80.pth')"  TEST.TEST_MODEL "stage_1_resnet50_80" 

python3 cluster_a.py

# triplet + oim, resnet50_ibn_a
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/tmp/data/pretrain_model/resnet50_ibn_a.pth')"  SOLVER.IMS_PER_BATCH "(560)" \
OUTPUT_DIR "('/tmp/data/model/Experiment-resnet50_ibn_a_stage_2')" \
MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  SOLVER.OIM_MARGIN "(0.1)"

python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/tmp/data/model/')" DATASETS.QUERY_DIR "('query_B')" DATASETS.GALLERY_DIR "('gallery_B')" \
DATASETS.ROOT_DIR "('/tmp/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/tmp/data/model/Experiment-resnet50_ibn_a_stage_2/resnet50_ibn_a_model_80.pth')"  TEST.TEST_MODEL "stage_2_resnet50_80" 

python3 cluster_b.py

# triplet + oim, resnet50_ibn_a
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/tmp/data/model/Experiment-resnet50_ibn_a_stage_2/resnet50_ibn_a_model_80.pth')" MODEL.PRETRAIN_CHOICE "('self')" \
OUTPUT_DIR "('/tmp/data/model/Experiment-resnet50_ibn_a_stage_3')" SOLVER.IMS_PER_BATCH "(560)" \
MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')" SOLVER.MAX_EPOCHS "(20)" SOLVER.BASE_LR "(0.000035)"

# triplet + oim, se_resnet101_ibn_a
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" \
MODEL.PRETRAIN_PATH "('/tmp/data/model/Experiment-se_resnet101_ibn_a_stage_1/se_resnet101_ibn_a_model_80.pth')" MODEL.PRETRAIN_CHOICE "('self')" \
OUTPUT_DIR "('/tmp/data/model/Experiment-se_resnet101_ibn_a_stage_3')" SOLVER.IMS_PER_BATCH "(360)" \
MODEL.METRIC_LOSS_TYPE "('triplet_oim')"  MODEL.IF_LABELSMOOTH "('off')"  SOLVER.MAX_EPOCHS "(20)" SOLVER.BASE_LR "(0.000035)"