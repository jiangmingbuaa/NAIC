python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/tmp/data/model/')" DATASETS.QUERY_DIR "('query_B')" DATASETS.GALLERY_DIR "('gallery_B')" \
DATASETS.ROOT_DIR "('/tmp/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/tmp/data/model/Experiment-se_resnet101_ibn_a_stage_1/se_resnet101_ibn_a_model_80.pth')"  TEST.TEST_MODEL "stage_1_se_resnet101_80" 

python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/tmp/data/model/')" DATASETS.QUERY_DIR "('query_B')" DATASETS.GALLERY_DIR "('gallery_B')" \
DATASETS.ROOT_DIR "('/tmp/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/tmp/data/model/Experiment-se_resnet101_ibn_a_stage_1/se_resnet101_ibn_a_model_90.pth')"  TEST.TEST_MODEL "stage_1_se_resnet101_90"

python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/tmp/data/model/')" DATASETS.QUERY_DIR "('query_B')" DATASETS.GALLERY_DIR "('gallery_B')" \
DATASETS.ROOT_DIR "('/tmp/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/tmp/data/model/Experiment-resnet50_ibn_a_stage_2/resnet50_ibn_a_model_90.pth')"  TEST.TEST_MODEL "stage_2_resnet50_90" 

python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('se_resnet101_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/tmp/data/model/')" DATASETS.QUERY_DIR "('query_B')" DATASETS.GALLERY_DIR "('gallery_B')" \
DATASETS.ROOT_DIR "('/tmp/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/tmp/data/model/Experiment-se_resnet101_ibn_a_stage_3/se_resnet101_ibn_a_model_20.pth')"  TEST.TEST_MODEL "stage_3_se_resnet101_20" 

python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('resnet50_ibn_a')" DATASETS.NAMES "('NAIC')"  \
OUTPUT_DIR "('/tmp/data/model/')" DATASETS.QUERY_DIR "('query_B')" DATASETS.GALLERY_DIR "('gallery_B')" \
DATASETS.ROOT_DIR "('/tmp/data')" MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/tmp/data/model/Experiment-resnet50_ibn_a_stage_3/resnet50_ibn_a_model_20.pth')"  TEST.TEST_MODEL "stage_3_resnet50_20" 

python3 ensemble_b.py