# NAIC

## 项目说明
NAIC复赛代码，在[Strong Baseline](https://github.com/michuanhaohao/reid-strong-baseline)的基础上进行改进。
具体说明详见提交的K-Lab Notebook。

项目使用ECCV 2018论文Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net中提出的IBN-Net作为预训练模型，具体使用了其中2种模型：`ResNet50_ibn_a`和`Se-ResNet101_ibn_a`，
初始化用到的的[预训练模型](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S)，下载的预训练权重放在```/tmp/data/pretrain_model```目录下。

训练预计时间41小时，测试预计时间7小时。项目在容器中的路径：```/tmp/data/code```

## 项目的运行步骤
1. 下载代码：```git clone -b stage2 https://github.com/jiangmingbuaa/NAIC```，切换到项目根目录。

2. 训练：

   在终端输入：

   docker build -t train .

   docker run train

3. 测试：

   删除原来的```Dockerfile```文件，修改```Dockerfile_test```文件名为```Dockerfile```，在终端输入：

   docker build -t test .

   docker run test

训练过程中需要读取并修改```/tmp/data/train/label/train_list.txt```，模型和中间结果保存在```/tmp/data/model```中，最终结果保存在```/tmp/data/answer```中。

## 运行结果的位置
1. 执行上述步骤后，最终结果保存在```/tmp/data/answer/NAIC_ensemble_submission_2b.json```文件中。
