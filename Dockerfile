# 训练容器
FROM python:3.6.8
# 作者和邮箱
MAINTAINER jiangming jiangming1406@buaa.edu.cn
# 将代码拷贝到/tmp/data/code
ADD . /tmp/data/code
# 进入代码目录
WORKDIR /tmp/data/code
# 安装必要的一些软件
RUN apt install python-pip
# 安装必要的一些软件
RUN pip install -r requirements.txt
# 运行训练代码
CMD sh Train.sh