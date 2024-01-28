# 使用Python镜像
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

# 设置工作目录
WORKDIR /app

# 复制并安装后端依赖
COPY requirements.txt ./
RUN apt-get update && apt-get install -y python3-pip
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# 复制其他后端代码
COPY . .

# ENV FLASK_APP app.py
ENV FLASK_APP /app/app.py
ENV FLASK_RUN_HOST 0.0.0.0

EXPOSE 5000

# CMD [ "bash" ]
# CMD flask run --host=0.0.0.0
CMD tail -f /dev/null

