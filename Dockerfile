# 使用 NVIDIA 官方 CUDA 镜像（解决 CUDA 头文件冲突问题）
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 设置环境变量
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=Asia/Shanghai
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV DEBIAN_FRONTEND=noninteractive

# 设置时区
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 替换为阿里云 Ubuntu 镜像源（加速下载）
RUN sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list || true

# 安装 Python 3.10 和系统依赖
# libgl1 用于 paddleocr/opencv（在 Ubuntu 22.04+ 中，libgl1-mesa-glx 已被替换为 libgl1）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    gcc \
    g++ \
    git \
    curl \
    libnuma-dev \
    zlib1g-dev \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建 python 和 pip 的软链接
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA 环境变量（NVIDIA 镜像已自动设置，显式设置以确保正确）
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置工作目录
WORKDIR /app

# 复制源码
COPY . /app/

# 安装 Python 依赖，flash-attn 后续通过 wheel 手动安装 https://github.com/Dao-AILab/flash-attention/releases
# 注意：vllm 0.12.0 需要 torch==2.9.0，而 lmdeploy 0.11.1 需要 torch<=2.8.0
RUN set -x && \
    grep -v "^flash-attn" requirements.full.txt | grep -v "^gradio" | grep -v "^lmdeploy" > /tmp/requirements.base.txt && \
    pip install --no-cache-dir -r /tmp/requirements.base.txt && \
    pip install --no-cache-dir --no-deps gradio==5.49.1 && \
    pip install --no-cache-dir --no-deps lmdeploy==0.11.1 || echo "Warning: lmdeploy installation skipped (torch version conflict)" && \
    pip install --no-cache-dir -e . && \
    pip cache purge

# 安装最新版本的 diffusers 以支持 Z-Image（需要从 GitHub 安装，使用代理）
RUN python -c 'from diffusers import ZImagePipeline' 2>/dev/null || ( \
    echo 'Installing latest diffusers from GitHub for Z-Image support...' && \
    pip install --upgrade --no-cache-dir 'git+https://githubproxy.cc/https://github.com/huggingface/diffusers' 2>&1 | tail -10 || echo 'Warning: diffusers installation may have failed, but continuing...' && \
    pip install --upgrade --no-cache-dir 'peft>=0.17.0' 2>&1 | tail -5 || echo 'Warning: peft upgrade may have failed, but continuing...' \
    )

# 设置环境变量
ENV PYTHONPATH=/app
ENV LAZYLLM_DEFAULT_LAUNCHER=empty

# 暴露端口
EXPOSE 31340 31341

# 默认命令（可以在 docker-compose.yml 中覆盖）
CMD ["/bin/bash"]
