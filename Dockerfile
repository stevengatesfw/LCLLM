# 使用 NVIDIA 官方 CUDA 镜像（解决 CUDA 头文件冲突问题）
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

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

# 设置编译选项（用于 flash-attn）
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
ENV MAX_JOBS=1

# 设置工作目录
WORKDIR /app

# 复制源码
COPY . /app/

# 安装 Python 依赖
# 先安装 torch（flash-attn 需要 torch 已安装）
# 指定 CUDA 12.6 版本的 torch 以匹配基础镜像
# 注意：如果 cu126 不可用，可以使用 cu121（CUDA 12.6 向后兼容 12.1）
RUN set -x && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 "torch>=2.1.2" torchvision torchaudio || \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "torch>=2.1.2" torchvision torchaudio && \
    # 先安装其他依赖（排除 flash-attn）
    grep -v "^flash-attn" requirements.full.txt > /tmp/requirements.full.noflash.txt || true && \
    pip install --no-cache-dir -r /tmp/requirements.full.noflash.txt && \
    # 单独安装 flash-attn（禁用构建隔离，因为需要访问已安装的 torch）
    pip install --no-cache-dir --no-build-isolation flash-attn && \
    pip install --no-cache-dir -e . && \
    pip cache purge

# 设置环境变量
ENV PYTHONPATH=/app
ENV LAZYLLM_DEFAULT_LAUNCHER=empty

# 暴露端口
EXPOSE 31340 31341

# 默认命令（可以在 docker-compose.yml 中覆盖）
CMD ["/bin/bash"]
