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
# 解决依赖冲突：
# 1. pydantic 版本冲突：vllm==0.11.2 需要 pydantic>=2.12.0，但 requirements.txt 中指定了 pydantic<=2.10.6
# 2. transformers 版本冲突：vllm==0.11.2 需要 transformers>=4.56.0，但 lazyllm-llamafactory==0.9.3.dev0 需要 transformers<=4.51.3
# 3. torch 版本冲突：vllm==0.11.2 需要 torch==2.9.0（固定版本），但 lmdeploy==0.8.0 需要 torch<=2.6.0
# 因此需要排除这些包，让 vllm 的依赖自动安装兼容版本，其他包作为可选依赖单独处理
RUN set -x && \
    pip install --no-cache-dir -r requirements.txt && \
    # 先安装其他依赖（排除 flash-attn、pydantic、transformers、torch、lmdeploy 和 lazyllm-llamafactory）
    # 让 vllm 的依赖自动解决 transformers、pydantic 和 torch 版本
    grep -v "^flash-attn" requirements.full.txt | grep -v "^pydantic" | grep -v "^transformers" | grep -v "^torch" | grep -v "^lmdeploy" | grep -v "^lazyllm-llamafactory" > /tmp/requirements.full.noflash.txt || true && \
    pip install --no-cache-dir -r /tmp/requirements.full.noflash.txt && \
    # vllm 会自动安装 torch==2.9.0、torchvision==0.24.0、torchaudio==2.9.0 和 transformers>=4.56.0
    # 单独安装 flash-attn（禁用构建隔离，因为需要访问已安装的 torch）
    pip install --no-cache-dir --no-build-isolation flash-attn && \
    # lazyllm-llamafactory 版本说明：
    # - 0.9.3.dev0 要求 transformers<=4.51.3（与 vllm>=4.56.0 冲突）
    # - 0.9.4.dev2 只要求 transformers（无版本限制，兼容 vllm 的 transformers>=4.56.0）
    # 使用最新版本 0.9.4.dev2 以兼容 vllm 的 transformers 版本要求
    pip install --no-cache-dir lazyllm-llamafactory==0.9.4.dev2 && \
    # lmdeploy 版本说明：
    # - lmdeploy 0.8.0 要求 torch<=2.6.0（与 vllm 的 torch==2.9.0 冲突）
    # - lmdeploy 0.9.2 要求 torch<=2.7.1（仍然与 vllm 的 torch==2.9.0 冲突）
    # 由于 vllm 要求 torch==2.9.0 是固定版本，而 lmdeploy 最高只支持到 torch<=2.7.1，无法同时满足
    # lmdeploy 是可选依赖（主要用于 VLM 部署），如果安装失败则跳过，不影响其他功能
    (pip install --no-cache-dir lmdeploy==0.8.0 || echo "Warning: lmdeploy installation skipped due to torch version conflict with vllm (vllm requires torch==2.9.0, lmdeploy requires torch<=2.6.0)") && \
    pip install --no-cache-dir -e . && \
    pip cache purge

# 设置环境变量
ENV PYTHONPATH=/app
ENV LAZYLLM_DEFAULT_LAUNCHER=empty

# 暴露端口
EXPOSE 31340 31341

# 默认命令（可以在 docker-compose.yml 中覆盖）
CMD ["/bin/bash"]
