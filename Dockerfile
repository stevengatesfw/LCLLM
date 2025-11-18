FROM registry.cn-hangzhou.aliyuncs.com/lazyllm/python:3.10-slim

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=Asia/Shanghai
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 设置时区
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 替换为阿里云 Debian 镜像源
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null || true

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    libnuma-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* || true

# 设置工作目录
WORKDIR /app

# 复制源码
COPY . /app/

# 安装 Python 依赖
# 先安装 torch（flash-attn 需要 torch 已安装）
# 在 CPU 环境下跳过 flash-attn 和 lmdeploy（需要 CUDA 或特定架构）
RUN set -x && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "torch>=2.1.2" && \
    grep -v "^flash-attn" requirements.full.txt | grep -v "^lmdeploy" | grep -v "^vllm" | grep -v "^zlib-state" > /tmp/requirements.full.filtered.txt || true && \
    pip install --no-cache-dir -r /tmp/requirements.full.filtered.txt || true && \
    pip install --no-cache-dir -e . && \
    pip cache purge

# 设置环境变量
ENV PYTHONPATH=/app
ENV LAZYLLM_DEFAULT_LAUNCHER=empty

# 暴露端口
EXPOSE 31340 31341

# 默认命令（可以在 docker-compose.yml 中覆盖）
CMD ["/bin/bash"]

