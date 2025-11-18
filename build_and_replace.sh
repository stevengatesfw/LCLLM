#!/bin/bash

# 本地构建 LazyLLM 镜像并替换 docker-compose.yml 中的镜像

set -e

# 配置
IMAGE_NAME="lazyllm-local"
IMAGE_TAG="0.6.4dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_COMPOSE_FILE="${SCRIPT_DIR}/../agentEasy/docker/docker-compose.yml"
OLD_IMAGE="registry.cn-hangzhou.aliyuncs.com/lazyllm/lazyllm:0.6.2"
NEW_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}构建本地 LazyLLM 镜像: ${NEW_IMAGE}${NC}"
echo ""

# 1. 构建镜像
echo -e "${GREEN}开始构建 Docker 镜像...${NC}"

# 确保有 .dockerignore 文件
if [ ! -f ".dockerignore" ]; then
    cat > .dockerignore << 'EOF'
.git
.gitignore
__pycache__
*.pyc
*.pyo
*.pyd
*.so
*.egg
*.egg-info
dist
build
.pytest_cache
.vscode
.idea
*.swp
*.swo
*~
*.tmp
*.bak
.DS_Store
*.md
*.log
!requirements.txt
!requirements.full.txt
# 排除大文件和目录
Tutorial
*.mp4
*.avi
*.mov
*.mkv
*.zip
*.tar.gz
*.tar
*.gz
examples
docs
tests
.github
EOF
fi

docker build --progress=plain -t ${NEW_IMAGE} .
BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ 镜像构建失败${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 镜像构建成功${NC}"

# 2. 更新 docker-compose.yml
echo -e "${GREEN}更新 docker-compose.yml...${NC}"
if [ ! -f "${DOCKER_COMPOSE_FILE}" ]; then
    echo -e "${YELLOW}⚠️  未找到 docker-compose.yml: ${DOCKER_COMPOSE_FILE}${NC}"
    echo "   请手动更新镜像地址为: ${NEW_IMAGE}"
else
    # 备份原文件
    BACKUP_FILE="${DOCKER_COMPOSE_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    cp "${DOCKER_COMPOSE_FILE}" "${BACKUP_FILE}"
    
    # 检查是否已经使用新镜像
    if ! grep -q "${NEW_IMAGE}" "${DOCKER_COMPOSE_FILE}"; then
        # 替换镜像地址（兼容 macOS 和 Linux）
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|${OLD_IMAGE}|${NEW_IMAGE}|g" "${DOCKER_COMPOSE_FILE}"
        else
            sed -i "s|${OLD_IMAGE}|${NEW_IMAGE}|g" "${DOCKER_COMPOSE_FILE}"
        fi
    fi
    echo -e "${GREEN}✅ 已更新 docker-compose.yml${NC}"
fi

# 3. 询问是否自动重启服务
echo ""
read -p "是否现在自动重启 cloud-service? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    DOCKER_COMPOSE_DIR="$(dirname "${DOCKER_COMPOSE_FILE}")"
    cd "${DOCKER_COMPOSE_DIR}"
    docker compose stop cloud-service 2>/dev/null || true
    docker compose rm -f cloud-service 2>/dev/null || true
    docker compose up -d cloud-service
    echo -e "${GREEN}✅ 服务已重启${NC}"
fi

