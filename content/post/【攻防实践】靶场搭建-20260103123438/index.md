+++
title= "【攻防实践】靶场搭建"
date= "2026-01-03T12:33:34+08:00"
lastmod= "2026-01-03T12:33:34+08:00"
draft=false
author= "leuco"
description= "【0】一些常见的靶场搭建"
keywords= ["Docker", "靶场", "PHP"]
categories= ["网络空间安全", "攻防实践"]
tags= ["Docker", "靶场", "PHP"]
math=true

+++

# 靶场搭建

{{<toc>}}

## 靶场环境

Ubuntu 24.01 64bit

## Pikachu-Docker

### 安装Docker

```bash
sudo apt update # 更新软件包列表
sudo install docker.io docker-compose -y # 安装Docker和Docker Compose
sudo systemctl start docker # 启动Docker服务
sudo systemctl enable docker # 设置Docker开机自启动 
docker --version # 查看Docker版本以验证安装
```

### 配置Docker镜像

- 编辑docker配置文件

```bash
sudo nano /etc/docker/daemon.json # 编辑Docekrs守护进程配置
```

- 添加镜像加速地址

```json
{
    "registry-mirrors": [
        "https://docker.1ms.run",
   	 	"https://docker-0.unsee.tech",
    	"https://docker.m.daocloud.io"
    ]
} // 写入国内镜像加速器
```

- 保存并重启Docker服务

```bash
sudo systemctl daemon-reload # 重新加载 systemd 守护进程配置
sudo systemctl restart docker # 重启 Docker 服务使配置生效
```

### 拉取Pikachu镜像

```bash
sudo docker pull area39/pikachu
```

### 运行容器

```bash
sudo docker run -d --name pikachu -p 8000:80 area39/pikachu
```

### 持久化运行

```bash
sudo docker stop pikachu
sudo docker rm pikachu # 停止并删除容器
```

```bash
# 创建命名数据卷
sudo docker volume create pikachu_data
```

- 启动脚本（pikachu-start.bash）:

```bash
#!/bin/bash
# pikachu-manager.sh - Pikachu Docker 容器管理脚本

CONTAINER_NAME="pikachu"
IMAGE_NAME="area39/pikachu"
PORT="8001:80"
VOLUME_NAME="pikachu_data"
VOLUME_PATH="/app/data"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_usage() {
    echo -e "${BLUE}Pikachu 容器管理脚本${NC}"
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start    启动容器"
    echo "  stop     停止容器"
    echo "  restart  重启容器"
    echo "  status   查看容器状态"
    echo "  logs     查看容器日志"
    echo "  remove   删除容器（保留数据卷）"
    echo "  purge    彻底删除（容器+数据卷）"
    echo "  backup   备份数据卷"
    echo "  shell    进入容器shell"
    echo ""
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: Docker 未安装${NC}"
        exit 1
    fi
    
    if ! sudo docker info &> /dev/null; then
        echo -e "${RED}错误: Docker 服务未运行或无权限${NC}"
        echo "请确保Docker服务正在运行，或使用sudo权限"
        exit 1
    fi
}

start_container() {
    check_docker
    
    echo -e "${BLUE}正在启动 Pikachu 容器...${NC}"
    
    # 检查镜像是否存在，不存在则拉取
    if ! sudo docker images | grep -q "$(echo $IMAGE_NAME | cut -d':' -f1)"; then
        echo -e "${YELLOW}正在拉取镜像...${NC}"
        sudo docker pull ${IMAGE_NAME}
    fi
    
    # 检查是否已存在同名容器
    if sudo docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        if sudo docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "${YELLOW}容器已在运行中${NC}"
            return
        else
            echo -e "${YELLOW}发现停止的容器，正在启动...${NC}"
            sudo docker start ${CONTAINER_NAME}
        fi
    else
        # 创建数据卷（如果不存在）
        if ! sudo docker volume ls | grep -q ${VOLUME_NAME}; then
            echo -e "${YELLOW}创建数据卷: ${VOLUME_NAME}${NC}"
            sudo docker volume create ${VOLUME_NAME}
        fi
        
        # 运行新容器
        sudo docker run -d \
            --name ${CONTAINER_NAME} \
            --restart unless-stopped \
            -p ${PORT} \
            -v ${VOLUME_NAME}:${VOLUME_PATH} \
            ${IMAGE_NAME}
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Pikachu 容器启动成功！${NC}"
        echo -e "${BLUE}容器信息:${NC}"
        echo "  名称: ${CONTAINER_NAME}"
        echo "  镜像: ${IMAGE_NAME}"
        echo "  端口: ${PORT}"
        echo "  数据卷: ${VOLUME_NAME}"
        echo -e "${BLUE}访问地址:${NC} http://localhost:8001"
    else
        echo -e "${RED}✗ 容器启动失败${NC}"
        exit 1
    fi
}

stop_container() {
    check_docker
    
    echo -e "${YELLOW}正在停止容器...${NC}"
    sudo docker stop ${CONTAINER_NAME}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 容器已停止${NC}"
    else
        echo -e "${RED}✗ 停止容器失败${NC}"
    fi
}

restart_container() {
    check_docker
    
    echo -e "${YELLOW}正在重启容器...${NC}"
    sudo docker restart ${CONTAINER_NAME}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 容器已重启${NC}"
    else
        echo -e "${RED}✗ 重启容器失败${NC}"
    fi
}

show_status() {
    check_docker
    
    echo -e "${BLUE}容器状态:${NC}"
    sudo docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"
    
    echo -e "\n${BLUE}数据卷信息:${NC}"
    sudo docker volume ls --filter "name=${VOLUME_NAME}"
    
    echo -e "\n${BLUE}资源使用情况:${NC}"
    sudo docker stats ${CONTAINER_NAME} --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

show_logs() {
    check_docker
    
    echo -e "${BLUE}容器日志 (最后50行):${NC}"
    sudo docker logs --tail 50 ${CONTAINER_NAME}
}

remove_container() {
    check_docker
    
    echo -e "${YELLOW}正在删除容器...${NC}"
    sudo docker rm ${CONTAINER_NAME}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 容器已删除（数据卷保留）${NC}"
    else
        echo -e "${RED}✗ 删除容器失败${NC}"
    fi
}

purge_all() {
    check_docker
    
    echo -e "${RED}警告: 这将删除容器和数据卷！${NC}"
    read -p "确认删除？(y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}正在删除容器...${NC}"
        sudo docker rm -f ${CONTAINER_NAME} 2>/dev/null
        
        echo -e "${YELLOW}正在删除数据卷...${NC}"
        sudo docker volume rm ${VOLUME_NAME} 2>/dev/null
        
        echo -e "${GREEN}✓ 容器和数据卷已彻底删除${NC}"
    else
        echo -e "${YELLOW}操作已取消${NC}"
    fi
}

backup_volume() {
    check_docker
    
    BACKUP_DIR="./backups"
    BACKUP_FILE="pikachu_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    mkdir -p ${BACKUP_DIR}
    
    echo -e "${BLUE}正在备份数据卷...${NC}"
    
    # 停止容器以确保数据一致性
    sudo docker stop ${CONTAINER_NAME} 2>/dev/null
    
    # 创建备份
    sudo docker run --rm \
        -v ${VOLUME_NAME}:/source \
        -v ${BACKUP_DIR}:/backup \
        alpine tar czf /backup/${BACKUP_FILE} -C /source .
    
    # 重新启动容器
    sudo docker start ${CONTAINER_NAME} 2>/dev/null
    
    echo -e "${GREEN}✓ 备份完成: ${BACKUP_DIR}/${BACKUP_FILE}${NC}"
    ls -lh ${BACKUP_DIR}/${BACKUP_FILE}
}

enter_shell() {
    check_docker
    
    echo -e "${BLUE}进入容器Shell...${NC}"
    echo -e "${YELLOW}按 Ctrl+D 或输入 exit 退出${NC}"
    sudo docker exec -it ${CONTAINER_NAME} /bin/sh
}

# 主程序
case "$1" in
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    remove)
        remove_container
        ;;
    purge)
        purge_all
        ;;
    backup)
        backup_volume
        ;;
    shell)
        enter_shell
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
```













