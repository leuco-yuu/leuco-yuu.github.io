---
title: "靶场搭建"
date: 2026-01-03T12:33:34+08:00
lastmod: 2026-06-02T12:33:49+08:00
draft: false
slug: cyber-range-setup
series_order: 4
description: "常见的靶场搭建"
summary: "整理 DVWA、Pikachu、WebGoat、Vulhub 等常见靶场环境的搭建方式。"
tags:
  - "Docker"
  - "靶场"
  - "DVWA"
  - "Pikachu"
  - "WebGoat"
  - "Vulhub"
categories:
  - "网络空间安全"
  - "攻防实践"
series:
  - "网络安全实践"
---

#  环境与 Docker

- **系统环境**：Kali

- **Docker 部署**：

```bash
# 更新并安装 Docker 和 Docker-Compose
sudo apt update
sudo apt install -y docker.io docker-compose

# 启动 Docker 服务，设置为开机自启动
sudo systemctl enable docker --now

# 将当前用户加入用户组，免 Docker 执行
sudo username -aG docker $USER

# 检查服务状态
# sudo systemctl status docker

# 免重启刷新当前终端用户组权限
# newgrp docker
```

# DVWA（Damn Vulnerable Web App）

- **简介**：DVWA是最经典的 PHP/MySQL 漏洞演示平台，包含了 SQL 注入、XSS、CSRF、文件包含等最核心的 Web 漏洞，并且可以直接在界面上调整安全级别（Low/Medium/High/Impossible），适合用来观察底层防御代码的逻辑变化。
- **初始化部署**：
```bash
docker run -d \
  --name dvwa_range \
  -p 8001:80 
  --restart=unless-stopped 
  vulnerables/web-dvwa	
```
  - **参数说明**：
    - `-d`：后台静默运行
    - `--name dvwa_range`：靶场命名
    - `-p 8001:80`：将 Kali 宿主机的端口（8001）映射到容器内部网页端口（DVWA固定为80）
    - `--restart=unless-stopped`：崩溃重启机制

  - **靶场管理**：

```bash
docker stop dvwa_range   # 暂停靶场
docker start dvwa_range  # 重启靶场（不会删除数据）
docker -rm -f dvwa_range # 彻底销毁靶场
```

- **初始化**：
  - 浏览器访问 `http://localhost:8001`
  - 默认账号密码：`admin / password`
  - 登录后点击左侧 `Setup / Resart DB` -> `Create / Reset Database`

# Pikachu（皮卡丘）

- **简介**：相比于 DVWA，Pikachu 的漏洞类型更贴近国内的安全测试习惯，涵盖了 DVWA 没有的 SSRF、XXE、反序列化、越权漏洞等。带有非常详尽的中文漏洞说明和通关提示。
- **初始化部署**：

```bash
docker run -d \
  --name pikachu_range \
  -p 8002:80 \
  --restart=unless-stopped \
  area39/pikachu
```

- **参数说明**：同上
- **初始化数据库**：
  - 浏览器访问 `http://127.0.0.1:8002`
  - 按提示完成初始化部署
- **靶场管理**：同上

# WebGoat（OWASP 官方靶场）

- **简介**：企业级应用大量基于 Java (Spring Boot) 开发。WebGoat 是 OWASP 维护的项目，专注于现代 Web 应用的安全风险（如 JWT 漏洞、XXE、身份认证缺陷）。其形式像闯关教程，每一步都有理论解析。
- **初始化部署**：

```bash
docker run -d \
  --name webgoat_range \
  -p 8003:8080 \
  -p 8004:9090 \
  -e TZ=Asia/Shanghai \
  --restart=unless-stopped \
  webgoat/webgoat
```

- **参数说明**
  - `-p 8003:8080`: Kali 的 8003 端口映射到 WebGoat 本身的 8080 端口。
  - `-p 8004:9090`: Kali 的 8004 端口映射到 WebWolf 的辅助测试端口（9090）。
  - `-e TZ=Asia/Shanghai`: 环境变量注入。Java 程序的日志和某些时间相关的漏洞（比如 JWT 令牌过期测试）非常依赖系统时间，强制容器使用北京时间。
- **账号注册与访问机制**：
  -  浏览器必须精确访问 `http://127.0.0.1:8003/WebGoat`。
  - 页面打开后没有默认账号密码，需要通过 **“Register new user”** 注册。
  - WebGoat 支持多用户隔离。
  - 如需使用 WebWolf，访问 `http://127.0.0.1:8004/WebWolf`

# Vulhub（真实 CVE 漏洞复现）

- **简介**：Vulhub 是一个基于 Docker-Compose 的漏洞环境集合。在研究某个具体的组件（比如 Nginx 解析漏洞、Redis 未授权访问、WebLogic 反序列化）或是复现某个真实世界的 CVE 时，Vulhub 是无可替代的。
- **初始化部署**：

```bash
# 克隆 Vulhub 仓库到本地
git clone https://github.com/vulhub/vulhub.git
cd vulhub

# 假设复现 Struts2 的 s2-052 漏洞
cd struts2/s2-052

# 启动该漏洞容器环境
sudo docker-compose up -d
# 获取访问端口
doker-compse ps

# 测试完毕后，销毁环境
sudo docker-compose down
```
