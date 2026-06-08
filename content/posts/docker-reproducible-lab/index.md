---
title: "用 Docker 固化一套可复现实验环境"
date: 2026-06-01T10:30:00+08:00
lastmod: 2026-06-01T10:30:00+08:00
draft: false
slug: docker-reproducible-lab
homepage: true
series_order: 2
description: "把临时实验环境整理成可复现、可回滚、可提交的 Docker 工作流。"
summary: "记录一次把本机实验环境拆成镜像、卷、网络和脚本的过程，让后续复盘不再依赖记忆。"
tags:
  - Docker
  - Linux
  - DevOps
  - 可复现环境
categories:
  - 工程实践
  - Linux基础
series:
  - 系统工程
cover: cover.jpg
---

很多实验一开始只是临时跑通，最后却变成长期依赖。真正麻烦的地方不在第一次成功，而在两周后想复现时发现端口、环境变量、依赖版本都只留在终端历史里。

这次整理的目标很简单：让实验环境可以被提交、被重建、被删除，也可以在另一台机器上尽量少解释地跑起来。

# 目录结构

我把实验目录拆成四层：

```text
lab/
  compose.yaml
  .env.example
  services/
  scripts/
  data/
```

`compose.yaml` 描述服务关系，`.env.example` 记录必须提供的变量，`services/` 放每个服务自己的 Dockerfile 或配置，`scripts/` 放初始化、清理和诊断脚本。`data/` 只保存本地运行产物，不进 Git。

# 镜像和状态分离

镜像负责软件版本，卷负责运行状态。只要这两件事混在一起，环境就会逐渐变成一个不可解释的黑盒。

实践里我会尽量让镜像构建结果是无状态的：依赖在 Dockerfile 里固定版本，运行时生成的数据进入 volume。这样升级依赖时可以重新构建镜像，遇到脏数据时可以单独清卷。

# 入口脚本

每个实验目录至少保留三个命令：

```powershell
docker compose up -d
docker compose logs -f
docker compose down
```

如果初始化步骤超过一条命令，就放进 `scripts/init.ps1`。这不是为了追求形式，而是为了让复盘时知道“当时到底做了什么”。

# 小结

可复现环境的关键不是把所有东西容器化，而是把环境的边界写清楚：哪些东西能删，哪些东西要备份，哪些东西是构建依赖，哪些东西是运行数据。只要边界清楚，后续迁移和复盘就轻松很多。
