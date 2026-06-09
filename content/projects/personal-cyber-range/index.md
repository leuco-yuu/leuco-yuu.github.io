---
title: Personal Cyber Range
date: 2026-05-30T14:00:00+08:00
lastmod: 2026-06-09T17:41:54+08:00
draft: false
slug: personal-cyber-range
homepage: true
description: 用于授权安全实验的个人靶场环境。
summary: 基于虚拟机和容器组合的安全实验环境，用于复现漏洞、记录攻击链和验证防护思路。
featured: true
tags:
- Cyber Range
- Docker
- Nmap
- Lab
categories: []
cover: cover.jpg
link: null
status: in_progress
---

## Project Overview

Personal Cyber Range 是一个面向授权实验的本地靶场项目。目标不是堆更多漏洞环境，而是把网络拓扑、服务版本、扫描记录和复盘文档放进同一套可管理流程。

## Features

- 使用独立虚拟网络隔离实验流量
- 用 Docker Compose 管理轻量服务
- 每个实验保留命令、截图和日志路径
- 为扫描、验证和清理步骤提供脚本

## Technologies Used

- VMware / VirtualBox
- Docker Compose
- Nmap
- Linux networking
- Hugo 文档归档

## Notes

这个项目会和“网络安全实践”系列文章同步推进。每次新增实验前先写授权范围和目标，结束后补复盘摘要。
