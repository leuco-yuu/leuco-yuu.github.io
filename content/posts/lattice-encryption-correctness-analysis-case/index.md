---
title: 格基加密正确性分析案例
date: 2026-06-16T16:13:54+08:00
lastmod: 2026-06-18T12:35:19+08:00
draft: false
slug: lattice-encryption-correctness-analysis-case
sections:
- 00-格基加密正确性分析总览.md
- 01-Regev型LWE公钥加密正确性分析.md
- 02-LPR型LWE公钥加密正确性分析.md
- 03-Frodo型矩阵LWE正确性分析.md
- 04-环LWE公钥加密正确性分析.md
- 05-MLKEM型模块LWE正确性分析.md
series_order:
- 1
- 7
description: ''
summary: 本文系统分析Regev型、LPR型、Frodo型、环LWE及ML-KEM型模块LWE加密的正确性，通过代入解密公式消去主项得到总解密噪声表达式，定义基于编码间隔的正确性条件，并采用范围界、Chebyshev、Bernstein及精确卷积方法估计失败概率，揭示各类方案噪声结构的共同模板与参数依赖关系。
tags:
- 格基密码学
- 后量子密码
- 概率论
- 正确性分析
- 环LWE
- 模块LWE
- 噪声分析
categories:
- 后量子安全理论
- 学习笔记
series:
- 学习案例
- 基于格的密码学理论
cover: ''
---

