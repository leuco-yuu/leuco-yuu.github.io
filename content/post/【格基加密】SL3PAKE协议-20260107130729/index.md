+++
title= "【格基加密】SL3PAKE协议"
date= "2026-01-07T13:01:39+08:00"
lastmod= "2026-01-07T13:01:39+08:00"
draft=false
author= "leuco"
description= "适用于后量子时代的基于格的简单三方口令认证密钥交换技术（Simple Lattice-based Three-party Password Authenticated Key Exchange for post-quantum world）"
keywords= ["后量子加密算法", "密钥协商", "格基加密", "密码学", "三方", "口令认证", "密钥交换"]
categories= ["网络空间安全", "后量子时代密码学"]
tags= ["后量子加密算法", "密钥协商", "格基加密", "密码学", "三方", "口令认证", "密钥交换"]
math=true
image="cover.png"

+++

# SL3PAKE协议

{{<toc>}}

## 参考文献

- 引用格式（IEEE）：

  [1] V. Dabra, S. Kumari, A. Bala, and S. Yadav, “SL3PAKE: Simple Lattice-based Three-party Password Authenticated Key Exchange for post-quantum world,” *Journal of Information Security and Applications*, vol. 84, p. 103826, Aug. 2024, doi: [10.1016/j.jisa.2024.103826](https://doi.org/10.1016/j.jisa.2024.103826).

- 网址：[原文链接](https://www.sciencedirect.com/science/article/pii/S2214212624001297)

- 附件：[SL3PAKE: Simple Lattice-based Three-party Password Authenticated Key Exchange for post-quantum world](Full-Text.pdf)

## 基础信息

### 困难性基础

​	SL3PAKE的困难性建立在 RLW 问题的计算困难性上，RLWE问题可以归约到格中的最短向量问题（SVP）或最近向量问题（CVP）

### 三方参与

- **Server(S)**：SL3PAKE服务器
- **ClientA(A)**：密钥交换客户端A
- **ClientB(B)**：密钥交换客户端B

### 符号表

- $n$：满足 $n=2k$ 的正整数（安全参数），其中 $k$ 为任意正数

- $q$：满足 $q\equiv 1 \ (mod\ 2n)$ 的大奇素数

- $R$：$\displaystyle{\frac{\mathbb{Z}\left[x\right]}{\langle f(x)\rangle} = \frac{\mathbb{Z}\left[x\right]}{\langle x^n+1 \rangle}}$（多项式商环，其中 $f(x)$ 为分圆多项式）

- $R_q$：$\displaystyle{\frac{R}{qR}}$

- $\chi_\beta$：$R_q$ 上的标准差为 $\beta$ 的离散高斯分布

- $e\leftarrow \chi_\beta$：来自 $\chi_\beta$ 中的随机样本

- $a\in_{ru}R_q$ ：从 $R_q$ 中的随机均匀的选取样本

- $x_S,s_S$：S的公钥与私钥

- $x_A,s_A$：A的公钥与私钥

- $x_B,s_B$：B的公钥与私钥

- $P,Q$：将字符串 $P$ 和 $Q$ 串联拼接合并

- $E$：近0集合，$$\displaystyle{E=\left\{ -\left\lfloor \frac{q}{4} \right\rfloor \cdots \left\lfloor \frac{q}{4} \right\rceil \right\}}$$

- $\operatorname{Cha}(m)$：特征函数，
  $$
  \operatorname{Cha}(m) = 
  \begin{cases}
  0, \quad  if\ m\in E \\
  1,  else\\
  \end{cases}\\
  $$

- $\operatorname{Mod_2}(m,n)$：校验函数，
  $$
  \operatorname{Mod_2}(m,n) = \left(m+n\cdot\frac{q-1}{2} \right)\ \ mod\ q \ \ mod\ 2
  $$

- $\operatorname{h_0}(\cdot)$：哈希函数，$\operatorname{h_0}: \lbrace 0,1 \rbrace ^* \rightarrow R_q$
- $\operatorname{h_1}(\cdot)$：哈希函数，$\operatorname{h_0}: \lbrace 0,1 \rbrace^* \rightarrow \lbrace 0,1\rbrace^g$
- $\operatorname{h_2}(\cdot)$：哈希函数，$\operatorname{h_0}:\lbrace 0,1\rbrace^* \rightarrow \chi_\beta$
- $pw_A,pw_B$：A和B的口令密码
- $ID_S,ID_A,ID_B$：S、A和B的唯一身份标识

## 协议流程

### Server S：初始化设置

- S选择安全参数 $n$ 与奇素数 $q$
- S选择离散高斯分布 $\chi_\beta$ 以及 $a\in_{ru}R_q$
- S计算服务器公钥 $x_S = a\cdot s_S + 2e_S$，其中  $ s_S,e_S\leftarrow \chi_\beta$
- S选取哈希函数 $\operatorname{h_0}(\cdot)$、$\operatorname{h_1}(\cdot)$、$\operatorname{h_2}(\cdot)$
- S通过安全信道接受A和B的口令的哈希值 $\operatorname{h_0}(pw_A)$ 和 $\operatorname{h_0}(pw_B)$
- S生成并公开S、A、B的身份标识 $ID_S$、$ID_A$、$ID_B$
- S保存参数：$\lbrace s_S,\operatorname{h_0}(pw_A),\operatorname{h_0}(pw_B) \rbrace$
- S公开参数：$\lbrace n,q,\chi_\beta,a,x_S,ID_A,ID_B,ID_S,\operatorname{h_0}(\cdot),\operatorname{h_1}(\cdot),\operatorname{h_2}(\cdot) \rbrace$

### Client A：实例化会话

- **Input**：$ID_A$ ， $pw_A$
- $x_A = a\cdot s_A + 2e_A$，其中 $s_A,e_A\leftarrow \chi_\beta$
- $x_A^* = x_A + \operatorname{h_0}(pw_A)$
- $h_{AS} = \operatorname{h_1}(ID_A,ID_S,x_A,x^*_A)$

### Client A => Client B

- $ A \Rightarrow B: \lbrace ID_A,x_A^\*,h_{AS} \rbrace $

### Client B：实例化会话

- **Input**：$ID_B$ ， $pw_B$
- $x_B = a\cdot s_B + 2e_B$，其中 $s_B,e_B\leftarrow \chi_\beta$
- $x^*_B = x_B + \operatorname{h_0}(pw_B)$
- $h_{BS} = \operatorname{h_1}(ID_B,ID_S,x_B,x^\*_B)$

### Client B => Server S

- $ B \Rightarrow S: \lbrace ID_A,ID_B,x_A^\*,x_B^\*,h_{AS},h_{BS} \rbrace $

### Server S：客户端身份认证

#### 对 Client A 的身份认证

- $x_A\' = x^*_A - \operatorname{h_0}(pw_A)$，这里有 $x_A == x_A\'$
- **Check if** $h_{AS} \overset{?}{==} \operatorname{h_1}(ID_A,ID_S,x_A\',x^\*_A)$
- **If not, then abort.**

#### 对 Client B 的身份认证

- $x_B\' = x^*_B - \operatorname{h_0}(pw_B)$，这里有 $x_B == x_B\'$
- **Check if** $h_{BS} \overset{?}{==} \operatorname{h_1}(ID_B,ID_S,x_B\',x^*_B)$
- **If not, then abort.**

### Server S：协助 Client A 和 Client B 建立协商密钥

- $x_S = a\cdot s_S +2e_S $，其中 $ s_S,e_S\leftarrow \chi_\beta$

#### 对于 Client A ：

- $c_A = x_B\' \cdot s_S + 2f_{S_4}$，其中 $f_{S_4} \leftarrow \chi_\beta$
- $m = \operatorname{h_2}(ID_S,ID_A,x_S,x_A\')$
- $k_{SA}=(x_A\'\cdot s_S+2m)\cdot m+2f_{S_1}$，其中 $f_{S_1} \leftarrow \chi_\beta$
- $\omega_{SA} = \operatorname{Cha}(k_{SA})$
- $\sigma_{SA} = \operatorname{Mod_2}(k_{SA},\omega_{SA})$
- $\alpha_{SA}=\operatorname{h_1}(ID_A,ID_B,ID_S,c_A,x_A\',\sigma_{SA})$

#### 对于 Client B：

- $c_B = x_A\'\cdot s_S + 2f_{S_5}$，其中 $f_{S_5} \leftarrow \chi_\beta$
- $n = \operatorname{h_2}(ID_S,ID_B,x_S,x_B\')$
- $k_{SB}=(x_B\'\cdot s_S+2n)\cdot n+2f_{S_3}$，其中 $f_{S_3} \leftarrow \chi_\beta$
- $\omega_{SB} = \operatorname{Cha}(k_{SB})$
- $\sigma_{SB} = \operatorname{Mod_2}(k_{SB},\omega_{SB})$
- $\alpha_{SB}=\operatorname{h_1}(ID_A,ID_B,ID_S,c_B,x_B\',\sigma_{SB})$

### Server S => Client B

- $S\Rightarrow B: \lbrace c_A,c_B,x_S,\omega_{SA},\omega_{SB},\alpha_{SA},\alpha_{SB}\rbrace$

### Client B：验证 S 身份并与 A 协商密钥

#### 验证 Server S 的身份

- $n = \operatorname{h_2}(ID_S,ID_B,x_S,x_B)$
- $k_{BS}=(x_S\cdot s_B+2n)\cdot n+2f_{B_1}$，其中 $f_{B_1} \leftarrow \chi_\beta$
- $\sigma_{BS} = \operatorname{Mod_2}(k_{BS},\omega_{SB})$
- $\alpha_{BS}=\operatorname{h_1}(ID_A,ID_B,ID_S,c_B,x_B,\sigma_{BS})$
- **Check if** $\alpha_{BS} \overset{?}{==} \alpha_{SB}$
- **If not, then abort.**

#### 与 Client A 完成协商

- $\nu_{BA} = c_B\cdot s_B +2f_{B_2}$，其中 $f_{B_2} \leftarrow \chi_\beta$
- $\omega_{BA}=\operatorname{Cha}(\nu_{BA})$
- $\sigma_{BA}=\operatorname{Mod_2}(\nu_{BA},\omega_{BA})$
- $ sk_{BA} = \operatorname{h_1}(ID_A,ID_B,ID_S,x_A^\*,x_B^\*,\sigma_{BA})$

### Client B => Client A

- $B\Rightarrow A: \lbrace ID_B,x_B^\*,c_A,x_S,\omega_{BA},\omega_{SA},\alpha_{SA}\rbrace$

### Client A：验证 S 身份并与 B 协商密钥

#### 验证 Server S 的身份

- $m = \operatorname{h_2}(ID_S,ID_A,x_S,x_A)$
- $k_{AS}=(x_S\cdot s_A+2m)\cdot m+2f_{A_1}$，其中 $f_{A_1} \leftarrow \chi_\beta$
- $\sigma_{AS} = \operatorname{Mod_2}(k_{AS},\omega_{SA})$
- $\alpha_{AS}=\operatorname{h_1}(ID_A,ID_B,ID_S,c_A,x_A,\sigma_{AS})$
- **Check if** $\alpha_{AS} \overset{?}{==} \alpha_{SA}$
- **If not, then abort.**

#### 与 Client B 完成协商

- $\nu_{AB} = c_A\cdot s_A +2f_{A_2}$，其中 $f_{A_2} \leftarrow \chi_\beta$
- $\sigma_{AB}=\operatorname{Mod_2}(\nu_{AB},\omega_{BA})$
- $sk_{AB} = \operatorname{h_1}(ID_A,ID_B,ID_S,x_A^\*,x_B^\*,\sigma_{AB})$

### 完成密钥协商

- 协商密钥：$sk_{BA} == sk_{AB}$
