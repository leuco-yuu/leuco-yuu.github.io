# SIS、ISIS、LWE、RLWE、MLWE、LWR 与 NTRU 假设

## SIS 与 ISIS

固定 $\mathbf A\xleftarrow{\$}\mathbb Z_q^{n\times m}$。

|符号|含义|
|-|-|
|$\mathsf{SIS}_{n,m,q,\beta}$|求非零 $\mathbf z\in\mathbb Z^m$，使 $\mathbf A\mathbf z=0\bmod q$ 且 $\|\mathbf z\|\le\beta$|
|$\beta_{\rm SIS}$|SIS 范数界|
|$\mathsf{ISIS}_{n,m,q,\beta}$|给定 $\mathbf u$ 求短 $\mathbf z$，使 $\mathbf A\mathbf z=\mathbf u\bmod q$|
|$\mathsf{MSIS}_{k,\ell,n,q,\beta}$|模块 SIS|
|$\mathbf z$|SIS/ISIS 解；不得用作 FO 隐式拒绝秘密|
|$h_{\mathbf A}(\mathbf x)$|Ajtai 哈希 $\mathbf A\mathbf x\bmod q$|

## LWE：统一列样本方向

固定 $\mathbf A\xleftarrow{\$}\mathbb Z_q^{n\times m}$、$\mathbf s\leftarrow\chi_s^n$、$\mathbf e\leftarrow\chi_e^m$：

\[
\mathbf b:=\mathbf A^{\top}\mathbf s+\mathbf e\pmod q.
\]

|符号|含义|
|-|-|
|$\mathsf{LWE}_{n,m,q,\chi_s,\chi_e}$|上述样本分布|
|$\mathbf s$|LWE 秘密向量|
|$\mathbf e$|LWE 误差向量|
|$\mathbf b$|带噪样本右端向量|
|$\chi_s$|秘密分布|
|$\chi_e$|误差分布|
|$\sigma_e$|误差标准差|
|$s_e$|$\rho_s$ 规范下的误差参数|
|$\alpha_{\rm sd}:=\sigma_e/q$|标准差归一化误差率|
|$\alpha_{\rho}:=s_e/q$|$\rho_s$ 规范下的归一化误差率|
|$\mathsf{LWE}^{\rm s}$|Search-LWE|
|$\mathsf{LWE}^{\rm d}$|Decision-LWE|
|$\mathsf U_{n,m,q}$|对应均匀分布 $(\mathbf A,\mathbf u)$|

> 引用使用 $\mathbf A\mathbf s+\mathbf e$ 方向的论文时，必须在章节开头注明矩阵已转置，禁止在同一证明中混用两种方向。

## Ring-LWE

|符号|含义|
|-|-|
|$a\xleftarrow{\$}R_q$|均匀环元素|
|$s\leftarrow\chi_s$|环秘密|
|$e\leftarrow\chi_e$|环误差|
|$b:=as+e\in R_q$|RLWE 样本|
|$\mathsf{RLWE}_{R,q,\chi_s,\chi_e}$|Ring-LWE 分布/问题|
|$\mathsf{PLWE}$|系数嵌入定义的 Polynomial-LWE|
|$\|e\|_{\rm coeff}$|系数范数|
|$\|e\|_{\rm can}$|典范嵌入范数|

## Module-LWE

固定 $\mathbf A\xleftarrow{\$}R_q^{k\times\ell}$、$\mathbf s\leftarrow\chi_s^\ell$、$\mathbf e\leftarrow\chi_e^k$：

\[
\mathbf b:=\mathbf A\mathbf s+\mathbf e\in R_q^k.
\]

|符号|含义|
|-|-|
|$\mathsf{MLWE}_{k,\ell,n,q,\chi_s,\chi_e}$|Module-LWE|
|$k$|方程/输出模块秩|
|$\ell$|秘密模块秩|
|$\mathbf A$|模块矩阵|
|$\mathbf s$|模块秘密|
|$\mathbf e$|模块误差|
|$\mathbf b$|模块公开样本|

## LWR、RLWR 与 MLWR

|符号|含义|
|-|-|
|$p<q$|舍入后的较小模数|
|$\lfloor\cdot\rceil_p$|从模 $q$ 缩放到模 $p$ 的规范舍入|
|$\mathbf b:=\left\lfloor\frac pq\mathbf A^{\top}\mathbf s\right\rceil\bmod p$|LWR 样本|
|$\mathsf{LWR}_{n,m,q,p,\chi_s}$|LWR 问题|
|$\mathsf{RLWR}$|Ring-LWR|
|$\mathsf{MLWR}$|Module-LWR|
|$\mathbf e_{\rm rnd}$|等效舍入误差|

## NTRU 假设

|符号|含义|
|-|-|
|$h_{\rm NTRU}=g f^{-1}\bmod q$|NTRU 公钥|
|$(f,g)\leftarrow\chi_f\times\chi_g$|短秘密对|
|$\mathsf{NTRU\text{-}Search}$|从 $h_{\rm NTRU}$ 恢复短关系|
|$\mathsf{NTRU\text{-}SIS}$|在 NTRU 模块中寻找短关系|
|$\mathsf{ModNTRU}$|Module-NTRU|

## 假设标签

|写法|用途|
|-|-|
|$\operatorname{Adv}_{\mathcal B}^{\mathsf{LWE}}$|归约算法对 LWE 的优势|
|$\operatorname{Succ}_{\mathcal B}^{\mathsf{SIS}}$|求解 SIS 的成功概率|
|$\mathsf{Hard}_{P}(\lambda)$|问题 $P$ 在给定参数下的困难性断言|
|$\mathsf{Inst}_P$|问题实例分布|

---

