# SIS、LWE、RLWE、MLWE 与 NTRU

## SIS 与 ISIS

固定 $\mathbf{A}\xleftarrow{\$}\mathbb{Z}_q^{n\times m}$。

|符号|含义|
|-|-|
|$\mathsf{SIS}_{n,m,q,\beta}$|求非零短向量 $\mathbf{z}$，使 $\mathbf{A}\mathbf{z}=\mathbf{0}\bmod q$|
|$\mathsf{ISIS}_{n,m,q,\beta}$|给定 $\mathbf{u}$，求短 $\mathbf{z}$ 使 $\mathbf{A}\mathbf{z}=\mathbf{u}\bmod q$|
|$\beta_{\rm SIS}$|短向量范数界|
|$\mathsf{MSIS}$|模块 SIS|

## LWE

统一采用列样本方向：

\[
\mathbf{b}:=\mathbf{A}^{\top}\mathbf{s}+\mathbf{e}\pmod q.
\]

|符号|含义|
|-|-|
|$\mathbf{s}\leftarrow\chi_s^n$|秘密向量|
|$\mathbf{e}\leftarrow\chi_e^m$|误差向量|
|$\mathsf{LWE}_{n,m,q,\chi_s,\chi_e}$|LWE 分布或问题|
|$\mathsf{LWE}^{\rm s}$|Search-LWE|
|$\mathsf{LWE}^{\rm d}$|Decision-LWE|
|$\alpha_{\rm LWE}$|归一化误差率|

## RLWE 与 MLWE

|符号|含义|
|-|-|
|$b:=as+e\in R_q$|RLWE 样本|
|$\mathsf{RLWE}_{R,q,\chi_s,\chi_e}$|Ring-LWE|
|$\mathbf{b}:=\mathbf{A}\mathbf{s}+\mathbf{e}\in R_q^k$|MLWE 样本|
|$\mathsf{MLWE}_{k,\ell,n,q,\chi_s,\chi_e}$|Module-LWE|
|$\mathsf{LWR},\mathsf{RLWR},\mathsf{MLWR}$|相应舍入问题|

## NTRU

|符号|含义|
|-|-|
|$f,g$|短秘密多项式|
|$h_{\rm NTRU}=g f^{-1}\bmod q$|NTRU 公钥|
|$\mathsf{NTRU\text{-}Search}$|从公钥恢复短关系的问题|
|$\mathsf{NTRU\text{-}SIS}$|NTRU 模块上的短关系问题|
