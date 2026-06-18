# Frodo 型矩阵 LWE 正确性分析

Frodo 型方案基于普通 LWE，而不是环 LWE 或模块 LWE。它的核心公钥加密部分可以看作 LPR 型方案的矩阵化版本：秘密、误差和消息都变成小矩阵；解密时主矩阵乘法项抵消，剩下一个误差矩阵。

本案例抽象分析 FrodoPKE 核心结构。实际 FrodoKEM 还包含消息编码、FO 变换、哈希和密钥派生等组件。正确性失败主要来自底层 FrodoPKE 的解密错误。

## 方案对象与参数

### 矩阵维度

取模数 $q$，主维度 $n$，小块维度 $\bar n$。公共矩阵为：

$$
\mathbf{A}\in\mathbb{Z}_q^{n\times n}.
$$

秘密矩阵和误差矩阵为：

$$
\mathbf{S},\mathbf{E}\in\mathbb{Z}^{n\times \bar n}.
$$

加密随机矩阵为：

$$
\mathbf{S}'\in\mathbb{Z}^{n\times \bar n}.
$$

加密误差矩阵为：

$$
\mathbf{E}'\in\mathbb{Z}^{\bar n\times n},
$$

$$
\mathbf{E}''\in\mathbb{Z}^{\bar n\times\bar n}.
$$

注意维度非常重要。后续推导中，每个矩阵乘法都必须维度匹配。

### 消息矩阵与编码

令消息块为：

$$
\mathbf{M}\in\mathcal{M}^{\bar n\times\bar n}.
$$

编码函数输出矩阵：

$$
\mathsf{Encode}(\mathbf{M})\in\mathbb{Z}_q^{\bar n\times\bar n}.
$$

为了抽象分析，假设每个位置的有效编码间隔为 $\Delta$。解码成功的充分条件是每个位置的噪声中心代表元绝对值小于 $\Delta/2$。

## 方案描述

### 密钥生成

采样公共矩阵：

$$
\mathbf{A}\xleftarrow{\$}\mathbb{Z}_q^{n\times n}.
$$

采样秘密和误差：

$$
\mathbf{S}\leftarrow\chi_s^{n\times\bar n},
$$

$$
\mathbf{E}\leftarrow\chi_e^{n\times\bar n}.
$$

计算：

$$
\mathbf{B}:=\mathbf{A}\mathbf{S}+\mathbf{E}\pmod q.
$$

公钥为：

$$
\mathsf{pk}:=(\mathbf{A},\mathbf{B}).
$$

私钥为：

$$
\mathsf{sk}:=\mathbf{S}.
$$

### 加密

加密消息矩阵 $\mathbf{M}$ 时，采样：

$$
\mathbf{S}'\leftarrow\chi_r^{n\times\bar n},
$$

$$
\mathbf{E}'\leftarrow\chi_1^{\bar n\times n},
$$

$$
\mathbf{E}''\leftarrow\chi_2^{\bar n\times\bar n}.
$$

计算第一密文分量：

$$
\mathbf{C}_1:=\mathbf{S}'^{\top}\mathbf{A}+\mathbf{E}'\pmod q.
$$

计算第二密文分量：

$$
\mathbf{C}_2:=\mathbf{S}'^{\top}\mathbf{B}+\mathbf{E}''+\mathsf{Encode}(\mathbf{M})\pmod q.
$$

密文为：

$$
\mathsf{ct}:=(\mathbf{C}_1,\mathbf{C}_2).
$$

### 解密

解密时计算：

$$
\mathbf{W}:=\mathbf{C}_2-\mathbf{C}_1\mathbf{S}\pmod q.
$$

然后对 $\mathbf{W}$ 的每个位置进行解码。

## 正确性推导

### 代入公钥矩阵

由密钥生成：

$$
\mathbf{B}=\mathbf{A}\mathbf{S}+\mathbf{E}.
$$

所以：

$$
\mathbf{S}'^{\top}\mathbf{B}=\mathbf{S}'^{\top}(\mathbf{A}\mathbf{S}+\mathbf{E}).
$$

展开：

$$
\mathbf{S}'^{\top}\mathbf{B}=\mathbf{S}'^{\top}\mathbf{A}\mathbf{S}+\mathbf{S}'^{\top}\mathbf{E}.
$$

### 代入密文矩阵

由加密定义：

$$
\mathbf{C}_2=\mathbf{S}'^{\top}\mathbf{B}+\mathbf{E}''+\mathsf{Encode}(\mathbf{M}).
$$

又有：

$$
\mathbf{C}_1=\mathbf{S}'^{\top}\mathbf{A}+\mathbf{E}'.
$$

因此：

$$
\mathbf{C}_1\mathbf{S}=\mathbf{S}'^{\top}\mathbf{A}\mathbf{S}+\mathbf{E}'\mathbf{S}.
$$

### 消去主矩阵项

解密矩阵为：

$$
\begin{aligned}
\mathbf{W}
&=\mathbf{C}_2-\mathbf{C}_1\mathbf{S} \\
&=\mathbf{S}'^{\top}\mathbf{B}+\mathbf{E}''+\mathsf{Encode}(\mathbf{M})-(\mathbf{S}'^{\top}\mathbf{A}+\mathbf{E}')\mathbf{S} \\
&=\mathbf{S}'^{\top}(\mathbf{A}\mathbf{S}+\mathbf{E})+\mathbf{E}''+\mathsf{Encode}(\mathbf{M})-\mathbf{S}'^{\top}\mathbf{A}\mathbf{S}-\mathbf{E}'\mathbf{S} \\
&=\mathsf{Encode}(\mathbf{M})+\mathbf{S}'^{\top}\mathbf{E}+\mathbf{E}''-\mathbf{E}'\mathbf{S}.
\end{aligned}
$$

主项 $\mathbf{S}'^{\top}\mathbf{A}\mathbf{S}$ 被完全消去。

### 总误差矩阵

定义总解密噪声矩阵：

$$
\mathbf{N}_{\rm dec}:=\mathbf{S}'^{\top}\mathbf{E}+\mathbf{E}''-\mathbf{E}'\mathbf{S}.
$$

于是：

$$
\mathbf{W}=\mathsf{Encode}(\mathbf{M})+\mathbf{N}_{\rm dec}\pmod q.
$$

正确性问题变成：矩阵 $\mathbf{N}_{\rm dec}$ 的每个条目是否都没有越过对应解码边界。

## 条目级误差表达式

### 展开单个条目

设 $a,b\in[\bar n]$。总误差矩阵第 $(a,b)$ 项为：

$$
(\mathbf{N}_{\rm dec})_{a,b}=(\mathbf{S}'^{\top}\mathbf{E})_{a,b}+\mathbf{E}''_{a,b}-(\mathbf{E}'\mathbf{S})_{a,b}.
$$

第一项展开为：

$$
(\mathbf{S}'^{\top}\mathbf{E})_{a,b}=\sum_{i=1}^{n}\mathbf{S}'_{i,a}\mathbf{E}_{i,b}.
$$

第三项展开为：

$$
(\mathbf{E}'\mathbf{S})_{a,b}=\sum_{j=1}^{n}\mathbf{E}'_{a,j}\mathbf{S}_{j,b}.
$$

因此：

$$
(\mathbf{N}_{\rm dec})_{a,b}
=
\sum_{i=1}^{n}\mathbf{S}'_{i,a}\mathbf{E}_{i,b}
+
\mathbf{E}''_{a,b}
-
\sum_{j=1}^{n}\mathbf{E}'_{a,j}\mathbf{S}_{j,b}.
$$

这个公式是 Frodo 型正确性分析的核心。

### 正确性事件

第 $(a,b)$ 个位置解码失败事件为：

$$
E_{a,b}:=\left\{|\langle (\mathbf{N}_{\rm dec})_{a,b}\rangle_q|\geq\frac{\Delta}{2}\right\}.
$$

整体失败事件为：

$$
E_{\rm fail}:=\bigcup_{a=1}^{\bar n}\bigcup_{b=1}^{\bar n}E_{a,b}.
$$

由 union bound：

$$
\Pr[E_{\rm fail}]\leq\sum_{a=1}^{\bar n}\sum_{b=1}^{\bar n}\Pr[E_{a,b}].
$$

若所有条目边缘分布相同，并且单条目失败概率为 $p_{\rm entry}$，则：

$$
\Pr[E_{\rm fail}]\leq\bar n^2p_{\rm entry}.
$$

这里不需要假设 $E_{a,b}$ 独立。

## 范围界分析

### 有界分布假设

假设：

$$
|\mathbf{S}'_{i,a}|\leq B_r,
$$

$$
|\mathbf{E}_{i,b}|\leq B_e,
$$

$$
|\mathbf{E}'_{a,j}|\leq B_1,
$$

$$
|\mathbf{S}_{j,b}|\leq B_s,
$$

$$
|\mathbf{E}''_{a,b}|\leq B_2.
$$

### 界定单条目误差

第一项满足：

$$
\left|\sum_{i=1}^{n}\mathbf{S}'_{i,a}\mathbf{E}_{i,b}\right|
\leq
\sum_{i=1}^{n}|\mathbf{S}'_{i,a}||\mathbf{E}_{i,b}|
\leq
nB_rB_e.
$$

第二项满足：

$$
|\mathbf{E}''_{a,b}|\leq B_2.
$$

第三项满足：

$$
\left|\sum_{j=1}^{n}\mathbf{E}'_{a,j}\mathbf{S}_{j,b}\right|
\leq
\sum_{j=1}^{n}|\mathbf{E}'_{a,j}||\mathbf{S}_{j,b}|
\leq
nB_1B_s.
$$

所以：

$$
|(\mathbf{N}_{\rm dec})_{a,b}|
\leq
nB_rB_e+B_2+nB_1B_s.
$$

### 零失败充分条件

若：

$$
nB_rB_e+B_2+nB_1B_s<\frac{\Delta}{2},
$$

则每个条目都满足正确性条件，从而整体解密不会失败。

这个条件通常非常保守，因为它假设所有乘积项以同一符号叠加。实际 Frodo 型参数会使用更精细的离散噪声分布和数值分析。

## 方差与尾界分析

### 条目方差计算

假设所有参与单条目的随机变量独立、均值为 $0$，并设：

$$
\operatorname{Var}(\mathbf{S}'_{i,a})=\sigma_r^2,
$$

$$
\operatorname{Var}(\mathbf{E}_{i,b})=\sigma_e^2,
$$

$$
\operatorname{Var}(\mathbf{E}'_{a,j})=\sigma_1^2,
$$

$$
\operatorname{Var}(\mathbf{S}_{j,b})=\sigma_s^2,
$$

$$
\operatorname{Var}(\mathbf{E}''_{a,b})=\sigma_2^2.
$$

则：

$$
\operatorname{Var}(\mathbf{S}'_{i,a}\mathbf{E}_{i,b})=\sigma_r^2\sigma_e^2.
$$

并且：

$$
\operatorname{Var}(\mathbf{E}'_{a,j}\mathbf{S}_{j,b})=\sigma_1^2\sigma_s^2.
$$

因此单条目总方差为：

$$
\sigma_N^2:=n\sigma_r^2\sigma_e^2+\sigma_2^2+n\sigma_1^2\sigma_s^2.
$$

### Chebyshev 条目界

由 Chebyshev：

$$
\Pr[E_{a,b}]
\leq
\frac{\sigma_N^2}{(\Delta/2)^2}.
$$

即：

$$
\Pr[E_{a,b}]
\leq
\frac{4\sigma_N^2}{\Delta^2}.
$$

于是整体失败率满足：

$$
\Pr[E_{\rm fail}]
\leq
\bar n^2\cdot\frac{4\sigma_N^2}{\Delta^2}.
$$

这个界通常偏松，但它清楚展示了维度放大因子 $\bar n^2$。

### Bernstein 条目界

令单项最大幅度为：

$$
M:=\max\{B_rB_e,B_1B_s,B_2\}.
$$

令总方差参数为：

$$
\nu^2:=n\sigma_r^2\sigma_e^2+\sigma_2^2+n\sigma_1^2\sigma_s^2.
$$

Bernstein 双侧界给出：

$$
\Pr[|(\mathbf{N}_{\rm dec})_{a,b}|\geq T]
\leq
2\exp\left(-\frac{T^2}{2(\nu^2+MT/3)}\right).
$$

取 $T=\Delta/2$，得到：

$$
\Pr[E_{a,b}]
\leq
2\exp\left(-\frac{\Delta^2}{8\nu^2+(4/3)M\Delta}\right).
$$

再由 union bound：

$$
\Pr[E_{\rm fail}]
\leq
2\bar n^2\exp\left(-\frac{\Delta^2}{8\nu^2+(4/3)M\Delta}\right).
$$

## 精确分布计算思路

### 乘积分布

单条目误差由两组乘积和加一个误差项组成。若各基础噪声分布支持集有限，可以先计算乘积分布。

例如令：

$$
X:=S'E.
$$

若 $S'$ 和 $E$ 独立，则：

$$
P_X(x)=\sum_{uv=x}P_{S'}(u)P_E(v).
$$

类似地，对：

$$
Y:=E'S
$$

可计算 $P_Y$。

### 卷积总分布

单条目误差可写为：

$$
N=X_1+\cdots+X_n+E''-(Y_1+\cdots+Y_n).
$$

若这些项可按独立模型处理，则总分布为：

$$
P_N=P_X^{*n}*P_{E''}*P_{-Y}^{*n}.
$$

其中 $*$ 表示卷积，$P_X^{*n}$ 表示 $n$ 次卷积。

失败概率为：

$$
p_{\rm entry}=\sum_{|z|\geq\Delta/2}P_N(z).
$$

整体失败率再用：

$$
\Pr[E_{\rm fail}]\leq\bar n^2p_{\rm entry}.
$$

## 小结

Frodo 型矩阵 LWE PKE 的解密推导给出：

$$
\mathbf{W}=\mathsf{Encode}(\mathbf{M})+\mathbf{N}_{\rm dec}\pmod q.
$$

其中：

$$
\mathbf{N}_{\rm dec}=\mathbf{S}'^{\top}\mathbf{E}+\mathbf{E}''-\mathbf{E}'\mathbf{S}.
$$

单条目噪声为：

$$
(\mathbf{N}_{\rm dec})_{a,b}
=
\sum_{i=1}^{n}\mathbf{S}'_{i,a}\mathbf{E}_{i,b}
+
\mathbf{E}''_{a,b}
-
\sum_{j=1}^{n}\mathbf{E}'_{a,j}\mathbf{S}_{j,b}.
$$

正确性分析要控制每个条目的中心代表元：

$$
|\langle(\mathbf{N}_{\rm dec})_{a,b}\rangle_q|<\frac{\Delta}{2}.
$$

由于 Frodo 型方案不使用环结构，条目表达式较长但代数关系清晰。其正确性分析的关键在于精确处理离散噪声分布、乘积和以及矩阵条目的 union bound 放大。
