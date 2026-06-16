# 环 LWE 公钥加密正确性分析

环 LWE 公钥加密可以看作 LPR 型 LWE 公钥加密在多项式环上的版本。它把向量内积换成环乘法，把普通矩阵运算换成卷积运算。这样可以显著压缩公钥和密文规模，但也引入一个新的正确性分析问题：噪声不再只是普通坐标乘积和，而是环卷积系数的乘积和。

本文分析一个简化环 LWE 公钥加密方案。目标不是完整复现某个标准方案，而是展示环结构下正确性证明的基本模板：先写出方案，再代入解密公式，最后得到总误差表达式，并把每个系数的越界概率转化为失败概率。

### 噪声分布

设 $\chi$ 是一个小噪声分布，例如中心二项分布、小区间均匀分布或离散 Gaussian 的截断版本。若写作 $e\leftarrow\chi^R$，表示 $e\in R$ 的每个系数从 $\chi$ 独立采样。

为了做正确性分析，记各噪声系数满足：

$$
|e_i|\leq B_e,
$$

$$
|s_i|\leq B_s,
$$

$$
|r_i|\leq B_r,
$$

$$
|(e_1)_i|\leq B_1,
$$

$$
|(e_2)_i|\leq B_2.
$$

这里 $B_e,B_s,B_r,B_1,B_2$ 是系数级别的幅度上界。它们不是概率，而是噪声支持集或高概率截断区间的边界。

## 方案描述

### 密钥生成

密钥生成算法采样：

$$
a\xleftarrow{\$}R_q,
$$

$$
s\leftarrow\chi_s^R,
$$

$$
e\leftarrow\chi_e^R.
$$

然后计算：

$$
t:=a\star s+e\pmod q.
$$

公钥和私钥分别为：

$$
\mathsf{pk}:=(a,t),
$$

$$
\mathsf{sk}:=s.
$$

其中 $a$ 是均匀环元素，$s$ 是短秘密，$e$ 是密钥生成误差。公钥 $t$ 不是均匀采样得到的独立环元素，而是由 $a,s,e$ 确定出来的带噪环 LWE 样本。

### 加密

为了加密消息 $\mu$，先把消息编码为环元素 $\mathsf{Encode}(\mu)\in R_q$。然后采样：

$$
r\leftarrow\chi_r^R,
$$

$$
e_1\leftarrow\chi_1^R,
$$

$$
e_2\leftarrow\chi_2^R.
$$

计算密文：

$$
u:=a\star r+e_1\pmod q,
$$

$$
v:=t\star r+e_2+\mathsf{Encode}(\mu)\pmod q.
$$

密文为：

$$
\mathsf{ct}:=(u,v).
$$

### 解密

解密时计算：

$$
w:=v-u\star s\pmod q.
$$

然后对 $w$ 的每个系数做中心提升和判决，恢复消息系数。

若消息比特采用二元编码，通常可以抽象为把 $0$ 映射到 $0$，把 $1$ 映射到接近 $q/2$ 的位置。此时判决边界约为 $q/4$。更一般地，若相邻编码点之间的距离为 $\Delta$，则每个系数的允许噪声幅度为 $\Delta/2$。

## 解密表达式推导

正确性分析的第一步，是把 $w$ 完整展开。

由加密公式：

$$
w=v-u\star s.
$$

代入 $v$：

$$
w=t\star r+e_2+\mathsf{Encode}(\mu)-u\star s.
$$

再代入 $u$：

$$
w=t\star r+e_2+\mathsf{Encode}(\mu)-(a\star r+e_1)\star s.
$$

展开右侧：

$$
w=t\star r+e_2+\mathsf{Encode}(\mu)-(a\star r)\star s-e_1\star s.
$$

再代入公钥项 $t=a\star s+e$：

$$
w=(a\star s+e)\star r+e_2+\mathsf{Encode}(\mu)-(a\star r)\star s-e_1\star s.
$$

继续展开：

$$
w=(a\star s)\star r+e\star r+e_2+\mathsf{Encode}(\mu)-(a\star r)\star s-e_1\star s.
$$

由于 $R_q=\mathbb{Z}_q[X]/(X^n+1)$ 是交换环，环乘法满足结合律与交换律，所以：

$$
(a\star s)\star r=(a\star r)\star s.
$$

因此主项抵消，得到：

$$
w=\mathsf{Encode}(\mu)+e\star r+e_2-e_1\star s.
$$

于是定义总解密噪声：

$$
N_{\rm dec}:=e\star r+e_2-e_1\star s.
$$

解密值可以写成：

$$
w=\mathsf{Encode}(\mu)+N_{\rm dec}.
$$

这就是环 LWE 正确性分析的核心等式。

## 系数级噪声展开

环元素 $N_{\rm dec}$ 本身仍然是一个多项式。正确性判决发生在系数层面，因此还要分析每个系数的噪声。

记：

$$
N_{\rm dec}=\sum_{k=0}^{n-1}N_kX^k.
$$

则第 $k$ 个系数满足：

$$
N_k=(e\star r)_k+(e_2)_k-(e_1\star s)_k.
$$

对于负循环卷积，可以把 $(e\star r)_k$ 写成 $n$ 个乘积项之和：

$$
(e\star r)_k=\sum_{i=0}^{n-1}\eta_{i,k}e_i r_{j(i,k)}.
$$

其中 $j(i,k)$ 是由同余关系确定的索引，$\eta_{i,k}\in\{+1,-1\}$ 是由 $X^n=-1$ 产生的符号。具体索引形式不影响正确性分析的主线；关键点是每个系数包含 $n$ 个乘积项。

类似地：

$$
(e_1\star s)_k=\sum_{i=0}^{n-1}\theta_{i,k}(e_1)_i s_{j'(i,k)},
$$

其中 $\theta_{i,k}\in\{+1,-1\}$。

因此，第 $k$ 个总噪声系数可以写成：

$$
N_k=\sum_{i=0}^{n-1}\eta_{i,k}e_i r_{j(i,k)}+(e_2)_k-\sum_{i=0}^{n-1}\theta_{i,k}(e_1)_i s_{j'(i,k)}.
$$

这个表达式说明：环 LWE 的单系数噪声不是单个误差，而是两组卷积乘积和再加一个直接误差项。

## 正确性条件

### 单系数正确条件

设消息编码间隔为 $\Delta$。若第 $k$ 个系数满足：

$$
|\langle N_k\rangle_q|<\frac{\Delta}{2},
$$

则该系数不会被噪声推过判决边界。

这里 $\langle x\rangle_q$ 表示模 $q$ 元素的中心代表元。例如，可以把 $\mathbb{Z}_q$ 中的值解释到区间 $(-q/2,q/2]$ 内。

### 整体正确条件

一个环元素通常包含 $n$ 个消息系数。整体解密成功要求所有系数都正确：

$$
E_{\rm fail}:=\bigcup_{k=0}^{n-1}\left\{|\langle N_k\rangle_q|\geq\frac{\Delta}{2}\right\}.
$$

因此：

$$
\Pr[E_{\rm fail}]\leq\sum_{k=0}^{n-1}\Pr\left[|\langle N_k\rangle_q|\geq\frac{\Delta}{2}\right].
$$

若每个系数的边缘失败概率都不超过 $p_{\rm coeff}$，则：

$$
\Pr[E_{\rm fail}]\leq n p_{\rm coeff}.
$$

这个并合界不要求不同系数的失败事件独立。即使不同 $N_k$ 之间因为卷积共享变量而相关，上式仍然成立。

## 最坏情况幅度分析

最直接的正确性证明是给出确定性幅度界。假设所有噪声系数满足前面给出的上界。

对卷积系数 $(e\star r)_k$，有：

$$
|(e\star r)_k|\leq\sum_{i=0}^{n-1}|e_i||r_{j(i,k)}|.
$$

由于 $|e_i|\leq B_e$ 且 $|r_j|\leq B_r$，得到：

$$
|(e\star r)_k|\leq nB_eB_r.
$$

同理：

$$
|(e_1\star s)_k|\leq nB_1B_s.
$$

又因为：

$$
|(e_2)_k|\leq B_2,
$$

所以：

$$
|N_k|\leq nB_eB_r+B_2+nB_1B_s.
$$

因此，一个充分的确定性正确性条件是：

$$
nB_eB_r+B_2+nB_1B_s<\frac{\Delta}{2}.
$$

若该条件成立，则无论噪声怎样取值，只要落在给定支持集内，就不会解密失败。

这个条件非常清楚，但通常偏保守。实际方案中，所有乘积项同时取到最坏方向的概率很低，因此还需要概率型分析。

## 方差型分析

为了得到更贴近实际的失败概率，可以计算单个系数的方差。

假设以下条件成立：

- $e_i,r_i,(e_1)_i,s_i,(e_2)_i$ 都以 $0$ 为中心；
- 参与同一个系数计算的乘积项可近似视为独立，或通过更严格的条件化方法处理；
- 各分布具有有限方差。

记：

$$
\operatorname{Var}(e_i)=\sigma_e^2,
$$

$$
\operatorname{Var}(r_i)=\sigma_r^2,
$$

$$
\operatorname{Var}((e_1)_i)=\sigma_1^2,
$$

$$
\operatorname{Var}(s_i)=\sigma_s^2,
$$

$$
\operatorname{Var}((e_2)_i)=\sigma_2^2.
$$

若 $e_i$ 与 $r_j$ 独立且均值为 $0$，则：

$$
\operatorname{Var}(e_i r_j)=\operatorname{Var}(e_i)\operatorname{Var}(r_j)=\sigma_e^2\sigma_r^2.
$$

于是可以得到单系数噪声方差的典型近似：

$$
\operatorname{Var}(N_k)\approx n\sigma_e^2\sigma_r^2+\sigma_2^2+n\sigma_1^2\sigma_s^2.
$$

这个式子不能无条件当作严格等式。原因是环卷积的不同系数之间共享随机变量，而且同一个系数内部的某些项也可能存在结构相关。严格证明时需要进一步检查独立性，或者改用条件化、上界分布、精确卷积或有证数值计算。

## Bernstein 型尾界示例

若把 $N_k$ 分解为一组中心化、有界、近似独立或已严格处理为独立的随机项之和，则可以使用 Bernstein 型尾界。

把第 $k$ 个噪声系数抽象写为：

$$
N_k=\sum_i X_i.
$$

假设：

$$
\mathbb{E}[X_i]=0,
$$

$$
|X_i|\leq M,
$$

$$
\nu^2:=\sum_i\operatorname{Var}(X_i).
$$

则 Bernstein 型界给出：

$$
\Pr[N_k\geq t]\leq\exp\left(-\frac{t^2}{2(\nu^2+Mt/3)}\right).
$$

对双侧尾部，可以写成：

$$
\Pr[|N_k|\geq t]\leq 2\exp\left(-\frac{t^2}{2(\nu^2+Mt/3)}\right).
$$

在环 LWE 正确性分析中，可以取：

$$
t:=\frac{\Delta}{2}.
$$

若每个乘积项的幅度不超过 $B_eB_r$ 或 $B_1B_s$，直接误差项幅度不超过 $B_2$，则可以取：

$$
M:=\max\{B_eB_r,B_1B_s,B_2\}.
$$

方差参数可取为保守上界：

$$
\nu^2:=n\sigma_e^2\sigma_r^2+\sigma_2^2+n\sigma_1^2\sigma_s^2.
$$

于是单系数失败概率可上界为：

$$
p_{\rm coeff}\leq 2\exp\left(-\frac{(\Delta/2)^2}{2(\nu^2+M\Delta/6)}\right).
$$

整体失败概率再由并合界得到：

$$
\Pr[E_{\rm fail}]\leq 2n\exp\left(-\frac{(\Delta/2)^2}{2(\nu^2+M\Delta/6)}\right).
$$

该公式展示了正确性分析中的几个关键参数：环次数 $n$ 放大失败概率，噪声方差控制典型波动，最大项幅度 $M$ 控制极端尾部，编码间隔 $\Delta$ 提供容错空间。

## 精确卷积视角

若噪声分布支持集很小，另一种分析方法是精确计算单系数噪声分布。

先计算乘积分布。若 $E$ 与 $R$ 独立，且 $X:=ER$，则：

$$
P_X(x)=\sum_{er=x}P_E(e)P_R(r).
$$

然后对 $n$ 个乘积项做卷积，得到 $(e\star r)_k$ 的分布近似或上界：

$$
P_{e\star r}\approx P_X^{*n}.
$$

同理计算 $(e_1\star s)_k$ 的分布，再与 $e_2$ 的分布卷积。最终得到 $N_k$ 的分布：

$$
P_{N_k}\approx P_{e\star r}*P_{e_2}*P_{-(e_1\star s)}.
$$

之后计算尾部：

$$
p_{\rm coeff}=\sum_{|z|\geq\Delta/2}P_{N_k}(z).
$$

精确卷积比通用尾界更紧，但必须处理两个问题：

- 负循环卷积中的符号不会改变对称分布的乘积分布，但会影响非对称分布；
- 不同系数共享输入变量，不能把所有系数失败事件当作独立事件。

因此，常见做法是精确估计单系数尾部，再使用并合界处理整体失败概率。

## 小结

环 LWE 正确性分析的关键公式是：

$$
N_{\rm dec}:=e\star r+e_2-e_1\star s.
$$

和普通 LWE 不同，$e\star r$ 与 $e_1\star s$ 的每个系数都是卷积乘积和。正确性证明应当沿着以下顺序展开：

- 先把解密值写成 $\mathsf{Encode}(\mu)+N_{\rm dec}$；
- 再把 $N_{\rm dec}$ 展开到系数层面；
- 然后根据编码间隔 $\Delta$ 定义失败事件；
- 最后选择确定性幅度界、Bernstein 型尾界或精确卷积计算失败概率。

环结构提高了效率，但也要求正确性分析明确处理卷积相关性。不能简单把环元素当作普通独立向量逐坐标相乘。
