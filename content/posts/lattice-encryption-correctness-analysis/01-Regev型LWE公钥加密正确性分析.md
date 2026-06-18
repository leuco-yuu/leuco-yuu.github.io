# Regev 型 LWE 公钥加密正确性分析

Regev 型 LWE 公钥加密是最适合入门正确性分析的案例。它的核心结构很简单：公钥给出一组带噪线性方程，加密时随机选取若干方程相加，解密时用私钥消去主线性项，剩下的就是误差和。

本案例只分析一比特消息加密。为了突出正确性推导，本文采用简化版本，不讨论安全归约、密钥压缩和高效实现。

## 方案对象与参数

### 基本空间

取模数 $q$，秘密维数 $n$，样本数 $m$。所有线性运算默认在 $\mathbb{Z}_q$ 中进行。

秘密向量为：

$$
\mathbf{s}\in\mathbb{Z}_q^n.
$$

公共矩阵为：

$$
\mathbf{A}\in\mathbb{Z}_q^{m\times n}.
$$

误差向量为：

$$
\mathbf{e}\in\mathbb{Z}^m.
$$

这里 $\mathbf{e}$ 通常先作为小整数向量采样，再嵌入到 $\mathbb{Z}_q^m$ 中。

### 消息编码

消息为一个比特：

$$
\mu\in\{0,1\}.
$$

采用二元间隔编码：

$$
\mathsf{Encode}(0):=0,
$$

$$
\mathsf{Encode}(1):=\left\lfloor\frac{q}{2}\right\rceil.
$$

为了简化分析，记编码间隔为：

$$
\Delta:=\left\lfloor\frac{q}{2}\right\rceil.
$$

解码时根据结果更接近 $0$ 还是更接近 $\Delta$ 来恢复比特。若噪声中心代表元的绝对值小于约 $q/4$，解码不会跨过判决边界。

## 方案描述

### 密钥生成

采样公共矩阵：

$$
\mathbf{A}\xleftarrow{\$}\mathbb{Z}_q^{m\times n}.
$$

采样秘密：

$$
\mathbf{s}\xleftarrow{\$}\mathbb{Z}_q^n.
$$

采样小误差：

$$
\mathbf{e}\leftarrow\chi_e^m.
$$

计算：

$$
\mathbf{b}:=\mathbf{A}\mathbf{s}+\mathbf{e}\pmod q.
$$

公钥为：

$$
\mathsf{pk}:=(\mathbf{A},\mathbf{b}).
$$

私钥为：

$$
\mathsf{sk}:=\mathbf{s}.
$$

### 加密

加密消息 $\mu\in\{0,1\}$ 时，采样选择向量：

$$
\mathbf{r}\leftarrow\{0,1\}^m.
$$

这里 $\mathbf{r}$ 表示选取哪些 LWE 样本参与求和。计算密文：

$$
\mathbf{c}_1:=\mathbf{A}^{\top}\mathbf{r}\pmod q,
$$

$$
c_2:=\mathbf{b}^{\top}\mathbf{r}+\mathsf{Encode}(\mu)\pmod q.
$$

密文为：

$$
\mathsf{ct}:=(\mathbf{c}_1,c_2).
$$

### 解密

解密时计算：

$$
w:=c_2-\mathbf{s}^{\top}\mathbf{c}_1\pmod q.
$$

然后根据 $w$ 更接近 $0$ 还是更接近 $\Delta$ 来输出 $0$ 或 $1$。

## 正确性推导

### 代入密文定义

由加密定义：

$$
c_2=\mathbf{b}^{\top}\mathbf{r}+\mathsf{Encode}(\mu).
$$

由公钥定义：

$$
\mathbf{b}=\mathbf{A}\mathbf{s}+\mathbf{e}.
$$

因此：

$$
\mathbf{b}^{\top}\mathbf{r}=(\mathbf{A}\mathbf{s}+\mathbf{e})^{\top}\mathbf{r}.
$$

展开得到：

$$
\mathbf{b}^{\top}\mathbf{r}=\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}^{\top}\mathbf{r}.
$$

同时：

$$
\mathbf{c}_1=\mathbf{A}^{\top}\mathbf{r}.
$$

所以：

$$
\mathbf{s}^{\top}\mathbf{c}_1=\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}.
$$

### 消去主项

解密值为：

$$
\begin{aligned}
w
&=c_2-\mathbf{s}^{\top}\mathbf{c}_1 \\
&=\mathbf{b}^{\top}\mathbf{r}+\mathsf{Encode}(\mu)-\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r} \\
&=\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}^{\top}\mathbf{r}+\mathsf{Encode}(\mu)-\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r} \\
&=\mathsf{Encode}(\mu)+\mathbf{e}^{\top}\mathbf{r}.
\end{aligned}
$$

主线性项 $\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}$ 被完全消去，剩下的只有消息编码和误差和。

### 总解密噪声

定义总解密噪声：

$$
N_{\rm dec}:=\mathbf{e}^{\top}\mathbf{r}.
$$

于是解密值满足：

$$
w=\mathsf{Encode}(\mu)+N_{\rm dec}\pmod q.
$$

正确性问题就变成：$N_{\rm dec}$ 是否足够小。

## 正确性条件

### 中心代表元

由于所有运算最终在模 $q$ 下进行，比较噪声大小时不能直接使用标准代表元 $[x]_q\in\{0,\ldots,q-1\}$，而应使用中心代表元 $\langle x\rangle_q$。

例如 $q=17$ 时，$15$ 的标准代表元是 $15$，但中心代表元是 $-2$。在正确性分析中，$15$ 实际离 $0$ 很近。

### 单比特判决边界

如果 $0$ 编码为 $0$，$1$ 编码为约 $q/2$，则两个编码点之间的距离约为 $q/2$。判决边界在中点，因此允许噪声大小约为 $q/4$。

**充分**正确条件为：
$$
|\langle N_{\rm dec}\rangle_q|<\frac{q}{4}.
$$

若这个条件成立，$w$ 会被解码为正确的 $\mu$。

失败事件为：

$$
E_{\rm fail}\subseteq E_{\rm noise}:=\left\{|\langle \mathbf{e}^{\top}\mathbf{r}\rangle_q|\geq\frac{q}{4}\right\}.
$$

## 范围界分析

### 有界误差假设

假设误差每个坐标都有界：

$$
|e_i|\leq B_e.
$$

又因为 $r_i\in\{0,1\}$，所以：

$$
|e_ir_i|\leq B_e r_i.
$$

于是：

$$
|N_{\rm dec}|=\left|\sum_{i=1}^m e_ir_i\right|
\leq
\sum_{i=1}^m |e_i|r_i.
$$

若记 $\mathbf{r}$ 的 Hamming weight 为：

$$
h:=\|\mathbf{r}\|_0=\sum_{i=1}^m r_i,
$$

则：

$$
|N_{\rm dec}|\leq hB_e.
$$

### 零失败充分条件

如果能保证：

$$
hB_e<\frac{q}{4},
$$

那么必然有：

$$
|N_{\rm dec}|<\frac{q}{4}.
$$

因此解密不会失败。

这个条件非常直观：选中的样本越多，误差累积越大；误差最大幅度越大，越容易越界；模数 $q$ 越大，容忍噪声的空间越大。

### 范围界的局限

范围界只看最坏情况。它假设所有被选中的误差都朝同一个方向叠加，因此通常很保守。实际中，误差可能正负抵消，所以失败概率远小于最坏情况界暗示的值。

因此，实际正确性分析通常还需要概率界或精确尾部计算。

## 概率界分析

### 条件在固定选择向量上

固定 $\mathbf{r}$ 后，噪声为：

$$
N_{\rm dec}\mid\mathbf{r}=\sum_{i:r_i=1}e_i.
$$

如果 $e_i$ 独立、均值为 $0$、方差为 $\sigma_e^2$，则：

$$
\mathbb{E}[N_{\rm dec}\mid\mathbf{r}]=0.
$$

方差为：

$$
\operatorname{Var}(N_{\rm dec}\mid\mathbf{r})=h\sigma_e^2.
$$

### Chebyshev 界

Chebyshev 不等式给出：

$$
\Pr\left[|N_{\rm dec}|\geq\frac{q}{4}\mid\mathbf{r}\right]
\leq
\frac{h\sigma_e^2}{(q/4)^2}.
$$

化简为：

$$
\Pr\left[|N_{\rm dec}|\geq\frac{q}{4}\mid\mathbf{r}\right]
\leq
\frac{16h\sigma_e^2}{q^2}.
$$

这个界只用方差，通常较松，但它展示了正确性参数之间的关系：失败概率随 $h$ 和 $\sigma_e^2$ 增大，随 $q^2$ 增大而减小。

### Hoeffding 界

若进一步知道：

$$
e_i\in[-B_e,B_e],
$$

并且被选中的 $e_i$ 独立，则 Hoeffding 双侧界给出：

$$
\Pr\left[|N_{\rm dec}|\geq T\mid\mathbf{r}\right]
\leq
2\exp\left(-\frac{T^2}{2hB_e^2}\right).
$$

取 $T=q/4$，得到：

$$
\Pr\left[|N_{\rm dec}|\geq\frac{q}{4}\mid\mathbf{r}\right]
\leq
2\exp\left(-\frac{q^2}{32hB_e^2}\right).
$$

这个界比 Chebyshev 更能反映独立小噪声相加时的指数衰减。

### 精确卷积思路

如果误差分布 $\chi_e$ 支持集有限，可以精确计算被选中 $h$ 个误差之和的分布。设 $P_e$ 是单个误差的概率质量函数，则：

$$
P_{N\mid h}=\underbrace{P_e*P_e*\cdots*P_e}_{h\text{ 次}}.
$$

失败概率为：

$$
\Pr[E_{\rm fail}\mid h] = \sum_{|z|\geq q/4}P_{N\mid h}(z).
$$

如果 $\mathbf{r}$ 的 Hamming weight 也是随机的，还需要对 $h$ 的分布取平均：

$$
\Pr[E_{\rm fail}]=\sum_h\Pr[\|\mathbf{r}\|_0=h]\Pr[E_{\rm fail}\mid h].
$$

## 小结

Regev 型 LWE 加密的正确性结构非常清楚：

$$
\mathsf{Dec}(\mathsf{sk},\mathsf{ct})\text{ 的中间值} = \mathsf{Encode}(\mu)+\mathbf{e}^{\top}\mathbf{r}.
$$

因此总噪声为：

$$
N_{\rm dec}=\mathbf{e}^{\top}\mathbf{r}.
$$

正确性分析就是控制：

$$
|\langle \mathbf{e}^{\top}\mathbf{r}\rangle_q|<\frac{q}{4}.
$$

这个案例体现了格基加密正确性证明的基本模式：先代入公式，消去主线性项，再把剩余误差写成总噪声，最后用范围界、尾界或精确卷积估计失败概率。
