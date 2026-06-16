# ML-KEM 型模块 LWE 正确性分析

ML-KEM 的底层公钥加密核心可以理解为模块 LWE 上的压缩型加密方案。它把环 LWE 从单个环元素推广到 $R_q^k$ 上的模块向量，并在密文中加入压缩与解压步骤。正确性分析的核心仍然是同一个目标：把解密值整理为消息编码加总噪声，然后证明总噪声不会越过判决边界。

本文分析的是 ML-KEM 型公钥加密核心的抽象版本，而不是完整标准中的所有哈希、种子展开、拒绝采样、编码细节和 CCA 变换。这样做可以把注意力集中到正确性分析最关键的代数结构和噪声表达式上。

## 基本对象

设：

$$
R_q:=\mathbb{Z}_q[X]/(X^n+1).
$$

模块秩为 $k$。向量 $\mathbf{s}\in R_q^k$ 是由 $k$ 个环元素组成的列向量，矩阵 $\mathbf{A}\in R_q^{k\times k}$ 是由环元素组成的矩阵。

矩阵向量乘法在 $R_q$ 上进行。若：

$$
\mathbf{A}=(a_{ij})_{i,j\in[k]},
$$

$$
\mathbf{s}=(s_1,\ldots,s_k)^{\top},
$$

则第 $i$ 个分量为：

$$
(\mathbf{A}\mathbf{s})_i=\sum_{j=1}^k a_{ij}\star s_j.
$$

其中 $\star$ 表示 $R_q$ 中的负循环卷积乘法。

## 方案描述

### 密钥生成

密钥生成采样：

$$
\mathbf{A}\xleftarrow{\$}R_q^{k\times k},
$$

$$
\mathbf{s}\leftarrow\chi_s^{R,k},
$$

$$
\mathbf{e}\leftarrow\chi_e^{R,k}.
$$

其中 $\chi_s^{R,k}$ 表示向量中每个环元素的每个系数都从短分布采样。然后计算：

$$
\mathbf{t}:=\mathbf{A}\mathbf{s}+\mathbf{e}\pmod q.
$$

公钥和私钥为：

$$
\mathsf{pk}:=(\mathbf{A},\mathbf{t}),
$$

$$
\mathsf{sk}:=\mathbf{s}.
$$

### 加密

消息先被编码成环元素 $\mathsf{Encode}(m)\in R_q$。加密时采样：

$$
\mathbf{r}\leftarrow\chi_r^{R,k},
$$

$$
\mathbf{e}_1\leftarrow\chi_1^{R,k},
$$

$$
e_2\leftarrow\chi_2^R.
$$

计算未压缩密文分量：

$$
\mathbf{u}:=\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}_1\pmod q,
$$

$$
v:=\mathbf{t}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(m)\pmod q.
$$

实际方案会对 $\mathbf{u}$ 和 $v$ 进行压缩。抽象地写作：

$$
\widehat{\mathbf{u}}:=\mathsf{Compress}(\mathbf{u},d_u),
$$

$$
\widehat{v}:=\mathsf{Compress}(v,d_v).
$$

密文为：

$$
\mathsf{ct}:=(\widehat{\mathbf{u}},\widehat{v}).
$$

### 解密

解密时先解压：

$$
\widetilde{\mathbf{u}}:=\mathsf{Decompress}(\widehat{\mathbf{u}},d_u),
$$

$$
\widetilde{v}:=\mathsf{Decompress}(\widehat{v},d_v).
$$

然后计算：

$$
w:=\widetilde{v}-\mathbf{s}^{\top}\widetilde{\mathbf{u}}\pmod q.
$$

最后对 $w$ 的系数做判决，恢复消息比特。

## 无压缩情形的总误差推导

先忽略压缩误差，分析核心代数抵消。若 $\widetilde{\mathbf{u}}=\mathbf{u}$ 且 $\widetilde{v}=v$，则解密值为：

$$
w=v-\mathbf{s}^{\top}\mathbf{u}.
$$

代入 $v$：

$$
w=\mathbf{t}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(m)-\mathbf{s}^{\top}\mathbf{u}.
$$

代入 $\mathbf{u}=\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}_1$：

$$
w=\mathbf{t}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(m)-\mathbf{s}^{\top}(\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}_1).
$$

展开：

$$
w=\mathbf{t}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(m)-\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}-\mathbf{s}^{\top}\mathbf{e}_1.
$$

再代入 $\mathbf{t}=\mathbf{A}\mathbf{s}+\mathbf{e}$：

$$
w=(\mathbf{A}\mathbf{s}+\mathbf{e})^{\top}\mathbf{r}+e_2+\mathsf{Encode}(m)-\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}-\mathbf{s}^{\top}\mathbf{e}_1.
$$

展开公钥项：

$$
w=(\mathbf{A}\mathbf{s})^{\top}\mathbf{r}+\mathbf{e}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(m)-\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}-\mathbf{s}^{\top}\mathbf{e}_1.
$$

在交换环 $R_q$ 上有：

$$
(\mathbf{A}\mathbf{s})^{\top}\mathbf{r}=\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}.
$$

因此主项抵消，得到：

$$
w=\mathsf{Encode}(m)+\mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1.
$$

于是无压缩情形的总解密噪声为：

$$
N_{\rm dec}^{(0)}:=\mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1.
$$

## 加入压缩误差

实际密文传输的是压缩值。定义解压后的误差：

$$
\mathbf{E}_{\mathbf{u}}:=\widetilde{\mathbf{u}}-\mathbf{u},
$$

$$
E_v:=\widetilde{v}-v.
$$

因此：

$$
\widetilde{\mathbf{u}}=\mathbf{u}+\mathbf{E}_{\mathbf{u}},
$$

$$
\widetilde{v}=v+E_v.
$$

解密值为：

$$
w=\widetilde{v}-\mathbf{s}^{\top}\widetilde{\mathbf{u}}.
$$

代入解压表达式：

$$
w=(v+E_v)-\mathbf{s}^{\top}(\mathbf{u}+\mathbf{E}_{\mathbf{u}}).
$$

展开：

$$
w=v-\mathbf{s}^{\top}\mathbf{u}+E_v-\mathbf{s}^{\top}\mathbf{E}_{\mathbf{u}}.
$$

再使用无压缩推导结果：

$$
w=\mathsf{Encode}(m)+\mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1+E_v-\mathbf{s}^{\top}\mathbf{E}_{\mathbf{u}}.
$$

因此压缩情形下的总解密噪声为：

$$
N_{\rm dec}:=\mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1+E_v-\mathbf{s}^{\top}\mathbf{E}_{\mathbf{u}}.
$$

这是 ML-KEM 型正确性分析中最重要的表达式。每一项都有明确来源：

| 噪声项 | 来源 | 说明 |
| :--- | :--- | :--- |
| $\mathbf{e}^{\top}\mathbf{r}$ | 公钥误差与加密随机性 | 模块内积，含 $k$ 个环乘积 |
| $e_2$ | 第二密文分量误差 | 直接加到消息编码上 |
| $-\mathbf{s}^{\top}\mathbf{e}_1$ | 私钥与第一密文误差 | 模块内积，含 $k$ 个环乘积 |
| $E_v$ | $v$ 的压缩误差 | 解压缩后与原值的差 |
| $-\mathbf{s}^{\top}\mathbf{E}_{\mathbf{u}}$ | $\mathbf{u}$ 的压缩误差经私钥放大 | 压缩误差会被私钥内积放大 |

{tableMode="stretch"}

## 压缩误差的抽象界

为了分析正确性，需要给压缩误差设定界。常见压缩形式可以抽象为：

$$
\mathsf{Compress}_q(x,d):=\left\lfloor \frac{2^d}{q}x\right\rceil\pmod {2^d}.
$$

对应解压为：

$$
\mathsf{Decompress}_q(y,d):=\left\lfloor \frac{q}{2^d}y\right\rceil\pmod q.
$$

这里 $d$ 是保留的位宽。$d$ 越小，压缩越强，通信越省，但误差越大。

定义单系数压缩误差上界：

$$
B_{\rm comp}(d):=\max_x |\langle \mathsf{Decompress}_q(\mathsf{Compress}_q(x,d),d)-x\rangle_q|.
$$

通常它与 $q/2^{d+1}$ 同阶，但具体常数取决于舍入规则和边界约定。为了避免符号不规范，本文不把它强行写成固定值，而统一记为 $B_{\rm comp}(d)$。

于是可以记：

$$
|(E_v)_j|\leq B_v,
$$

$$
|(\mathbf{E}_{\mathbf{u}})_{i,j}|\leq B_u.
$$

其中 $B_v:=B_{\rm comp}(d_v)$，$B_u:=B_{\rm comp}(d_u)$。

## 系数级噪声展开

记 $N_j$ 为 $N_{\rm dec}$ 的第 $j$ 个系数。先看 $\mathbf{e}^{\top}\mathbf{r}$：

$$
\mathbf{e}^{\top}\mathbf{r}=\sum_{a=1}^k e_a\star r_a.
$$

其中每个 $e_a\star r_a$ 的第 $j$ 个系数又是 $n$ 个乘积项之和。因此：

$$
(\mathbf{e}^{\top}\mathbf{r})_j=\sum_{a=1}^k\sum_{i=0}^{n-1}\eta_{a,i,j}(e_a)_i(r_a)_{\ell(a,i,j)}.
$$

这里 $\eta_{a,i,j}\in\{+1,-1\}$ 是负循环卷积产生的符号，$\ell(a,i,j)$ 是由系数索引关系确定的下标。

类似地：

$$
(\mathbf{s}^{\top}\mathbf{e}_1)_j=\sum_{a=1}^k\sum_{i=0}^{n-1}\theta_{a,i,j}(s_a)_i(e_{1,a})_{\ell'(a,i,j)}.
$$

压缩误差项满足：

$$
(\mathbf{s}^{\top}\mathbf{E}_{\mathbf{u}})_j=\sum_{a=1}^k\sum_{i=0}^{n-1}\rho_{a,i,j}(s_a)_i(E_{\mathbf{u},a})_{\ell''(a,i,j)}.
$$

因此：

$$
\begin{aligned}
N_j={}&\sum_{a=1}^k\sum_{i=0}^{n-1}\eta_{a,i,j}(e_a)_i(r_a)_{\ell(a,i,j)}+(e_2)_j \\
&-\sum_{a=1}^k\sum_{i=0}^{n-1}\theta_{a,i,j}(s_a)_i(e_{1,a})_{\ell'(a,i,j)}+(E_v)_j \\
&-\sum_{a=1}^k\sum_{i=0}^{n-1}\rho_{a,i,j}(s_a)_i(E_{\mathbf{u},a})_{\ell''(a,i,j)}.
\end{aligned}
$$

这个公式说明：模块秩 $k$ 和环次数 $n$ 会共同放大单个系数中的乘积项数量。相较环 LWE，ML-KEM 型方案的一个系数噪声中有 $kn$ 级别的乘积项。

## 正确性条件

### 单系数判决条件

如果消息系数使用二元编码，通常把 $0$ 和 $1$ 放在相距约 $q/2$ 的两个区域。此时单系数正确的充分条件为：

$$
|\langle N_j\rangle_q|<\frac{q}{4}.
$$

更一般地，若编码间隔为 $\Delta$，则条件为：

$$
|\langle N_j\rangle_q|<\frac{\Delta}{2}.
$$

### 整体失败事件

消息环元素有 $n$ 个系数。定义失败事件：

$$
E_{\rm fail}:=\bigcup_{j=0}^{n-1}\left\{|\langle N_j\rangle_q|\geq\frac{\Delta}{2}\right\}.
$$

由并合界：

$$
\Pr[E_{\rm fail}]\leq\sum_{j=0}^{n-1}\Pr\left[|\langle N_j\rangle_q|\geq\frac{\Delta}{2}\right].
$$

若每个系数失败概率不超过 $p_{\rm coeff}$，则：

$$
\Pr[E_{\rm fail}]\leq n p_{\rm coeff}.
$$

该界不要求各系数独立。因此即使卷积和模块内积造成系数相关，也可以使用。

## 最坏情况幅度界

假设系数满足：

$$
|(e_a)_i|\leq B_e,
$$

$$
|(r_a)_i|\leq B_r,
$$

$$
|(s_a)_i|\leq B_s,
$$

$$
|(e_{1,a})_i|\leq B_1,
$$

$$
|(e_2)_j|\leq B_2,
$$

$$
|(E_v)_j|\leq B_v,
$$

$$
|(E_{\mathbf{u},a})_i|\leq B_u.
$$

对第 $j$ 个系数，$\mathbf{e}^{\top}\mathbf{r}$ 含有 $kn$ 个乘积项，每项幅度不超过 $B_eB_r$，因此：

$$
|(\mathbf{e}^{\top}\mathbf{r})_j|\leq knB_eB_r.
$$

同理：

$$
|(\mathbf{s}^{\top}\mathbf{e}_1)_j|\leq knB_sB_1.
$$

压缩误差放大项满足：

$$
|(\mathbf{s}^{\top}\mathbf{E}_{\mathbf{u}})_j|\leq knB_sB_u.
$$

因此：

$$
|N_j|\leq knB_eB_r+B_2+knB_sB_1+B_v+knB_sB_u.
$$

一个充分的确定性正确性条件是：

$$
knB_eB_r+B_2+knB_sB_1+B_v+knB_sB_u<\frac{\Delta}{2}.
$$

这个条件非常保守，因为它假设所有乘积项都以最坏符号同向叠加。实际参数分析通常不会只依赖这个界，而会使用更紧的概率尾界或有证数值计算。

## 方差与典型噪声规模

为了估计典型失败概率，记各噪声系数方差为：

$$
\operatorname{Var}((e_a)_i)=\sigma_e^2,
$$

$$
\operatorname{Var}((r_a)_i)=\sigma_r^2,
$$

$$
\operatorname{Var}((s_a)_i)=\sigma_s^2,
$$

$$
\operatorname{Var}((e_{1,a})_i)=\sigma_1^2,
$$

$$
\operatorname{Var}((e_2)_j)=\sigma_2^2.
$$

若暂时把压缩误差也建模为零均值噪声，记：

$$
\operatorname{Var}((E_v)_j)=\sigma_v^2,
$$

$$
\operatorname{Var}((E_{\mathbf{u},a})_i)=\sigma_u^2.
$$

在独立性或条件化处理成立的理想化模型下，有近似方差：

$$
\nu^2:=kn\sigma_e^2\sigma_r^2+\sigma_2^2+kn\sigma_s^2\sigma_1^2+\sigma_v^2+kn\sigma_s^2\sigma_u^2.
$$

这里每一项对应总噪声中的一个来源：

- $kn\sigma_e^2\sigma_r^2$ 来自 $\mathbf{e}^{\top}\mathbf{r}$；
- $\sigma_2^2$ 来自 $e_2$；
- $kn\sigma_s^2\sigma_1^2$ 来自 $\mathbf{s}^{\top}\mathbf{e}_1$；
- $\sigma_v^2$ 来自 $v$ 的压缩误差；
- $kn\sigma_s^2\sigma_u^2$ 来自 $\mathbf{u}$ 的压缩误差经私钥内积放大。

需要强调：压缩误差通常是密文分量的确定性函数，不一定独立，也不一定零均值。把它写成方差项是一种分析近似或上界模型。严格证明应使用标准参数分析中的离散分布计算、压缩误差枚举或有证数值脚本。

## Bernstein 型正确性上界

把单个系数噪声写成中心化项之和：

$$
N_j=\sum_i X_i.
$$

设：

$$
\mathbb{E}[X_i]=0,
$$

$$
|X_i|\leq M,
$$

$$
\sum_i\operatorname{Var}(X_i)\leq \nu^2.
$$

其中可以取：

$$
M:=\max\{B_eB_r,B_sB_1,B_2,B_v,B_sB_u\}.
$$

于是双侧 Bernstein 型界给出：

$$
\Pr[|N_j|\geq t]\leq 2\exp\left(-\frac{t^2}{2(\nu^2+Mt/3)}\right).
$$

令：

$$
t:=\frac{\Delta}{2}.
$$

得到：

$$
p_{\rm coeff}\leq 2\exp\left(-\frac{(\Delta/2)^2}{2(\nu^2+M\Delta/6)}\right).
$$

整体失败概率满足：

$$
\Pr[E_{\rm fail}]\leq 2n\exp\left(-\frac{(\Delta/2)^2}{2(\nu^2+M\Delta/6)}\right).
$$

这个公式清楚展示了模块 LWE 参数的相互关系：$k$ 和 $n$ 增大会增大噪声方差，$d_u,d_v$ 通过压缩误差影响 $B_u,B_v$，而 $q$ 和编码间隔 $\Delta$ 决定解码容忍度。

## 标准方案中的失败率估计方式

真实 ML-KEM 的失败率不是简单由上述通用 Bernstein 公式给出。标准化参数通常使用更细的分布分析和脚本计算，因为：

- 噪声来自具体的中心二项分布；
- 压缩误差由确定性舍入产生；
- 模块环卷积带来结构相关；
- 实际失败事件与编码、压缩和解码规则精确相关；
- 目标失败率极低，普通随机实验无法直接证明。

因此，标准分析会围绕精确或有证的单系数分布、尾部概率和整体并合界展开。公开标准中给出的失败率可以作为参数设计结果，而不是由一行简单尾界直接推出。

## 小结

ML-KEM 型模块 LWE 正确性分析的核心公式是：

$$
N_{\rm dec}:=\mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1+E_v-\mathbf{s}^{\top}\mathbf{E}_{\mathbf{u}}.
$$

与普通 LWE 和环 LWE 相比，它额外包含两个重要放大因素：

- 模块秩 $k$ 使每个系数中出现 $kn$ 个乘积项；
- 密文压缩引入 $E_v$ 和 $\mathbf{E}_{\mathbf{u}}$，其中 $\mathbf{E}_{\mathbf{u}}$ 还会被私钥内积放大。

因此，正确性分析必须同时记录噪声分布、模块维数、环次数、压缩位宽、编码间隔和失败事件。只要能证明每个系数的总噪声满足：

$$
|\langle N_j\rangle_q|<\frac{\Delta}{2},
$$

就能保证该系数解码正确；再通过并合界或更精细的相关性分析得到整体解密失败率。
