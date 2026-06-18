# LPR 型 LWE 公钥加密正确性分析

LPR 型 LWE 公钥加密可以看作许多现代格基加密方案的基础模板。它比 Regev 型方案多了加密误差项，因此总误差不再只是 $\mathbf{e}^{\top}\mathbf{r}$，而是由三类噪声共同组成：密钥生成误差与加密随机性的乘积、加密标量误差、秘密与加密向量误差的乘积。

这个结构也是 FrodoPKE、RLWE PKE 和 ML-KEM 核心 PKE 的共同原型。

## 方案对象与参数

### 基本空间

取模数 $q$，矩阵维数 $m,n$。令：

$$
\mathbf{A}\in\mathbb{Z}_q^{m\times n}.
$$

私钥秘密为：

$$
\mathbf{s}\in\mathbb{Z}^n.
$$

密钥生成误差为：

$$
\mathbf{e}\in\mathbb{Z}^m.
$$

加密随机向量为：

$$
\mathbf{r}\in\mathbb{Z}^m.
$$

加密向量误差为：

$$
\mathbf{e}_1\in\mathbb{Z}^n.
$$

加密标量误差为：

$$
e_2\in\mathbb{Z}.
$$

这些小整数变量最终都嵌入 $\mathbb{Z}_q$ 中参与模运算。

### 消息编码

消息为一比特：

$$
\mu\in\{0,1\}.
$$

采用编码：

$$
\mathsf{Encode}(0):=0,
$$

$$
\mathsf{Encode}(1):=\left\lfloor\frac{q}{2}\right\rceil.
$$

记编码间隔为：

$$
\Delta:=\left\lfloor\frac{q}{2}\right\rceil.
$$

若总噪声的中心代表元绝对值小于 $\Delta/2$，则解码正确。

## 方案描述

### 密钥生成

采样公共矩阵：

$$
\mathbf{A}\xleftarrow{\$}\mathbb{Z}_q^{m\times n}.
$$

采样秘密和误差：

$$
\mathbf{s}\leftarrow\chi_s^n,
$$

$$
\mathbf{e}\leftarrow\chi_e^m.
$$

计算公钥向量：

$$
\mathbf{t}:=\mathbf{A}\mathbf{s}+\mathbf{e}\pmod q.
$$

公钥为：

$$
\mathsf{pk}:=(\mathbf{A},\mathbf{t}).
$$

私钥为：

$$
\mathsf{sk}:=\mathbf{s}.
$$

### 加密

加密消息 $\mu$ 时，采样：

$$
\mathbf{r}\leftarrow\chi_r^m,
$$

$$
\mathbf{e}_1\leftarrow\chi_1^n,
$$

$$
e_2\leftarrow\chi_2.
$$

计算：

$$
\mathbf{u}:=\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}_1\pmod q,
$$

$$
v:=\mathbf{t}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(\mu)\pmod q.
$$

密文为：

$$
\mathsf{ct}:=(\mathbf{u},v).
$$

### 解密

解密时计算：

$$
w:=v-\mathbf{s}^{\top}\mathbf{u}\pmod q.
$$

然后对 $w$ 做中心提升与解码，得到消息比特。

## 正确性推导

### 代入公钥表达式

由公钥定义：

$$
\mathbf{t}=\mathbf{A}\mathbf{s}+\mathbf{e}.
$$

因此：

$$
\mathbf{t}^{\top}\mathbf{r}=(\mathbf{A}\mathbf{s}+\mathbf{e})^{\top}\mathbf{r}.
$$

展开：

$$
\mathbf{t}^{\top}\mathbf{r}=\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}^{\top}\mathbf{r}.
$$

### 代入密文表达式

由加密定义：

$$
v=\mathbf{t}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(\mu).
$$

又有：

$$
\mathbf{u}=\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}_1.
$$

所以：

$$
\mathbf{s}^{\top}\mathbf{u}=\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}+\mathbf{s}^{\top}\mathbf{e}_1.
$$

### 消去主项

将上述表达式代入解密值：

$$
\begin{aligned}
w
&=v-\mathbf{s}^{\top}\mathbf{u} \\
&=\mathbf{t}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(\mu)-\mathbf{s}^{\top}(\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}_1) \\
&=(\mathbf{A}\mathbf{s}+\mathbf{e})^{\top}\mathbf{r}+e_2+\mathsf{Encode}(\mu)-\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}-\mathbf{s}^{\top}\mathbf{e}_1 \\
&=\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}+\mathbf{e}^{\top}\mathbf{r}+e_2+\mathsf{Encode}(\mu)-\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}-\mathbf{s}^{\top}\mathbf{e}_1 \\
&=\mathsf{Encode}(\mu)+\mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1.
\end{aligned}
$$

主项 $\mathbf{s}^{\top}\mathbf{A}^{\top}\mathbf{r}$ 完全抵消。

### 总解密噪声

定义：

$$
N_{\rm dec}:=\mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1.
$$

于是：

$$
w=\mathsf{Encode}(\mu)+N_{\rm dec}\pmod q.
$$

正确性分析归结为控制 $N_{\rm dec}$ 的大小。

## 总误差结构分析

### 噪声项分解

总噪声包含三项：

$$
N_{\rm dec}=N_1+N_2-N_3,
$$

其中：

$$
N_1:=\mathbf{e}^{\top}\mathbf{r}=\sum_{i=1}^{m}e_ir_i,
$$

$$
N_2:=e_2,
$$

$$
N_3:=\mathbf{s}^{\top}\mathbf{e}_1=\sum_{j=1}^{n}s_j(e_1)_j.
$$

这三个部分的来源不同：$N_1$ 来自密钥生成误差和加密随机性，$N_2$ 是加密标量误差，$N_3$ 来自秘密和加密向量误差。

### 正确性条件

解码成功的充分条件是：

$$
|\langle N_{\rm dec}\rangle_q|<\frac{\Delta}{2}.
$$

对于二元编码 $\Delta\approx q/2$，常写成：

$$
|\langle N_{\rm dec}\rangle_q|<\frac{q}{4}.
$$

失败事件为：

$$
E_{\rm fail}:=\left\{|\langle \mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1\rangle_q|\geq\frac{\Delta}{2}\right\}.
$$

## 范围界分析

### 有界分布假设

假设所有小变量有如下确定性界：

$$
|e_i|\leq B_e,
$$

$$
|r_i|\leq B_r,
$$

$$
|s_j|\leq B_s,
$$

$$
|(e_1)_j|\leq B_1,
$$

$$
|e_2|\leq B_2.
$$

### 分别界定三项

第一项满足：

$$
|\mathbf{e}^{\top}\mathbf{r}|
=\left|\sum_{i=1}^{m}e_ir_i\right|
\leq
\sum_{i=1}^{m}|e_i||r_i|
\leq
mB_eB_r.
$$

第二项满足：

$$
|e_2|\leq B_2.
$$

第三项满足：

$$
|\mathbf{s}^{\top}\mathbf{e}_1|
=\left|\sum_{j=1}^{n}s_j(e_1)_j\right|
\leq
\sum_{j=1}^{n}|s_j||(e_1)_j|
\leq
nB_sB_1.
$$

因此：

$$
|N_{\rm dec}|\leq mB_eB_r+B_2+nB_sB_1.
$$

### 零失败充分条件

若：

$$
mB_eB_r+B_2+nB_sB_1<\frac{\Delta}{2},
$$

则必然解密正确。

对于二元编码，可写成：

$$
mB_eB_r+B_2+nB_sB_1<\frac{q}{4}.
$$

这个条件非常**保守**，因为它假设所有乘积项同号叠加。实际方案通常依赖概率界或精确分布，而不是仅靠范围界。

## 方差界分析

### 零均值与独立性假设

假设以下变量相互独立，且均值为 $0$：

$$
e_i,r_i,s_j,(e_1)_j,e_2.
$$

设方差分别为：

$$
\operatorname{Var}(e_i)=\sigma_e^2,
$$

$$
\operatorname{Var}(r_i)=\sigma_r^2,
$$

$$
\operatorname{Var}(s_j)=\sigma_s^2,
$$

$$
\operatorname{Var}((e_1)_j)=\sigma_1^2,
$$

$$
\operatorname{Var}(e_2)=\sigma_2^2.
$$

### 计算乘积项方差

对于 $e_ir_i$，由于 $e_i$ 和 $r_i$ 独立且均值为 $0$：

$$
\operatorname{Var}(e_ir_i)=\mathbb{E}[e_i^2r_i^2].
$$

由独立性：

$$
\mathbb{E}[e_i^2r_i^2]=\mathbb{E}[e_i^2]\mathbb{E}[r_i^2].
$$

因为二者均值为 $0$，所以：

$$
\mathbb{E}[e_i^2]=\sigma_e^2,
$$

$$
\mathbb{E}[r_i^2]=\sigma_r^2.
$$

因此：

$$
\operatorname{Var}(e_ir_i)=\sigma_e^2\sigma_r^2.
$$

同理：

$$
\operatorname{Var}(s_j(e_1)_j)=\sigma_s^2\sigma_1^2.
$$

### 总方差

若各乘积项之间也可视为独立，则：

$$
\operatorname{Var}(\mathbf{e}^{\top}\mathbf{r})=m\sigma_e^2\sigma_r^2.
$$

并且：

$$
\operatorname{Var}(\mathbf{s}^{\top}\mathbf{e}_1)=n\sigma_s^2\sigma_1^2.
$$

由于 $e_2$ 独立，得到：

$$
\operatorname{Var}(N_{\rm dec})
=m\sigma_e^2\sigma_r^2+
\sigma_2^2+
 n\sigma_s^2\sigma_1^2.
$$

记：

$$
\sigma_N^2:=m\sigma_e^2\sigma_r^2+
\sigma_2^2+
 n\sigma_s^2\sigma_1^2.
$$

### Chebyshev 正确性界

由 Chebyshev 不等式：

$$
\Pr\left[|N_{\rm dec}|\geq\frac{\Delta}{2}\right]
\leq
\frac{\sigma_N^2}{(\Delta/2)^2}.
$$

即：

$$
\Pr[E_{\rm fail}]
\leq
\frac{4\sigma_N^2}{\Delta^2}.
$$

这个界通常很松，但它清楚说明了正确性和方差之间的关系。

## Bernstein 界分析

### 单项最大幅度

对于 $e_ir_i$，有：

$$
|e_ir_i|\leq B_eB_r.
$$

对于 $s_j(e_1)_j$，有：

$$
|s_j(e_1)_j|\leq B_sB_1.
$$

对于 $e_2$，有：

$$
|e_2|\leq B_2.
$$

令统一最大幅度为：

$$
M:=\max\{B_eB_r,B_sB_1,B_2\}.
$$

### Bernstein 参数

令总方差参数为：

$$
\nu^2:=m\sigma_e^2\sigma_r^2+\sigma_2^2+n\sigma_s^2\sigma_1^2.
$$

若相关独立性前提成立，则 Bernstein 双侧界给出：

$$
\Pr[|N_{\rm dec}|\geq T]
\leq
2\exp\left(-\frac{T^2}{2(\nu^2+MT/3)}\right).
$$

取：

$$
T:=\frac{\Delta}{2},
$$

得到：

$$
\Pr[E_{\rm fail}]
\leq
2\exp\left(-\frac{\Delta^2/4}{2(\nu^2+M\Delta/6)}\right).
$$

化简为：

$$
\Pr[E_{\rm fail}]
\leq
2\exp\left(-\frac{\Delta^2}{8\nu^2+(4/3)M\Delta}\right).
$$

这个界比范围界更贴近典型噪声行为，但必须明确**独立性和有界性**前提。

## 小结

LPR 型 LWE 公钥加密的正确性推导得到：

$$
w=\mathsf{Encode}(\mu)+N_{\rm dec}\pmod q.
$$

其中总解密噪声为：

$$
N_{\rm dec}=\mathbf{e}^{\top}\mathbf{r}+e_2-\mathbf{s}^{\top}\mathbf{e}_1.
$$

正确性条件为：

$$
|\langle N_{\rm dec}\rangle_q|<\frac{\Delta}{2}.
$$

范围界给出零失败充分条件：

$$
mB_eB_r+B_2+nB_sB_1<\frac{\Delta}{2}.
$$

概率界则利用方差和最大幅度估计失败概率。这个案例是后续 Frodo 型矩阵 LWE、环 LWE 和 ML-KEM 型模块 LWE 正确性分析的共同模板。
