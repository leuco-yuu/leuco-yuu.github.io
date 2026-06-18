# Gaussian、平滑参数与采样

## Gaussian 规范

理论格文献统一采用

\[
\rho_s(\mathbf{x})=\exp\!\left(-\pi\frac{\|\mathbf{x}\|_2^2}{s^2}\right).
\]

其中 $s$ 是 Gaussian 参数，不是标准差；对应一维连续 Gaussian 的标准差为 $\sigma=s/\sqrt{2\pi}$。

|符号|含义|
|-|-|
|$\rho_s(\mathbf{x})$|Gaussian 质量函数|
|$\rho_s(S)$|集合 $S$ 上的 Gaussian 总质量|
|$\rho_{s,\mathbf{c}}(\mathbf{x})$|以 $\mathbf{c}$ 为中心的 Gaussian 质量|
|$D_{\Lambda,s,\mathbf{c}}$|格或格陪集上的离散 Gaussian|
|$\eta_\varepsilon(\Lambda)$|平滑参数|

## 常用采样接口

|符号|含义|
|-|-|
|$\mathcal{U}(S)$|集合$S$上的均匀分布|
|$u\overset{$}{\leftarrow}S$|从集合$S$中均匀随机取样$u$|
|$v\leftarrow\mathcal{D}$|从分布$\mathcal{D}$中采样$v$|
|$w\leftarrow\mathsf{Sample}(\mathcal{D};r)$|从分布$\mathcal{D}$中以种子$r$确定性采样$v$|
|$\mathsf{SamplePre}$|短预像采样|
|$\mathsf{RejSample}$|拒绝采样|
|$\epsilon_{\rm samp}$|实现分布与理想分布的统计距离|

## 陷门与 Gadget

|符号|含义|
|-|-|
|$\mathbf{T}_{\mathbf{A}}$|矩阵 $\mathbf{A}$ 的陷门|
|$\mathsf{TrapGen}$|陷门生成算法|
|$b_{\rm gad}$|Gadget 分解基数|
|$\mathbf{g}^{\top}=(1,b_{\rm gad},\ldots,b_{\rm gad}^{\ell_g-1})$|Gadget 向量|
|$\mathbf{G}=\mathbf{I}_n\otimes\mathbf{g}^{\top}$|Gadget 矩阵|
|$\operatorname{GInv}$|Gadget 逆分解|
