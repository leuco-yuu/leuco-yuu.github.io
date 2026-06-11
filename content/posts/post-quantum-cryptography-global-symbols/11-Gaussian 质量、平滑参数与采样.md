# Gaussian 质量、平滑参数与采样

## Gaussian 参数规范

> **重要统一规则**：理论格文献常用 $\rho_s(\mathbf x)=e^{-\pi\|\mathbf x\|^2/s^2}$，其中 $s$ 不是概率统计中的标准差。对应连续一维 Gaussian 的标准差为 $\sigma=s/\sqrt{2\pi}$。全书必须显式区分 $s$ 与 $\sigma$。

|符号|定义|
|-|-|
|$\rho_s(\mathbf x)$|$\exp(-\pi\|\mathbf x\|_2^2/s^2)$|
|$\rho_s(S)$|$\sum_{\mathbf x\in S}\rho_s(\mathbf x)$，连续集合时改用积分|
|$\rho_{s,\mathbf c}(\mathbf x)$|$\rho_s(\mathbf x-\mathbf c)$|
|$D_{\Lambda,s,\mathbf c}$|概率质量 $\rho_{s,\mathbf c}(\mathbf x)/\rho_{s,\mathbf c}(\Lambda)$|
|$s$|理论 Gaussian 参数|
|$\sigma$|统计标准差|
|$s=\sqrt{2\pi}\sigma$|两种参数的换算|
|$\eta_\varepsilon(\Lambda)$|满足 $\rho_{1/s}(\Lambda^*\setminus\{0\})\le\varepsilon$ 的最小 $s$|

## 采样器接口

|符号|含义|
|-|-|
|$\mathsf{SampleU}(S)$|均匀采样|
|$\mathsf{SampleCBD}(\eta)$|CBD 采样|
|$\mathsf{SampleD}(\Lambda,s,\mathbf c)$|离散高斯采样|
|$\mathsf{SamplePre}$|给定综合和陷门的短预像采样|
|$\mathsf{SampleLeft}$|左扩展矩阵的预像采样|
|$\mathsf{SampleRight}$|右扩展矩阵的预像采样|
|$\mathsf{RejSample}$|拒绝采样|
|$M_{\rm rej}$|拒绝采样包络常数|
|$p_{\rm acc}$|单次接受概率|
|$N_{\rm retry}$|重试次数|
|$\epsilon_{\rm samp}$|实现分布与理想分布的统计距离|

## 分布实现与精度

|符号|含义|
|-|-|
|$B_{\rm trunc}$|截断界|
|$\epsilon_{\rm trunc}$|截断尾概率|
|$\epsilon_{\rm fp}$|浮点/定点近似误差|
|$\epsilon_{\rm table}$|概率表量化误差|
|$w_{\rm fp}$|浮点或定点有效位数|
|$\mathsf{CDT}$|累积分布表|
|$\mathsf{KY}$|Knuth–Yao 采样器|
|$\mathsf{BerExp}$|Bernoulli exponential 采样器|

---

