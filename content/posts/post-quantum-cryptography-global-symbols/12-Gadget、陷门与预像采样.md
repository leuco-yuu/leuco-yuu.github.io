# Gadget、陷门与预像采样

## Gadget 记号

|符号|定义或含义|
|-|-|
|$b_{\rm gad}\ge2$|Gadget 分解基数|
|$\ell_g:=\lceil\log_{b_{\rm gad}}q\rceil$|Gadget 位数/分解长度|
|$\mathbf g^{\top}:=(1,b_{\rm gad},\ldots,b_{\rm gad}^{\ell_g-1})$|Gadget 向量|
|$\mathbf G:=\mathbf I_n\otimes\mathbf g^{\top}$|Gadget 矩阵|
|$\operatorname{GInv}(\mathbf u)$|满足 $\mathbf G\operatorname{GInv}(\mathbf u)=\mathbf u\bmod q$ 的规范分解|
|$\operatorname{BitDecomp}$|$b_{\rm gad}=2$ 时的位分解|
|$\operatorname{PowersOfBase}$|幂基扩展|

## 陷门接口

|符号|含义|
|-|-|
|$(\mathbf A,\mathbf T_{\mathbf A})\leftarrow\mathsf{TrapGen}(1^\lambda,n,m,q)$|生成近均匀公开矩阵及其陷门|
|$\mathbf T_{\mathbf A}$|$\Lambda_q^\perp(\mathbf A)$ 的短基型陷门|
|$\mathbf R$|$\mathbf G$-trapdoor 或短矩阵型陷门|
|$s_1(\mathbf T)$|陷门质量，可取最大奇异值|
|$\|\widetilde{\mathbf T}\|$|GSO 陷门质量|
|$\mathbf H_{\rm tag}$|Gadget/tag 矩阵，避免与哈希 $\mathsf H$ 冲突|
|$\mathsf{DelTrap}$|陷门委托|
|$\mathsf{RandBasis}$|陷门/短基随机化|
|$\epsilon_{\rm td}$|公开矩阵与均匀分布的统计距离|

## 预像采样

|符号|含义|
|-|-|
|$\mathbf u\in\mathbb Z_q^n$|目标综合|
|$\mathbf x\in\Lambda_q^{\mathbf u}(\mathbf A)$|预像|
|$s_{\rm pre}$|预像采样 Gaussian 参数|
|$D_{\Lambda_q^{\mathbf u}(\mathbf A),s_{\rm pre}}$|目标陪集离散高斯|
|$\epsilon_{\rm pre}$|实际预像分布与目标分布的距离|
|$\mathbf A_{\rm L},\mathbf A_{\rm R}$|左、右扩展矩阵|

---

