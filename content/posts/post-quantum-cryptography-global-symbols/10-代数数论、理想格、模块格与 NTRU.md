# 代数数论、理想格、模块格与 NTRU

## 数域与嵌入

|符号|含义|
|-|-|
|$\mathbb K$|数域；避免与共享密钥 $\mathsf K$ 冲突|
|$[\mathbb K:\mathbb Q]=n$|数域次数|
|$\mathcal O_{\mathbb K}$|数域整数环|
|$r_1,r_2$|实嵌入数与共轭复嵌入对数，$n=r_1+2r_2$|
|$\sigma_i$|第 $i$ 个域嵌入|
|$\iota_{\mathbb K}$|Minkowski/典范嵌入|
|$\operatorname{Tr}_{\mathbb K/\mathbb Q}(a)$|域迹|
|$N_{\mathbb K/\mathbb Q}(a)$|域范数|
|$\Delta_{\mathbb K}$|数域判别式|
|$\mathfrak D_{\mathbb K}$|不同理想|
|$\mathfrak D_{\mathbb K}^{-1}$|余不同理想|
|$\mathfrak a^\vee$|关于迹配对的对偶理想|

## 理想与素理想

|符号|含义|
|-|-|
|$\mathfrak a,\mathfrak b$|整理想或分式理想|
|$\mathfrak p$|素理想|
|$N(\mathfrak a)$|理想范数 $\lvert\mathcal O_{\mathbb K}/\mathfrak a\rvert$|
|$e(\mathfrak p/p)$|分歧指数|
|$f(\mathfrak p/p)$|剩余次数|
|$\operatorname{Cl}(\mathbb K)$|理想类群|

## 分圆结构

|符号|含义|
|-|-|
|$M$|分圆指数/导数|
|$\zeta_M$|原始 $M$ 次单位根|
|$\Phi_M(X)$|第 $M$ 个分圆多项式|
|$\mathbb Q(\zeta_M)$|分圆域|
|$n=\varphi(M)$|分圆域次数|
|$R=\mathbb Z[X]/(\Phi_M(X))$|分圆整数环表示|
|$R_q$|模 $q$ 的分圆商环|

## 模块格

|符号|含义|
|-|-|
|$R_q^k$|秩为 $k$ 的 $R_q$-模块|
|$\mathbf a\in R_q^k$|环元素向量|
|$\mathbf A\in R_q^{k\times\ell}$|模块矩阵|
|$\|\mathbf a\|_{p,\rm coeff}$|所有系数拼接后的 $\ell_p$ 范数|
|$\|\mathbf a\|_{\rm can}$|逐分量典范嵌入后的欧氏范数|
|$\operatorname{rank}_R(M)$|$R$-模秩|

## NTRU 结构

|符号|含义|
|-|-|
|$f,g$|NTRU 短秘密多项式|
|$F,G$|NTRU 辅助多项式|
|$fG-gF=q$|统一采用的 NTRU 方程方向|
|$h_{\rm NTRU}=g/f\bmod q$|NTRU 公钥比值；歧义时不得只写 $h$|
|$\mathbf B_{\rm NTRU}$|NTRU 格基|
|$\Lambda_{\rm NTRU}(h,q)$|NTRU 格|
|$\mathsf{NTRU\text{-}SIS}$|NTRU-SIS 问题|

---

