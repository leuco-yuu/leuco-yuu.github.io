# 公钥加密、KEM、FO、IBE、ABE 与同态加密

## 公钥加密

|符号|含义|
|-|-|
|$\Pi_{\rm PKE}=(\mathsf{KeyGen},\mathsf{Enc},\mathsf{Dec})$|公钥加密方案|
|$(\mathsf{pk},\mathsf{sk})$|公钥、私钥|
|$\mu\in\mathcal M$|明文|
|$r\in\mathcal R_{\rm enc}$|加密随机币|
|$\mathsf{ct}\in\mathcal C$|密文|
|$\mathsf{Enc}(\mathsf{pk},\mu;r)$|显式随机币加密|
|$\mathsf{Dec}(\mathsf{sk},\mathsf{ct})$|解密，输出 $\mu$ 或 $\perp$|
|$\mathsf{ValidPK}(\mathsf{pk})$|公钥合法性检查|
|$\mathsf{ValidCT}(\mathsf{ct})$|密文语法/代数合法性检查|

## KEM

|符号|含义|
|-|-|
|$\Pi_{\rm KEM}=(\mathsf{KeyGen},\mathsf{Encaps},\mathsf{Decaps})$|KEM|
|$(\mathsf{ct},\mathsf K)\leftarrow\mathsf{Encaps}(\mathsf{pk})$|封装|
|$\mathsf K'\leftarrow\mathsf{Decaps}(\mathsf{sk},\mathsf{ct})$|解封|
|$\mathsf K\in\mathcal K_{\rm ss}$|共享秘密；用无衬线体避免与数域冲突|
|$\ell_K$|共享密钥比特长度|
|$\mathsf K_{\rm bad}$|无效密文时的隐式拒绝密钥|
|$\mathsf z_{\rm rej}$|私钥中保存的隐式拒绝秘密；不得只写 $z$|
|$\mathsf{ct}^*$|挑战密文|
|$\mathsf K^*$|挑战密钥|

## Fujisaki–Okamoto 变换

|符号|含义|
|-|-|
|$\mu_{\rm FO}$|FO 内部随机消息/种子|
|$r_{\rm FO}:=\mathsf H_{\rm coins}(\mu_{\rm FO}\Vert\mathsf H_{\rm pk}(\mathsf{pk})\Vert\mathsf{ctx})$|确定性加密随机币|
|$\widehat{\mathsf{ct}}$|重加密得到的密文|
|$\mathsf{ok}:=[\widehat{\mathsf{ct}}=\mathsf{ct}]$|重加密检查结果|
|$\mathsf K_{\rm good}$|合法密文派生密钥|
|$\mathsf K_{\rm bad}$|非法密文派生密钥|
|$\mathsf{cmov}$|常数时间条件选择|
|$\epsilon_{\rm FO}$|FO 归约总损失|

## DEM 与混合加密

|符号|含义|
|-|-|
|$\Pi_{\rm DEM}$|数据封装机制|
|$\mathsf{ct}_{\rm kem}$|KEM 密文|
|$\mathsf{ct}_{\rm dem}$|对称密文|
|$\mathsf{ct}_{\rm hyb}$|混合密文|
|$\mathsf K_{\rm dem}$|DEM 密钥|
|$\mathsf{ad}$|认证关联数据|

## IBE/HIBE

|符号|含义|
|-|-|
|$\Pi_{\rm IBE}=(\mathsf{Setup},\mathsf{Extract},\mathsf{Enc},\mathsf{Dec})$|IBE|
|$(\mathsf{mpk},\mathsf{msk})$|主公钥、主私钥|
|$\mathsf{id}$|身份字符串|
|$\mathsf{sk}_{\mathsf{id}}$|身份私钥|
|$\boldsymbol{id}=(\mathsf{id}_1,\ldots,\mathsf{id}_d)$|层次身份|
|$d_{\rm id}$|身份层级深度|
|$\mathsf{Delegate}$|HIBE 委托算法|

## ABE、谓词加密与功能加密

|符号|含义|
|-|-|
|$\mathbb A$|属性全集|
|$S_{\rm attr}\subseteq\mathbb A$|属性集合|
|$\mathcal P_{\rm pol}$|访问策略|
|$(\mathbf M,\rho_{\rm LSSS})$|LSSS 矩阵与行标记映射|
|$f_{\rm pred}$|谓词|
|$f_{\rm FE}$|功能加密所允许计算的函数|
|$\mathsf{sk}_f$|功能密钥|

## 同态加密

|符号|含义|
|-|-|
|$\Pi_{\rm HE}$|同态加密方案|
|$t_{\rm pt}$|明文模数|
|$q_0<q_1<\cdots<q_L$|模数链，具体方向可按实现固定但必须全篇一致|
|$\ell_{\rm lvl}$|当前密文层级|
|$\mathsf{ct}^{(\ell)}$|第 $\ell$ 层密文|
|$\mathsf{Eval}$|同态求值|
|$\mathsf{Relin}$|重线性化|
|$\mathsf{ModSwitch}$|模数切换|
|$\mathsf{Rescale}$|CKKS 缩放|
|$\Delta_{\rm CKKS}$|CKKS 编码尺度|
|$\mathsf{rlk}$|重线性化密钥|
|$\mathsf{gk}$|Galois/旋转密钥|
|$\mathsf{evk}$|通用求值密钥|
|$B_{\rm noise}^{(\ell)}$|第 $\ell$ 层噪声预算|
|$d_{\rm mult}$|乘法深度|
|$\mathsf{Boot}$|自举|
|$\mathbf C$|GSW 矩阵密文|
|$\boxtimes$|GSW 外积运算|

---

