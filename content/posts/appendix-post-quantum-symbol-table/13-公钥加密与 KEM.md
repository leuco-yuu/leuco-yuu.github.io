# 公钥加密与 KEM

## 公钥加密

公钥加密方案统一写为

\[
\Pi_{\rm PKE}=(\mathsf{KeyGen},\mathsf{Enc},\mathsf{Dec}).
\]

|接口|输入与输出|
|-|-|
|$(\mathsf{pk},\mathsf{sk})\leftarrow\mathsf{KeyGen}(1^\lambda)$|生成密钥对|
|$\mathsf{ct}\leftarrow\mathsf{Enc}(\mathsf{pk},\mu;r)$|加密消息|
|$\mu'\leftarrow\mathsf{Dec}(\mathsf{sk},\mathsf{ct})$|解密密文|

正确性写为

\[
\Pr[\mathsf{Dec}(\mathsf{sk},\mathsf{Enc}(\mathsf{pk},\mu))=\mu]
\ge 1-\operatorname{negl}(\lambda).
\]

## 密钥封装机制

KEM 统一写为

\[
\Pi_{\rm KEM}=(\mathsf{KeyGen},\mathsf{Encaps},\mathsf{Decaps}).
\]

|接口|输入与输出|
|-|-|
|$(\mathsf{pk},\mathsf{sk})\leftarrow\mathsf{KeyGen}(1^\lambda)$|生成密钥对|
|$(\mathsf{ct},K)\leftarrow\mathsf{Encaps}(\mathsf{pk})$|封装会话密钥|
|$K'\leftarrow\mathsf{Decaps}(\mathsf{sk},\mathsf{ct})$|解封会话密钥|

## FO 类变换常用记号

|符号|含义|
|-|-|
|$m$|内部随机消息；仅在 FO 局部语境使用|
|$\mathsf{H}_{\rm coins}$|导出加密随机币|
|$\mathsf{H}_{\rm key}$|导出会话密钥|
|$z$|隐式拒绝秘密|
|$\mathsf{ct}^*$|挑战密文|
