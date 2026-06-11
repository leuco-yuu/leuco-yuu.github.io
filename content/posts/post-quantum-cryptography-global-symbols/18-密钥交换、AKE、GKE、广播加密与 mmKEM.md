# 密钥交换、AKE、GKE、广播加密与 mmKEM

## 参与方、身份与会话

|符号|含义|
|-|-|
|$\mathcal U=\{P_1,\ldots,P_N\}$|参与方集合|
|$P_i$|第 $i$ 个参与方|
|$\mathsf{id}_i$|参与方身份|
|$\mathsf{Sess}_{i,s}$|$P_i$ 的第 $s$ 个本地会话实例|
|$\mathsf{sid}$|会话标识|
|$\mathsf{gid}$|群组标识|
|$\mathsf{pid}$|伙伴身份集合或伙伴标识|
|$\mathsf{role}$|initiator/responder/member/coordinator 等角色|
|$\mathsf{epoch}$|群组状态轮次|
|$\mathsf{tr}$|协议 transcript|
|$\mathsf{trHash}$|transcript 哈希|

## 长期密钥、临时密钥与会话密钥

|符号|含义|
|-|-|
|$(\mathsf{lpk}_i,\mathsf{lsk}_i)$|长期公私钥|
|$(\mathsf{epk}_{i,s},\mathsf{esk}_{i,s})$|第 $s$ 会话的临时密钥对|
|$\mathsf K_{\rm sess}$|两方会话密钥|
|$\mathsf K_{\rm grp}$|群组会话密钥|
|$\mathsf K_{\rm conf}$|密钥确认密钥|
|$\mathsf K_{\rm exp}$|导出器密钥|
|$\mathsf K_{\rm app}$|应用流量密钥|
|$\mathsf{KC}$|密钥确认标签|

## AKE 查询与属性

|符号|含义|
|-|-|
|$\mathsf{Send}(i,s,M)$|向会话实例发送消息|
|$\mathsf{Reveal}(i,s)$|泄露会话密钥|
|$\mathsf{Corrupt}(i)$|泄露长期秘密|
|$\mathsf{EphemeralReveal}(i,s)$|泄露临时秘密|
|$\mathsf{StateReveal}(i,s)$|泄露内部状态|
|$\mathsf{Test}(i,s)$|测试查询|
|$\mathsf{Partner}(i,s;j,t)$|伙伴关系谓词|
|$\mathsf{Fresh}(i,s)$|新鲜性谓词|
|$\mathsf{KCI}$|密钥泄漏冒充安全|
|$\mathsf{FS}$|前向保密|
|$\mathsf{PCS}$|后妥协安全|

## 群组密钥协商

|符号|含义|
|-|-|
|$\mathcal U_{\rm grp}$|当前群组成员集合|
|$N_{\rm grp}:=\lvert\mathcal U_{\rm grp}\rvert$|群组规模|
|$\mathsf{state}^{(e)}$|epoch $e$ 的群组状态|
|$\mathsf{tree}^{(e)}$|群组状态树|
|$\mathsf{commit}^{(e)}$|群组状态更新消息|
|$\mathsf{welcome}^{(e)}$|新成员欢迎消息|
|$\mathsf{Add},\mathsf{Update},\mathsf{Remove}$|动态成员操作|
|$\mathsf{Agree}$|所有诚实成员输出相同密钥事件|
|$\mathsf{Contrib}$|贡献性事件/性质|

## 多接收者与多密钥 KEM

|符号|含义|
|-|-|
|$\mathcal U_{\rm rec}=\{\mathsf{id}_1,\ldots,\mathsf{id}_N\}$|接收者集合；不用 $\mathcal R$，避免与关系冲突|
|$\mathsf{PKSet}:=(\mathsf{pk}_1,\ldots,\mathsf{pk}_N)$|接收者公钥集合|
|$\mathsf{hdr}$|公共共享头|
|$\mathsf{ct}_i$|第 $i$ 个接收者的局部分量|
|$\mathsf{CT}:=(\mathsf{hdr},\{(\mathsf{id}_i,\mathsf{ct}_i)\}_{i=1}^N)$|完整多接收者密文|
|$\mathsf K_i$|第 $i$ 个接收者的封装密钥|
|$\mathsf K_{\rm common}$|多接收者共享同一密钥时的公共密钥|
|$\mathsf{bind}_i$|接收者绑定标签|
|$\mathsf{tag}_{\rm set}$|接收者集合绑定标签|
|$\mathsf{tag}_{\rm hdr}$|公共头绑定标签|
|$\mathsf{Decaps}_i$|第 $i$ 个接收者的解封算法|
|$\mathsf{ValidHdr}_i$|接收者 $i$ 对共享头的局部验证|
|$\mathcal C_{\rm corr}$|已腐化接收者集合|
|$\mathcal T_{\rm tgt}$|目标接收者集合|
|$L_{\rm hdr}$|公共头长度|
|$L_{{\rm loc},i}$|第 $i$ 个局部分量长度|
|$L_{\rm total}$|总密文长度|
|$\operatorname{BW}_{\rm amort}$|摊销到每个接收者的带宽|

## mmKEM CCA2 查询边界

|符号|含义|
|-|-|
|$\mathsf{CT}^*$|挑战多接收者密文|
|$\mathsf{hdr}^*$|挑战共享头|
|$\mathsf{ct}_i^*$|目标接收者挑战局部分量|
|$\mathsf{DecapOracle}(i,\mathsf{CT})$|面向接收者 $i$ 的解封预言机|
|$\mathsf{Forbidden}(i,\mathsf{CT};\mathsf{CT}^*)$|禁止查询谓词|
|$\mathsf{Mix}(\mathsf{CT}_1,\mathsf{CT}_2)$|混合头/局部分量操作|
|$\mathsf{Portable}(i\to j)$|跨接收者可移植事件|
|$\epsilon_{\rm bind}$|接收者/头绑定失败概率|
|$\epsilon_{\rm corr}$|部分腐化下的安全损失|

## 广播加密与撤销

|符号|含义|
|-|-|
|$\mathcal U_{\rm auth}$|授权接收者集合|
|$\mathcal U_{\rm rev}$|撤销集合|
|$\mathsf{BE.Enc}(\mathcal U_{\rm auth},\mu)$|广播加密|
|$\mathsf{BE.Dec}_i$|第 $i$ 个接收者解密|
|$\mathsf{Anon}$|接收者匿名性实验|
|$\mathsf{Trace}$|叛徒追踪算法；与侧信道 trace 需加上下文下标|

---

