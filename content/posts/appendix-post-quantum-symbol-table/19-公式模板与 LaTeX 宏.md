# 公式模板与 LaTeX 宏

## 常用公式模板

### 安全实验

$$
\operatorname{Adv}_{\Pi,\mathcal{A}}^{\mathsf{goal}}(\lambda)
:= \left|\Pr[\mathsf{Exp}_{\Pi,\mathcal{A}}^{\mathsf{goal}}(\lambda)=1]-c\right|.
$$

### 游戏跳转

$$
\left|\Pr[\mathsf{G}_i=1]-\Pr[\mathsf{G}_{i+1}=1]\right|
\le \epsilon_i.
$$

### 正确性

$$
\Pr[\mathsf{Dec}(\mathsf{sk},\mathsf{Enc}(\mathsf{pk},\mu))=\mu]
\ge 1-\operatorname{negl}(\lambda).
$$

### LWE 样本

$$
\mathbf{b}:=\mathbf{A}^{\top}\mathbf{s}+\mathbf{e}\pmod q.
$$

### 归约结论

$$
\operatorname{Adv}_{\Pi,\mathcal{A}}^{\mathsf{goal}}
\le
\operatorname{Adv}_{\mathcal{B}}^{P}
+\epsilon_{\rm stat}
+\epsilon_{\rm fail}.
$$

## 推荐宏

```latex
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Zq}{\mathbb{Z}_q}
\newcommand{\R}{\mathbb{R}}
\newcommand{\bits}{\{0,1\}}
\newcommand{\A}{\mathcal{A}}
\newcommand{\Bred}{\mathcal{B}}
\newcommand{\negl}{\operatorname{negl}}
\newcommand{\Adv}{\operatorname{Adv}}
\newcommand{\Sample}{\xleftarrow{\$}}
\newcommand{\pk}{\mathsf{pk}}
\newcommand{\sk}{\mathsf{sk}}
\newcommand{\ct}{\mathsf{ct}}
\newcommand{\KeyGen}{\mathsf{KeyGen}}
\newcommand{\Enc}{\mathsf{Enc}}
\newcommand{\Dec}{\mathsf{Dec}}
\newcommand{\Encaps}{\mathsf{Encaps}}
\newcommand{\Decaps}{\mathsf{Decaps}}
\newcommand{\Sign}{\mathsf{Sign}}
\newcommand{\Vrfy}{\mathsf{Vrfy}}
```

宏名称应避免覆盖 LaTeX 内置命令，也不应为只出现一次的局部符号定义全局宏。

