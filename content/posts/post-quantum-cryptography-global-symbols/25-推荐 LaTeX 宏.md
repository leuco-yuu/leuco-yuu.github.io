# 推荐 LaTeX 宏

```latex
% Number sets
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\F}{\mathbb{F}}
\newcommand{\Zq}{\mathbb{Z}_q}

% Probability and information
\newcommand{\Prb}{\Pr}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Cov}{\operatorname{Cov}}
\newcommand{\Supp}{\operatorname{Supp}}
\newcommand{\SD}{\Delta}
\newcommand{\negl}{\operatorname{negl}}
\newcommand{\poly}{\operatorname{poly}}

% Linear algebra and lattices
\newcommand{\ip}[2]{\left\langle #1,#2\right\rangle}
\newcommand{\norm}[1]{\left\lVert #1\right\rVert}
\newcommand{\round}[1]{\left\lfloor #1\right\rceil}
\newcommand{\Lat}{\Lambda}
\newcommand{\dual}{^{*}}
\newcommand{\detlat}{\operatorname{det}}
\newcommand{\dist}{\operatorname{dist}}

% Sampling
\newcommand{\sample}{\xleftarrow{\$}}
\newcommand{\getsD}{\leftarrow}
\newcommand{\rhoG}{\rho}
\newcommand{\DGauss}{D}

% Algorithms and objects
\newcommand{\Setup}{\mathsf{Setup}}
\newcommand{\KeyGen}{\mathsf{KeyGen}}
\newcommand{\Enc}{\mathsf{Enc}}
\newcommand{\Dec}{\mathsf{Dec}}
\newcommand{\Encaps}{\mathsf{Encaps}}
\newcommand{\Decaps}{\mathsf{Decaps}}
\newcommand{\Sign}{\mathsf{Sign}}
\newcommand{\Verify}{\mathsf{Verify}}
\newcommand{\pk}{\mathsf{pk}}
\newcommand{\sk}{\mathsf{sk}}
\newcommand{\ct}{\mathsf{ct}}
\newcommand{\sskey}{\mathsf{K}}
\newcommand{\fail}{\perp}

% Security games
\newcommand{\Adv}{\operatorname{Adv}}
\newcommand{\Succ}{\operatorname{Succ}}
\newcommand{\Exp}{\mathsf{Exp}}
\newcommand{\Bad}{\mathsf{Bad}}
\newcommand{\GameIdx}{\mathsf{G}}

% Hashing and domain separation
\newcommand{\Hash}{\mathsf{H}}
\newcommand{\XOF}{\mathsf{XOF}}
\newcommand{\KDF}{\mathsf{KDF}}
\newcommand{\ctx}{\mathsf{ctx}}
\newcommand{\dom}{\mathsf{dom}}

% Proof systems
\newcommand{\Rel}{\mathcal{R}}
\newcommand{\Prover}{\mathcal{P}}
\newcommand{\Verifier}{\mathcal{V}}
\newcommand{\Simulator}{\mathcal{S}}
\newcommand{\Extractor}{\mathcal{E}}
\newcommand{\crs}{\mathsf{crs}}
\newcommand{\zkproof}{\pi_{\rm zk}}
```

> 宏只负责排版，不应隐藏数学含义。例如不要定义同时依赖上下文改变语义的 `\newcommand{\s}{...}`，也不要把不同方案的压缩函数全部映射为同一个无参数宏。

---

