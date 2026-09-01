# 形式模型：把 NVFP4 recipe 映射到 HiF4 encoding

本文只陈述文献与公开文档中的方程、网格与约束，并由此导出可检验的假设。未测数字一律标「待测」。来源见 [`01-papers.md`](01-papers.md)。

## 1. NVFP4

### 1.1 元素与解码

NVFP4 元素为 E2M1（1 sign + 2 exponent + 1 mantissa），幅度上界 \(\pm 6\)。正值网格：

\[
\{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\}
\]

（及对应负值）。层次解码（Transformer Engine NVFP4 文档；arXiv:2509.25149）：

\[
x = x_{\mathrm{E2M1}} \cdot s_{\mathrm{block}} \cdot s_{\mathrm{global}}.
\]

- \(s_{\mathrm{block}}\)：每 16 个连续元素共享一个 FP8 E4M3。
- \(s_{\mathrm{global}}\)：整个 tensor 一个 FP32。
- 激活 / 梯度默认 **1×16**；权重默认 **16×16 2D**（可用 `disable_2d_quantization=True` 改回 1D）。2D 动机是让 rowwise 与 columnwise 量化在数值上等价。

TE 给出的 scale 计算：

\[
s_{\mathrm{global}} = \frac{\mathrm{global\_amax}}{448 \cdot 6},\qquad
s_{\mathrm{block}} = \frac{\mathrm{block\_amax}/6}{s_{\mathrm{global}}}.
\]

其中 \(448\) 是 E4M3 最大有限幅度，\(6\) 是 E2M1 最大幅度。因此 \(448 \times 6 = 2688\)：HiF4 论文把 NVFP4+PTS 描述为先把 tensor 峰值缩放到 2688 再量化（arXiv:2602.11287 §III-A，引用 NVIDIA Introducing NVFP4 blog）。

arXiv:2509.25149 Appendix B 用 encode/decode 对偶写法，与 TE 文档等价：先用全局 encode scale \(s_{\mathrm{enc}} = (6\cdot 448)/\mathrm{amax}_x\) 把 tensor 送进「E2M1×E4M3」盒子，再令块 decode \(s_{\mathrm{dec},b}=\mathrm{amax}_b/6\)，量化为 E4M3 后取倒数得到块 encode。

存储：每 16 值一个 E4M3 scale → **4.5 bits/value**，外加每 tensor 一个 FP32。

### 1.2 动态范围

- Intra-group（E2M1 自身）：\(\log_2(6/0.5)=3.58\) binades。MXFP4 的 UE8M0 上取整可能把可用范围压到 \(\log_2(3/0.5)=2.58\)（arXiv:2509.25149 Appendix B.4）；NVFP4 用 E4M3 避免这一档损失。
- Inter-group（E4M3×E2M1，HiF4 论文 Table II）：global \([-10,11]\) 约 **22 binades**；最大正值 \(2^{11}\times 1.3125\)，最小正值 \(2^{-10}\)。22 binades 不足以覆盖许多 LLM 张量，故 **PTS 经常是必须的**（推理 blog / TE 的 \(s_{\mathrm{global}}\)；HiF4 论文对 NVFP4 提到 peak-to-2688）。

### 1.3 TE extras（训练配方，不是格式本身）

来源：TE NVFP4 文档；arXiv:2509.25149 §4。

| 组件 | 行为 |
| --- | --- |
| Stochastic rounding (SR) | 仅梯度；无偏量化。可用 `disable_stochastic_rounding=True` 关掉。 |
| RHT | \(d=16\)，作用在 **wgrad** 的 columnwise 输入（激活与梯度）；\(x'=xH\)，\(H=d^{-1/2} S_d H_d\)。权重不做 RHT，以免破坏前/反向同一量化。 |
| 2D weight scale | 16×16 一块，复制成 1×16 再进 Tensor Core。 |
| Layout | NVFP4 GEMM **只支持 TN**。Columnwise 存转置。Scale 需 **swizzle**（块大小 16，E4M3）。 |
| Mixed precision | 部分敏感 linear 留 BF16；embedding / LM head / norm / softmax / QK 与 PV batched GEMM 留更高精度。 |

这些 extras 是「NVFP4 级精度」的一部分：适配 HiF4 时必须声明是否一并移植。

## 2. HiF4

### 2.1 单元与解码

一个 HiF4 unit = 32-bit 共享 metadata + 64 个 4-bit 元素 → **4.5 bits/value**（arXiv:2602.11287 §II-A）。

- **E6M2**：无符号 8-bit。6-bit exponent，bias 48；2-bit mantissa + hidden 1。只支持 normal。NaN = `111111_11`。无 Inf、无 0。解释：\(X=2^{E}\times 1.M\)。最大 \(2^{15}\times 1.50\)（`111111_10`），最小 \(2^{-48}\)（`000000_00`）。
- **E1_8**：8 bit，每位覆盖 8 个元素（level-2 micro-exponent）。
- **E1_16**：16 bit，每位覆盖 4 个元素（level-3）。
- **S1P2** ≡ E1M2：最大 \(\pm 1.75\)，最小非零 \(\pm 0.25\)。零：\(S\,0.00_2\)。

解码（论文 Eq. 2），\(i\in[1,64]\)：

- 若 \(E6M2=\mathrm{NaN}\)，则全体 \(V_i=\mathrm{NaN}\)。
- 否则

\[
V_i = \mathrm{E6M2}\cdot 2^{\bigl(\{E1\_8\}_{\lceil i/8\rceil}+\{E1\_16\}_{\lceil i/4\rceil}\bigr)}\cdot \{S1P2\}_i.
\]

Intra-group：最大 \(2^{1+1}\times 1.75=7\)，最小正 \(2^{0+0}\times 0.25=0.25\)，\(\log_2(7/0.25)=\) **4.81 binades**。Global（Table II）：\([-50,18]\) 约 **69 binades**；最大正 \(2^{18}\times 1.3125\)，最小正 \(2^{-50}\)。

### 2.2 转换（BF16 → HiF4，Algorithm 1）

三层 amax 树：4 → 8 → 64。

1. 16 个局部峰：\(V16[i]=\max|V64[4i-3:4i]|\)。
2. 8 个局部峰：\(V8[i]=\max(V16[2i-1:2i])\)。
3. 全局峰：\(V_{\max}=\max(V8)\)。
4. \(SF=V_{\max}/7\)（实现为乘 \((1/7)_{\mathrm{BF16}}\)），再 `BF16_to_E6M2`。
5. \(E1\_8=(V8\cdot E6M2^{-1}\ge 4)\,?\,1:0\)。
6. \(E1\_16[i]=(V16[i]\cdot E6M2^{-1}\cdot 2^{-E1\_8[\lceil i/2\rceil]}\ge 2)\,?\,1:0\)。
7. 元素：\(V64[i]\cdot E6M2^{-1}\cdot 2^{-E1\_8[\lceil i/8\rceil]}\cdot 2^{-E1\_16[\lceil i/4\rceil]}\)，再 `BF16_to_S1P2`；越界 clamp 保号。

舍入：round-half-to-even 或 round-half-away-from-zero。E6M2 无 subnormal，倒数可用 4-entry LUT。

### 2.3 64-dot

一对 HiF4 vs 四对 NVFP4 才能填满 64-length PE（Cube / Tensor Core 相对 8-bit 加倍吞吐的宽度）。论文 Eq. 3：

\[
\mathrm{Dot}(A,B)=\mathrm{E6M2}^{(A)}\mathrm{E6M2}^{(B)}
\sum_{i=1}^{8} 2^{E1\_8^{(A)}[i]+E1\_8^{(B)}[i]}
\sum_{j=2i-1}^{2i} 2^{E1\_16^{(A)}[j]+E1\_16^{(B)}[j]}
\sum_{k=4j-3}^{4j} S1P2^{(A)}[k]\,S1P2^{(B)}[k].
\]

实现上可把 E1_16 吸进 S1P2，乘数变成 5-bit 整数（S2P2）；NVFP4 的 E2M1 变成 S3P1。HiF4 在 64→1 压缩中几乎全程整数，末级只要 1 个小 FP 乘 + 1 个大整数乘；NVFP4 先压到 4 个部分和，再 4 个小 FP 乘 + 4 个大整数乘，最后 FP 累加。论文：**增量面积约 NVFP4 的 1/3，功耗约低 10%**。Ascend Cube PE 256-bit 输入带宽对齐 64×FP4（arXiv:2604.08826 §4）。

### 2.4 量化误差（文献，非本仓库测量）

18 张 \(1024\times 1024\) 零均值高斯，\(\sigma=0.01\times 2^{x}\)，\(x\in[0,17]\)。排除 NVFP4 近边界 overflow/underflow 后，MSE 比稳定为

\[
\mathrm{HiF4}:\mathrm{NVFP4}:\mathrm{MXFP4}=1:1.32:1.89
\]

（相对 HiF4 归一化；约合相对 NVFP4 降 24% MSE）。NVFP4 无 PTS 时在数值边界误差显著上升；PTS 可消掉这一项，但转换有软件开销。

## 3. 网格对照

| | HiF4 | NVFP4 |
| --- | --- | --- |
| bits/value | 4.5 | 4.5 + 每 tensor 一个 FP32 |
| group | 64 | 16（权重默认可 16×16） |
| 4-bit 元素 | S1P2 (E1M2)，3-bit significand | E2M1，2-bit significand |
| 正元素网格 | \(\{0,0.25,0.50,0.75,1.00,1.25,1.50,1.75\}\) | \(\{0,0.5,1,1.5,2,3,4,6\}\) |
| 组 scale | E6M2 + E1_8 + E1_16（2 的幂 micro-exp） | E4M3（含分数）+ FP32 全局 |
| intra-group | 4.81 binades，上界 7 | 3.58 binades，上界 6 |
| global | \([-50,18]\) ~69 binades | \([-10,11]\) ~22 binades |
| PTS | 格式本身不需要 | 常需要（peak-to-2688） |
| 64-dot 输入 | 1 对 unit | 4 对 unit |
| 特殊值 | NaN 与 \(\pm 0\) | NaN 与 \(\pm 0\) |

S1P2 在 micro-exponent \(\{0,1\}\times\{0,1\}\) 下的 **正 intra-group 可达集**（共享同一 \((E1\_8,E1\_16)\) 的 4 元组）：

\[
\{0.25,0.5,0.75,1,1.25,1.5,1.75\}\ \cup\
\{0.5,1,1.5,2,2.5,3,3.5\}\ \cup\
\{1,2,3,4,5,6,7\}.
\]

E2M1 正网格 \(\{0.5,1,1.5,2,3,4,6\}\) 是该并集的真子集。因此 **单个元素** 的 NVFP4 值在合适的组 scale 下可以落到 HiF4 网格上；障碍在 **共享约束** 与 **E4M3 的非 2 幂**。

## 4. 16-into-64 精确可表示性

设沿量化轴把 64 个元素分成四个相邻 NVFP4 块 \(B_0,\dots,B_3\)，每块 16 元、各有 E4M3 scale \(s_t\)，并共享同一个 \(s_{\mathrm{global}}\)。H2 问：是否存在一个 HiF4 unit（一个 E6M2、8 个 E1_8、16 个 E1_16、64 个 S1P2）使

\[
\forall t\in\{0,1,2,3\},\ \forall k\in B_t:\quad
x_{\mathrm{E2M1},k}\,s_t\,s_{\mathrm{global}}
=
\mathrm{E6M2}\cdot 2^{e8(\lceil i/8\rceil)+e16(\lceil i/4\rceil)}\cdot \{S1P2\}_i
\]

对所有合法 NVFP4 块精确成立（允许 \(s_{\mathrm{global}}\) 折进 E6M2）。

### 4.1 共享拓扑

HiF4 约束：

- 同一 E1_16 位覆盖 **4** 个连续元素 → 这 4 个必须共用 \(2^{e8+e16}\)。
- 同一 E1_8 位覆盖 **8** 个连续元素。
- 四个 NVFP4 块跨 64 元，但 **一个** E6M2 覆盖全部 64。

因此：块内 16 个 NVFP4 元素共用一个浮点 \(s_t\)；映射到 HiF4 后，这 16 个至多被切成 4 个 4-元组（E1_16）或 2 个 8-元组（E1_8）。**不能**给 16 个元素各自独立的浮点 scale。

### 4.2 元素网格嵌入

固定某个 4-元组的 \((e8,e16)\)。S1P2×\(2^{e8+e16}\) 的正网格见 §3。E2M1 正值都能在 **某个** \((e8,e16)\) 下出现（例如 \(6=1.5\times 2^{2}\)，\(0.5=0.25\times 2^{1}\) 或 \(0.5\times 2^{0}\)）。但同一 4-元组必须共用一个 \((e8,e16)\)：

- 若 4 元组里同时需要「小 significand 档」（如 0.5）和「大 significand 档」（如 6），单一 \(2^{e8+e16}\) 可能无法同时命中两者。
- E2M1 的 4 与 6 只能落在 \(2^{2}\) 档（\(4=1.00\times 4\)，\(6=1.50\times 4\)）；0.5 不能落在 \(2^{2}\) 档。故 **同一 4-元组若同时含 0.5 与 6（相对同一 \(s_t\)）则不可精确嵌入**。

这是 H2 的第一类反例候选；是否在真实量化后频繁出现是待测问题。

### 4.3 E4M3 不是 2 的幂

NVFP4 的 \(s_t\) 是 E4M3，有 3-bit mantissa，一般 **不是** 2 的幂。HiF4 的 intra-group 调节只有 \(2^{e8+e16}\in\{1,2,4\}\)。四个块的相对 scale 比 \(s_t/s_{t'}\) 必须由 E1 位差吸收，或被单个 E6M2 统一（那要求 \(s_0=s_1=s_2=s_3\)）。

因此：

- 若四块 E4M3 **碰巧** 只相差 0/1/2 档 2 的幂，且块内 E2M1 不触发 §4.2 冲突，则 64 元组可精确编码（E6M2 吸收 \(s_{\mathrm{global}}\) 与公共尾数）。
- 一般 E4M3 尾数差 **不能** 被 E1 位表示 → H2 只能近似（把四个 \(s_t\) 量化进「E6M2 + 每 8/4 元 1-bit」）。近似误差待测，且依赖于块对齐。

### 4.4 2D 16×16 权重

NVFP4 权重默认 16×16 共享一个 E4M3。打进 HiF4 的 64 元向量（沿 K 或沿 16×4 切片）时，一块 16×16 会跨越多个 HiF4 unit 或多个 E1 分区。H2 在 2D 权重上比 1×16 更难做到精确嵌入。H1 则按 Algorithm 1 沿选定轴做 64 分组，**不**保留 16×16 相等约束。前/反向权重表示是否一致（NVFP4 用 2D 的原因）是 H1 的独立风险。

### 4.5 打包算法（H2 的构造性目标，非已证算法）

尚未实现。方向：

1. 按 NVFP4 规则量化得到 \(\{x_{\mathrm{E2M1}}, s_t, s_{\mathrm{global}}\}\)。
2. 令目标实值为 \(x_k=x_{\mathrm{E2M1},k}s_t s_{\mathrm{global}}\)。
3. 在 HiF4 可行集合上最小化 \(\|V-x\|_2\)（或逐块匹配 \(s_t\)）：先选 E6M2 覆盖 64 元 amax/7，再按 Algorithm 1 的阈值 4 与 2 设 E1，最后把 \(x_k/\mathrm{scale}_i\) 投到 S1P2。
4. 记录：(a) 精确命中率；(b) 相对 NVFP4 反量化的 MSE；(c) 与 H1 直接转换的 MSE。

Cube 64-dot 只吃一对 HiF4 metadata：H2 若在软件里「假装」四套 E4M3，硬件仍按一套 E6M2+E1 做 Eq. 3。仿真必须区分 **存储仿真** 与 **compute 仿真**。

## 5. Linear GEMM 误差

记 \(Y=XW^{\top}\)，\(X\in\mathbb{R}^{M\times K}\)，\(W\in\mathbb{R}^{N\times K}\)。量化 GEMM 为 \(\hat Y=Q(X)Q(W)^{\top}\)（硬件上沿 K 按 group 做 scaled integer dot，再乘组 scale）。

### 5.1 块对齐

- NVFP4：K 上每 16 个一次 E4M3×E2M1 部分积，再 FP 累加四个 16-dot 才到 64。
- HiF4：K 上每 64 个一次 Eq. 3。部分积的 scale 粒度更粗。

误差来源拆开（定性，无新数字）：

1. **元素舍入**：E2M1 2-bit significand vs S1P2 3-bit significand。高斯实验支持 HiF4 单点 MSE 更低。
2. **组 scale 量化**：E4M3（含分数）vs E6M2 + 1-bit micro-exp。E4M3 对「把块峰精确放到 6」更有利；E6M2 把块峰放到 7，但 intra 调节只有 2 的幂。
3. **组大小**：16 vs 64。更大组被单个 outlier 主导的风险更高；HiF4 用 E1_8/E1_16 缓解，不能等同于「裸 BFP-64」。
4. **累加**：NVFP4 更早回到浮点部分和；HiF4 更长整数树。两者最终都进较高精度累加，但中间动态范围与舍入不同。

### 5.2 与 PTS / RHT / SR 的耦合

- PTS（NVFP4）：改变 \(s_{\mathrm{global}}\)，不改变块内 E2M1 网格。HiF4 69 binades 通常使 PTS 变得不必要（Mistral-7B 上 NVFP4 无 PTS 会 crash，HiF4 不会；arXiv:2602.11287 Table III）。适配时「NVFP4+PTS」是精度参照，不是 HiF4 必须复现的步骤。
- RHT：NVFP4 训练用 \(d=16\) 对齐块大小；HiF4 预训练用 \(k=64\) 对齐块大小（arXiv:2604.08826）。适配 recipe 时应 **d 与 group 对齐**，否则旋转后的 scale 轴与硬件 group 错位。
- SR：NVFP4 配方用于梯度；HiF4 预训练消融里 SR 反而伤害 HiF4，RHT 单独有帮助（arXiv:2604.08826 Table 4）。H1 默认 **不要** 把 TE 的 SR 原样搬过来；H2 若目标是比特级仿真 NVFP4，则应保留 SR。

### 5.3 前/反向一致性

NVFP4 用 16×16 2D 权重避免 \(W_{\mathrm{fprop}}\neq W_{\mathrm{bprop}}\)。HiF4 若沿不同轴做 64 分组，同样会打破 chain rule。形式选项：

- 沿 K 的 1×64（类似 NVFP4 1D）；
- 8×8 或其它能被 64 整除的 2D tiling（文献未给出 HiF4 2D 配方，待设计）；
- 只量化正向，反向用更高精度（偏离「全 HiF4 Cube」目标）。

## 6. Attention 缺口

### 6.1 算子拆分

标准 attention：\(S=QK^{\top}/\sqrt{d}\)，\(P=\mathrm{softmax}(S)\)，\(O=PV\)。Backward：\(dV=P^{\top}dO\)，\(dP=dO V^{\top}\)，\(dS=P\odot(dP-\mathrm{rowsum}(dO\odot O))\)，\(dQ=dS K/\sqrt{d}\)，\(dK=dS^{\top} Q/\sqrt{d}\)。

两类 4-bit 对象：

| 路径 | 角色 | 文献做法 |
| --- | --- | --- |
| Q/K/V/O **projection linears** | 普通 GEMM | NVFP4 预训练主战场（arXiv:2509.25149）；HiF4 PTQ 也只转 linear（除 embedding 与 LM head）（arXiv:2602.11287 §IV-B） |
| **Score 路径** \(QK^{\top}\)、\(PV\) 等 | 行随机、softmax 敏感 | NVFP4 原配方显式留 BF16；Full-Stack FP4（arXiv:2607.04422）把 \(QK^{\top}\) 与 \(dS\) 相关 matmul 放 NVFP4，**\(PV\)、\(P^{\top}dO\)、\(dOV^{\top}\) 留 BF16** |

因此「linear + attention」必须拆开写：投影层的 NVFP4/HiF4 对标，与 score 路径的对标是两件事情。

### 6.2 为何 PV 更脆（Full-Stack FP4 Appendix C，转述）

\(P\) 行和为 1 且稀疏。块量化由块内最大值定 scale，小概率被粗量化或变 0，破坏凸组合：\(\|\Delta O_i\|\le\|\Delta P_i\|_1\max_j\|V_j\|\)。\(dOV^{\top}\) 进入 \(dS_{ij}=P_{ij}(dP_{ij}-D_i)\)；若 \(dP\) 被量化而 \(D_i\) 仍高精度，误差被 \(P_{ij}\) 乘到 \(dS\) 上，且 \(|dP|\) 的排序不必与 \(P\) 一致。这是把 PV 留在 BF16 的理由，不是本仓库的新证明。

### 6.3 对 HiF4 的含义

- H1 若把 \(QK^{\top}\) 也做成 HiF4 64-dot：softmax 前的 score 动态范围与线性层不同；69 binades 有助于避免 PTS，但 64 元块可能跨越多个 attention 位置/head 维，outlier 结构与线性层不同。待测。
- H2 在 score 上仿真 NVFP4 1×16：更接近 Full-Stack 的 \(QK^{\top}\) 设定，但 Cube 仍按 64-dot 吃 HiF4。需要「NVFP4 语义、HiF4 执行」的打包（§4）。
- 任何假设下，**第一轮不量化 PV**（与 Full-Stack 一致），除非消融明确打开 `+attention-score` 且单独报告 PV。

## 7. 假设

### H1 — 原生 HiF4

按 Algorithm 1 转换 W 与/或 A；linear GEMM 走 Eq. 3 的 64-dot。不模拟 E4M3 块。训练侧优先 RHT \(k=64\)、梯度 NR（跟随 arXiv:2604.08826），而不是 TE 的 SR+RHT-\(d{=}16\)。精度参照：同一套模型上的 NVFP4+PTS（及 TE extras 的 NVFP4 训练配方，若做训练）。预期：高斯与 PTQ 上不弱于 NVFP4+PTS（已有 Table III 支持 inference PTQ）；开销走单对 64-dot。风险：64 分组与 16×16 2D 权重语义不一致；attention score 未在 HiF4 论文中评测。

### H2 — NVFP4-emulation packing

先按 NVFP4 recipe（1×16 / 16×16、E4M3、可选 PTS/RHT/SR）得到 4-bit 值，再把四个 16-block 打进一个 64-group 的 HiF4 位布局，使 Cube 只看到合法 HiF4 unit。目标是 **NVFP4 数值 ≈ HiF4 存储/计算**。§4 表明一般不能精确嵌入。H2 成功标准：在 eval 上与 NVFP4+PTS 对齐（差距待测），同时仍跑 64-dot。失败模式：E4M3 尾数无法用 E1 表示；4-元组 significand 冲突；2D 权重跨 unit。

### H3 — 混合

按模块选 H1/H2/高精度，而不是单一格式。候选分割（待消融，不是已选方案）：

- Linear 用 H1（吃满 Cube、PTS-free），attention score 用 H2 或 BF16（对齐 Full-Stack：QKᵀ 低比特、PV 高比特）。
- 权重 H1（可学、对粗 scale 更容忍），激活 H2（更细的 16 元动态范围）。
- 仅 W 量化走 H1；W+A 时激活走 H2 或 BF16。
- 训练：Fprop/Dgrad 用 H1，Wgrad 用带 RHT-\(d{=}64\) 的 H1（arXiv:2604.08826），不强制 SR。

H3 的评价轴是 **精度–开销 Pareto**：每条路径是否还在 Cube 64-dot 上、是否引入 BF16 GEMM、是否需要 PTS 式的预缩放。

## 8. 开放问题

1. H2 精确嵌入的充要条件：四块 E4M3 比、块内 E2M1 的 (min,max) 是否落在同一 \((e8,e16)\) 4-元组网格。需要枚举，不是实验数字。
2. 沿 K 的 64 分组 vs NVFP4 16 分组，在真实 LLM 激活（非高斯）上的 MSE 差。高斯 1:1.32 不能外推到 Mistral 式宽分布。
3. HiF4 的 2D 权重 tiling：何种 \(r\times c\)（\(rc=64\)）能恢复前/反向同一 \(Q(W)\)。
4. Attention：只量化投影 vs 再量化 \(QK^{\top}\) vs 再量化 \(PV\)。Full-Stack 已表明后两者在 NVFP4 上不等价；HiF4 上未知。
5. TE extras 哪些必须随 H2 移植（SR、RHT \(d=16\)、swizzle、TN）。H1 预训练证据指向「RHT-64 + NR，不要 SR」。
6. 仿真栈：CPU 参考、CUDA fake-quant、Ascend 910B 真格式是否 bit-exact。910B 上 HiF4 是否已是硬件 dtype 还是仍 sim，需在 eval 日志里标明。
7. Packing 与 Cube 的 metadata 路径：若硬件强制按 Algorithm 1 从 BF16 转 HiF4，H2 的「先 NVFP4 再重编码」可能无法下到 910B，只能 CUDA/CPU sim。

评测如何回答上述问题见 [`02-eval-plan.md`](02-eval-plan.md)。
