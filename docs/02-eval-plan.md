# 评测计划

原则：先 CPU/CUDA 仿真对齐形式模型，再上 Ascend 910B。每个数字标注 **来源文献** 或 **待测**。不把 2602.11287 / 2509.25149 的表格抄成「本仓库结果」。

## 0. 设备标签（强制）

每条 run 的日志必须带下列之一：

| 标签 | 含义 |
| --- | --- |
| `cpu-ref` | 主机参考实现（Algorithm 1 / TE 公式的逐元素量化）。bit-level 对照用。 |
| `cuda-sim` | GPU fake-quant：量化到 HiF4 或 NVFP4 可表示点后，用 BF16/FP32 GEMM。非 Cube/Tensor Core 真 4-bit 累加。 |
| `cuda-nvfp4-native` | Blackwell 上 TE/cuDNN 真 NVFP4 GEMM（若有硬件）。 |
| `ascend-910B-sim` | 910B 上伪量化 + 更高精度 Cube/Vector。与 2602.11287 小模型设定同类。 |
| `ascend-910B-hif4` | 910B 上真实 HiF4 存储与 64-dot（仅当驱动/CANN 暴露该 dtype 时）。 |

H2 packing 若硬件拒绝非 Algorithm-1 的 metadata，只能停在 `cpu-ref` / `cuda-sim`。

## 1. 单元：高斯 \(1024\times 1024\) MSE

复现 2602.11287 §III-A 的协议，作为量化器回归，**不是**新 SOTA 声明。

- 数据：18 张矩阵，零均值高斯，\(\sigma=0.01\times 2^{x}\)，\(x\in[0,17]\)，尺寸 \(1024\times 1024\)。
- 方法：HiF4 = Algorithm 1；NVFP4 = 每组峰收到 6 的 E4M3（direct-cast）；NVFP4+PTS = 先把 tensor 峰收到 2688 再 NVFP4；MXFP4 按 MX 文献（可选对照）。
- 指标：相对高精矩阵的 MSE；以 HiF4 为 1 的比值。文献稳定比 **1 : 1.32 : 1.89**（HiF4:NVFP4:MXFP4）。本仓库复现应报告绝对值与比值，并写设备标签。
- 额外（本项目）：同一批矩阵上跑 **H2 packing**，报告 (i) 相对原矩阵 MSE；(ii) 相对 NVFP4 反量化的 MSE；(iii) 精确命中元素比例（§4 可表示性）。

通过标准：cpu-ref 上 HiF4 vs NVFP4 的比值与 1:1.32 同方向、同量级；若偏差大，先修量化器再做 LLM。

## 2. 模型与任务（PTQ / 前向）

模型（与 2602.11287 Table III 对齐，便于对照而非抄录）：

- Llama-2-7B
- Llama-3-8B
- Qwen2.5-14B
- Mistral-7B

任务：

- 准确率：ARC-C、ARC-E、BoolQ、HellaSwag、LAMBADA（OpenAI）、PIQA、WinoGrande、MMLU。
- **Canary**：WikiText-2 word PPL。任何格式切换后先看 Wikitext-2；异常（爆炸、近随机）再决定是否跑满 8 任务。Mistral-7B + NVFP4 无 PTS 在文献中接近 crash——canary 应能提前抓住。

范围（第一轮）：linear 层 W/A（除 embedding 与 LM head），与 2602.11287 §IV-B 相同。Attention score 默认不量化，除非进入 §3 的 `+attention-score` 消融。

## 3. 消融矩阵

固定校准/prompt 模板。第一轮 PTQ 用 direct-cast（无 GPTQ），避免 HiGPTQ 与 packing 耦合。

### 3.1 量化对象

| 代号 | W | A | 说明 |
| --- | --- | --- | --- |
| `W-only` | 4-bit | 高精 | 下界；MXFP4 社区常用，不是本项目目标 |
| `W+A` | 4-bit | 4-bit | 主设定；对应文献 A-W 表 |

### 3.2 算子范围

| 代号 | 内容 |
| --- | --- |
| `linear-only` | QKV/O 与 FFN 的 GEMM。QKᵀ / PV 高精。 |
| `+attention-score` | 另把 \(QK^{\top}\)（及 backward 时 \(dS\) 相关）量化。**PV / \(dOV^{\top}\) 仍 BF16**（Full-Stack FP4 Table 2）。单独再加一行 `+PV` 仅作危险消融。 |

### 3.3 格式假设 vs 参照

| 代号 | 定义 | 设备 |
| --- | --- | --- |
| `H1` | 原生 HiF4，Algorithm 1，64-dot 语义 | cpu-ref → cuda-sim → 910B |
| `H2` | NVFP4 量化后 16-into-64 packing | cpu-ref / cuda-sim；910B 仅当 metadata 可注入 |
| `H3` | 见 00-formal-model §7；第一候选：linear=H1，score=BF16 或 H2 | 同左列子路径 |
| `NVFP4` | TE 公式，无 PTS | cuda-sim；有 Blackwell 则 native |
| `NVFP4+PTS` | peak-to-2688 再 NVFP4 | 同上 |
| `BF16` | 基线 | 任意 |

训练向（第二阶段，不阻塞 PTQ）：H1 跟 2604.08826（RHT-64、梯度 NR）；`NVFP4` 训练跟 TE extras（SR、RHT-16、2D W）。不要混用 extras 还报同一名字。

### 3.4 建议跑序

1. `cpu-ref` 高斯 MSE：H1、NVFP4、NVFP4+PTS、H2。
2. WikiText-2 canary，Llama-2-7B：`W-only`/`W+A` × `linear-only` × {H1, H2, NVFP4, NVFP4+PTS, BF16}。
3. 同一模型满 8 任务，仅 `W+A` + `linear-only`。
4. Mistral-7B canary（专门看无 PTS 的 NVFP4 是否崩、H1 是否稳）。
5. `+attention-score` 只在 canary 通过的格式上打开。
6. 其余三模型；910B 只重复已冻结的 H1 / H3 配置。

## 4. 指标与对照方式

- 任务准确率：与 BF16 的差（百分点）。文献 Table III 可作 **外部对照**，必须写「2602.11287，非本 run」。
- WikiText-2 PPL：原值 + 相对 BF16。
- 高斯 MSE：绝对值 + 相对 HiF4。
- H2：精确命中率、相对 NVFP4 的逐元素误差。
- 开销（有硬件才报）：相对 BF16 或相对 NVFP4 的 GEMM 时间、是否走 64-dot、是否仍有 PTS 预缩放。没有测量就不写数字。

不报告未跑的配置。不平均「有 crash 的 NVFP4」与正常 run，除非显式拆分（文献 Table IV 那样 w/ vs w/o Mistral）。

## 5. 实现备注（评测前最小集合）

- 量化轴：linear 沿 K（GEMM 内积维）；attention score 沿 head dim 或沿 sequence 必须在日志写明，两种不可比。
- 权重 2D：NVFP4 参照用 16×16；H1 第一轮 1×64。H1-2D 是独立条目。
- 随机性：2602.11287 对小模型用多种子、双设备平均。本仓库第一轮可单 seed，但 canary 与主表 seed 分开写。
- 禁止：在 eval 脚本里用未写进 00-formal-model 的 scale 公式。

## 6. 完成定义（建模阶段）

本阶段结束当且仅当：

1. cpu-ref 高斯 MSE 跑通并写入 lab-log。
2. H2 可表示性有计数（即使精确命中率为 0）。
3. 至少 Llama-2-7B WikiText-2 canary 在 H1 / NVFP4+PTS / BF16 上有数。
4. 上述 run 的设备标签齐全。

满模型 8 任务与 910B 属于下一阶段。
