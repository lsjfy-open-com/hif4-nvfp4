# 实验日志

时区：Asia/Shanghai（CST, UTC+8）。条目按日期追加，不改写历史。

## 2026-09-01（CST）

- **阶段**：建库。远程 `lsjfy-open-com/hif4-nvfp4`（private）。无实验数字。
- **产物**：
  - `README.md`：一页项目说明。
  - `docs/00-formal-model.md`：NVFP4 / HiF4 方程、网格、16-into-64 可表示性、linear GEMM、attention 缺口、H1/H2/H3、开放问题。
  - `docs/01-papers.md`：TE 文档、NVFP4 blog、arXiv:2509.25149、2607.04422、2602.11287、2604.08826、HiFloat4 GitHub。
  - `docs/02-eval-plan.md`：高斯 1024×1024 MSE；Llama-2-7B / Llama-3-8B / Qwen2.5-14B / Mistral-7B；ARC / BoolQ / HellaSwag / LAMBADA / PIQA / WinoGrande / MMLU；WikiText-2 PPL canary；W-only vs W+A；linear-only vs +attention-score；H1/H2/H3 vs NVFP4+PTS；设备标签 cpu-ref / cuda-sim / 910B。
- **形式化结论（待实验）**：
  - NVFP4：`x = E2M1 * E4M3_block16 * FP32_global`，`s_global = amax / (448*6)`，local 3.58 binades，E4M3 盒约 22 binades → 常要 PTS（2688）。
  - HiF4：`V = E6M2 * 2^(E1_8+E1_16) * S1P2`（E6M2=NaN 则全 NaN），block-64，local 4.81、global ~69 binades，PTS 非必须。
  - E2M1 正网格是 HiF4 intra-group 可达集的真子集，但 E4M3 非 2 幂 + 4 元组共享 micro-exp ⇒ H2 一般不能精确嵌入。
  - Attention：投影 linear ≠ QKᵀ ≠ PV；Full-Stack FP4 把 PV 留 BF16。第一轮评测默认 linear-only。
- **下一步**：cpu-ref 量化器 + 高斯 MSE（H1 / NVFP4 / NVFP4+PTS / H2 命中率）。不在本条目编造 MSE。
- **未做**：clone 外部仓库；CloudAgent；在 910B / CUDA 上跑模型。
