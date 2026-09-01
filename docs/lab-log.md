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

## 2026-09-01（CST）续：cpu-ref 量化器

- **阶段**：实现 `cpu-ref` 量化器与高斯 MSE harness。未 clone 外部仓库；无 GPU。
- **产物**：
  - `src/hif4_nvfp4/`：`formats.py`（E2M1 / S1P2 / E6M2 / E4M3 / BF16）、`hif4.py`（Algorithm 1）、`nvfp4.py`（TE 公式，PTS on/off）、`pack.py`（H2：NVFP4 反量化后再 Algorithm 1）、`gaussian_mse.py`。
  - `tests/`：网格 roundtrip、E6M2 NaN 广播、Algorithm 1 阈值 4 与 2、高斯协议性质（有限 MSE、内部 σ 上 HiF4 < NVFP4、无 PTS 在端点变差）。**没有**把文献比 1:1.32:1.89 写成断言。
  - MXFP4：跳过。
- **如何跑**：`pip install -e ".[dev]"`；`pytest`；`python -m hif4_nvfp4.gaussian_mse`。
- **本机一次 run**（设备 `cpu-ref`，seed=0，1024×1024，沿 last axis 分组；**不是**论文表格抄录，也不是多种子平均）：
  - 内部 σ（x=4..12）NVFP4+PTS / HiF4 比值约 **1.31**（文献外部对照 1.32，同方向同量级）。
  - x=0（σ=0.01）：HiF4 6.91e-7；NVFP4 1.59e-6（r=2.29）；NVFP4+PTS 9.04e-7（r=1.31）。无 PTS 因 E4M3 小 scale 变差。
  - x=17（σ=1310.72）：HiF4 1.19e4；NVFP4 3.13e4（r=2.64）；NVFP4+PTS 1.56e4（r=1.31）。无 PTS 在 448×6 上溢。
  - H2（PTS 目标）：对原矩阵 MSE 高于 HiF4 直转（两级量化叠加）；对 NVFP4 反量化 MSE 与 HiF4 直转同量级；**精确命中率约 6.8%**（18 张均在 6.7–6.9%）。符合「一般不能精确嵌入」。
- **未做**：MXFP4；Llama WikiText-2 canary；RHT / SR / 2D 16×16；cuda-sim / 910B。

