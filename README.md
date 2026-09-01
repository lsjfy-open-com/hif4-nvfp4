# HiF4-NVFP4：Ascend HiFloat4 对 NVFP4 的高精度适配

远程：[`lsjfy-open-com/hif4-nvfp4`](https://github.com/lsjfy-open-com/hif4-nvfp4)（private）。

本仓库研究如何把 NVIDIA **NVFP4** 的量化配方（block-16、E4M3 scale、可选 PTS / RHT / stochastic rounding）映射到昇腾 **HiFloat4 (HiF4)** 硬件（block-64、E6M2 + 两级 micro-exponent），在尽量不损失 NVFP4 级精度的前提下，使用 HiF4 更便宜的 Cube-core 64-dot 路径。当前阶段：**建模优先**。

## 问题一句话

NVFP4 用更细的 16 元块与 FP8 浮点 scale 换精度，但 inter-group 动态范围约 22 binades，常需 PTS，且 64-dot 需要四对 NVFP4 块。HiF4 用 64 元块 + 32-bit 分层 metadata（同为 ~4.5 bits/value），intra-group 4.81 binades、global ~69 binades，一对 HiF4 即可填满 64-dot。目标是：**精度对齐 NVFP4（含 PTS / TE extras），开销走 HiF4 硬件路径**。

## 适配含义

「HiFloat4 高精度适配 NVFP4」不是把 NVFP4 比特布局原样搬到昇腾，而是：

1. 保留 NVFP4 **recipe**（1×16 act/grad、默认 16×16 2D weights、E4M3×E2M1 两级 scale、可选 PTS / RHT / SR）。
2. 把这些 recipe 编译/仿真到 HiF4 **encoding**（E6M2 × 2^{E1_8+E1_16} × S1P2）。
3. 在 linear GEMM 与 attention 上分别量化误差与硬件约束（QKᵀ vs projection linears；Full-Stack FP4 将 PV 留在 BF16）。

三条假设见 [`docs/00-formal-model.md`](docs/00-formal-model.md)：H1 原生 HiF4、H2 四个 16-block 打进一个 64-group 的 NVFP4 仿真打包、H3 混合。

## 文档

| 文件 | 内容 |
| --- | --- |
| [`docs/00-formal-model.md`](docs/00-formal-model.md) | 格式方程、网格对比、16-into-64 可表示性、linear / attention 误差、H1–H3、开放问题 |
| [`docs/01-papers.md`](docs/01-papers.md) | 带注释的论文与文档目录 |
| [`docs/02-eval-plan.md`](docs/02-eval-plan.md) | Gaussian MSE、LLM PTQ、消融、CPU/CUDA sim vs Ascend 910B |
| [`docs/lab-log.md`](docs/lab-log.md) | 实验与建模日志 |

## cpu-ref 量化器

Python 包 `hif4_nvfp4`（`src/hif4_nvfp4/`）是主机参考实现，设备标签 **`cpu-ref`**。无 GPU。方程只采用 [`docs/00-formal-model.md`](docs/00-formal-model.md)：

- `hif4.py`：Algorithm 1（BF16 → E6M2 + E1_8 / E1_16 阈值 4 与 2 + S1P2）
- `nvfp4.py`：TE 公式 `x = E2M1 * E4M3_block16 * FP32_global`；`use_pts=True` 为 peak-to-2688，`False` 为 direct-cast
- `pack.py`：H2 把四个 NVFP4-16 块打进一个 HiF4-64 group（先 NVFP4 再对反量化结果跑 Algorithm 1；精确嵌入一般不可能，只报告命中率）

MXFP4 未实现。

### 运行

```bash
pip install -e ".[dev]"
pytest
python -m hif4_nvfp4.gaussian_mse          # 1024×1024，σ=0.01·2^x，x=0..17
python -m hif4_nvfp4.gaussian_mse --size 128 --seed 0
```

高斯脚本打印绝对 MSE、相对 HiF4 的比值，以及 H2 的「对原矩阵 MSE / 对 NVFP4 反量化 MSE / 精确命中率」。文献比 1:1.32:1.89 只作外部对照，不写进测试断言。

## 工作约定

- 不发明未在文献中出现的数字；评测数字一律标来源或标「待测」。
- 量化公式必须能在 `docs/00-formal-model.md` 里找到；禁止在 eval 脚本里另写一套 scale。

## 关键来源（摘要）

- NVFP4：Transformer Engine NVFP4 文档；NVIDIA *Introducing NVFP4* blog；arXiv:2509.25149；arXiv:2607.04422。
- HiF4：arXiv:2602.11287（inference）；arXiv:2604.08826（pretraining）；[github.com/global-computing-consortium/HiFloat4](https://github.com/global-computing-consortium/HiFloat4)。

文献给出的对照（勿当作本仓库新测）：Gaussian MSE 比 HiF4:NVFP4:MXFP4 = 1:1.32:1.89；64-dot 相对 NVFP4 增量面积约 1/3、功耗约低 10%（arXiv:2602.11287）。
