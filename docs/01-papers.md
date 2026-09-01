# 文献目录（注释）

只收录本项目直接依赖的格式、配方与评测来源。摘要里的数字均来自该文献本身，不是本仓库复现。

## NVFP4 格式与训练配方

### Transformer Engine — NVFP4 用户文档

- HTML：<https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html>
- 备用（开发版）：<https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html>
- 内容：E2M1；\(x=x_{\mathrm{E2M1}}\cdot s_{\mathrm{block}}\cdot s_{\mathrm{global}}\)；\(s_{\mathrm{global}}=\mathrm{global\_amax}/(448\cdot 6)\)；\(s_{\mathrm{block}}=(\mathrm{block\_amax}/6)/s_{\mathrm{global}}\)，存 E4M3。默认权重 16×16、激活/梯度 1×16。SR 仅梯度。RHT \(d=16\) 仅 wgrad 输入。GEMM 仅 TN；columnwise 转置存储；scale swizzle。`NVFP4BlockScaling` API。
- 对本项目：H2 的「官方 recipe」定义；TE extras 清单。

### NVIDIA Technical Blog — Introducing NVFP4 for Efficient and Accurate Low-Precision Inference

- <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- Alvarez et al., 2025。两级 scale：每 16 值 E4M3 + 每 tensor FP32。4.5 bits/value。相对 FP16 约 3.5×、相对 FP8 约 1.8× 的模型内存（blog 陈述）。HiF4 论文将「peak → 2688 再量化」的 PTS 管线引用到此文。
- 对本项目：PTS / 2688 的推理侧来源；与 MXFP4（E8M0、block-32）的对照表。

### NVIDIA Technical Blog — NVFP4 Trains with Precision of 16-Bit and Speed and Efficiency of 4-Bit

- <https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/>
- 12B hybrid Mamba–Transformer、10T tokens、相对 FP8 的训练曲线宣传稿。细节以 arXiv:2509.25149 为准。

### arXiv:2509.25149 — Pretraining Large Language Models with NVFP4

- <https://arxiv.org/abs/2509.25149> · HTML：<https://arxiv.org/html/2509.25149>
- NVIDIA。格式相对 MXFP4 的三点：块 32→16、UE8M0→E4M3、加 FP32 张量 scale。配方：少量敏感 linear 留高精度、RHT 16×16 于 wgrad、权重 2D 16×16、梯度 SR、其余 RTN。12B / 10T。注意力、embedding、softmax、QK/PV batched GEMM 留更高精度。Appendix B 给出 encode/decode 过程。
- 对本项目：NVFP4 训练语义；「linear 已量化、attention score 未量化」的基线边界。

### arXiv:2607.04422 — Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention

- <https://arxiv.org/abs/2607.04422> · HTML：<https://arxiv.org/html/2607.04422>
- Ding, Ma, Tong, Xing, Wang, Li。把 NVFP4 从线性层扩到 AdamW state、Root/Muon、attention。Attention 表：\(\hat Q\hat K^{\top}\) 与 \(\hat{dS}\) 路径 NVFP4；**\(P\hat V\)、\(P^{\top}dO\)、\(dO\hat V^{\top}\) 留 BF16**。Appendix C 分析 PV / \(dOV^{\top}\) 敏感性。3B / 64B tokens。
- 对本项目：attention 缺口的主文献；eval 里「linear-only vs +attention-score」且 PV 默认 BF16 的依据。

## HiF4 格式、推理与预训练

### arXiv:2602.11287 — HiFloat4 Format for Language Model Inference

- <https://arxiv.org/abs/2602.11287> · HTML：<https://arxiv.org/html/2602.11287>
- Luo et al.。格式定义（Eq. 2）、Algorithm 1、Table II 与 NVFP4 对照、高斯 MSE 1:1.32:1.89、64-dot 面积/功耗、Llama2-7B / Llama3-8B / Qwen2.5-14B / Mistral-7B 的 BF16 vs NVFP4 vs NVFP4+PTS vs HiF4 vs HiF4+HiGPTQ（Table III），以及 DeepSeek-V3.1 / LongCat。PTQ 只转 linear（除 embedding 与 LM head）。小模型实验：Nvidia GPU 与 Ascend NPU 上的 **simulated** 4-bit；大模型：vLLM + Ascend 910B。
- 对本项目：H1 的规范；网格与 16-into-64 讨论的事实表；eval 的模型与基准清单。

### arXiv:2604.08826 — HiFloat4 Format for Language Model Pre-training on Ascend NPUs

- <https://arxiv.org/abs/2604.08826> · HTML：<https://arxiv.org/html/2604.08826>
- Taghian, Peng, et al.。Ascend 上 HiF4 vs MXFP4（**实验不含 NVFP4**，因 NVFP4 与 Cube 路径更不合）。Linear / expert GEMM 全 FP4。HiF4：RHT \(k=64\) + 梯度 NR；MXFP4：RHT + SR + truncation-free scaling。OpenPangu-1B / Llama3-8B / Qwen3-MoE-30B，50B tokens。相对 BF16 的 loss gap：HiF4 约 1.0% 量级、MXFP4 约 1.5% 量级（具体见表 3）。消融：HiF4 加 SR 变差，单加 RHT 变好。
- 对本项目：H1 训练 extras **不要**照搬 TE-SR 的证据；RHT 块大小应跟 group（64）对齐。

### GitHub — global-computing-consortium/HiFloat4

- <https://github.com/global-computing-consortium/HiFloat4>
- 描述为跨 CUDA 与 Ascend 的 HiFloat4 量化/仿真库（pseudo-quant linear）。本仓库不 clone 该 repo；需要 API 细节时查其 README / 引用 arXiv:2602.11287。

## 辅助（格式族、评测集）

### OCP MX / MXFP4

- Rouhani et al., *Microscaling Data Formats for Deep Learning*, arXiv:2310.10537。
- OCP MX spec。MXFP4：E2M1 + UE8M0、block-32、4.25 bits/value。作为 NVFP4 / HiF4 的共同对照，不是本项目适配目标。

### 评测数据与模型卡（eval plan 引用）

- ARC：Clark et al., arXiv:1803.05457。
- BoolQ：Clark et al., NAACL 2019。
- HellaSwag：Zellers et al., ACL 2019。
- LAMBADA：Paperno et al., ACL 2016。
- PIQA：Bisk et al., AAAI 2020。
- WinoGrande：Sakaguchi et al., arXiv:1907.10641。
- MMLU：Hendrycks et al., ICLR 2021。
- WikiText-2：Merity et al.（canary PPL，见 eval plan）。
- Llama 2：Touvron et al., arXiv:2307.09288。
- Llama 3：Dubey et al., arXiv:2407.02373（e-print）。
- Qwen2.5：Yang et al., arXiv:2412.15115。
- Mistral 7B：Jiang et al., arXiv:2310.06825。

## 阅读顺序（本项目）

1. TE NVFP4 文档 + 2602.11287 §II（两种 encoding）。
2. 2509.25149 §2 / §4（recipe extras）与 2602.11287 §III（误差与 64-dot）。
3. 2607.04422 §3.5 / Table 2 / Appendix C（attention 分割）。
4. 2604.08826 §3–§5（Ascend 预训练 extras 差异）。
5. 本仓库 [`00-formal-model.md`](00-formal-model.md) 的 H1/H2/H3。


## ICME 2026 赛题与 Hugging Face 资源

- 官网：<https://challenge.gccorg.com/>
- Sub-Challenge 1：Wan2.2-I2V-A14B 上 HiF4/MXFP4 W4A4；数据 [`BestWishYsh/OpenS2V-5M`](https://huggingface.co/datasets/BestWishYsh/OpenS2V-5M)、[`BestWishYsh/OpenS2V-Eval`](https://huggingface.co/datasets/BestWishYsh/OpenS2V-Eval)；指标 VBench。
- Mini-Challenge 训练：[`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb)。
- 论文页（无关联数据集）：<https://huggingface.co/papers/2602.11287>
- 参赛/社区权重：ReopenAI/Wan2.2-I2V-14B-HiF4；nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4；BitsPlease/HiSQRot4。
