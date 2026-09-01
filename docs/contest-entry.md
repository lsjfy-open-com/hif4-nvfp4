# 还能用来测试的入口（2026-09-01）

今天是 2026-09-01。IEEE ICME 2026 低位宽量化挑战赛 **已经结束**（报名 2/10–4/10，提交到 4/30，系统 5/15 关闭，5 月下旬颁奖）。官网还留着报名按钮，但榜已经定了，再交也不会进 ICME 名次。

能继续用的是：**同一套协议做本地复现**，以及 **还开着的公开榜**。

## 1. 已关闭：ICME 主赛

- 规则 / 旧榜：https://challenge.gccorg.com/
- GitCode 总仓：https://gitcode.com/GCC-GlobalComputingConsortium/Low-precision_Large_Language_Model_Efficient_Computation_Challenge
- 官方仿真：https://github.com/global-computing-consortium/HiFloat4
- Demo：https://github.com/global-computing-consortium/ICME-Demo
- 参赛复现（Wan2.2 HiF4 PTQ）：https://github.com/GCC-HiFloat/PTQ_Wan2.2
- 问询邮箱（不要指望还能报名）：zhaoy21@tsinghua.org.cn

Sub-1 榜（关榜后的分数，只作对照）：hust_Yao 88.25 / Bits Please 87.9 / Efficient-ai 86.275 / USTC-zhaslab 85.225。HiF4 相对 BF16 的 VBench 平均损失须 &lt; 0.5% 才算达标。

## 2. 还开着、方便测试的入口

### A. VBench 榜（赛题用的客观指标）

赛题 Sub-1 打的就是 VBench I2V。这条现在还能交。

1. 用 `Wan-AI/Wan2.2-I2V-A14B` + 官方 HiF4 仿真做 W4A4 生成（720×1280×61）。
2. 按 [VBench Usage](https://github.com/Vchitect/VBench) 跑评测，得到 `evaluation_results/*.json`。
3. I2V 总分说明：https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#submit-to-leaderboard
4. 把 json 打成 zip（zip **根目录**就是各 json，不要套一层文件夹）。
5. 上传：https://huggingface.co/spaces/Vchitect/VBench_Leaderboard  
   点 **Submit here!**，填模型名和模型链接。

本仓库不上传生成视频；榜只收 json。

### B. OpenS2V-Eval 榜（180 条 prompt，比 5M 轻）

1. 数据：https://huggingface.co/datasets/BestWishYsh/OpenS2V-Eval
   ```bash
   huggingface-cli download --repo-type dataset BestWishYsh/OpenS2V-Eval --local-dir OpenS2V-Eval
   ```
2. 生成视频，文件名用 sample id（见 https://github.com/PKU-YuanGroup/OpenS2V-Nexus/tree/main/eval ）。
3. 跑官方脚本得到 `model_name_eval-type.json`。
4. 榜：https://huggingface.co/spaces/BestWishYsh/OpenS2V-Eval

校准不要拉 OpenS2V-5M 全量。参赛方案里用过约 98 条 prompt；评测用 Eval 的 180 条。

### C. Mini-Challenge 协议（更贴本仓库的 LLM linear/attention）

这条 **没有在线提交盒**，但任务全在 Hugging Face / lm-eval 上，CPU/GPU 都能复现，适合迭代 H1/H2。

W4A4 mini（Pangu-72B-2512，目标：相对 BF16 平均精度损失 ≤ 1%）：

- SuperGPQA
- IF-Eval
- AIME2025
- LiveCodeBench V6
- BFCL-V3

W8A8 mini（OpenPangu-1B + `HuggingFaceFW/fineweb` sample-10BT）：MMLU / GSM8K / MATH500 / HellaSwag / ARC / PIQA。

本仓库第一轮仍用更小的金丝雀：Llama-2-7B + WikiText-2，通过后再上 mini 任务表。

## 3. 建议测试顺序

1. 已有：`cpu-ref` 高斯 MSE（PR #1）。
2. LLM：WikiText-2 canary → mini 任务表。
3. 要对齐赛题视频指标：Wan2.2 + OpenS2V-Eval 180 → 本地 VBench → 可选交 VBench HF 榜。
4. 不要把 5M 视频或 14B 权重下载到这个 cpu-ref 环境。

对照权重（生成侧 HiF4 vs NVFP4，不是 LLM attention 替代实验）：

- https://huggingface.co/ReopenAI/Wan2.2-I2V-14B-HiF4
- https://huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4
