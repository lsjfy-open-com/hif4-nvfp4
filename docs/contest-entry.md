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

本仓库第一轮用 **tiny smoke Transformer** 把 Mini-Challenge 协议跑通（CI / pytest，无 7B 权重）。Llama-2-7B + WikiText-2 是 opt-in，见 §4。

## 3. 建议测试顺序

1. 已有：`cpu-ref` 高斯 MSE（PR #1）。
2. 本仓库 LLM harness：tiny smoke 模型上 WikiText-2 canary 路径 + Mini-Challenge 任务表（数据缺失则 skip）。
3. Opt-in：Llama-2-7B WikiText-2 word PPL，再 lm-eval subset。
4. 要对齐赛题视频指标：Wan2.2 + OpenS2V-Eval 180 → 本地 VBench → 可选交 VBench HF 榜。那是视频 DiT，**不是**本仓库 linear/attention 实验。
5. 不要把 5M 视频或 14B 权重下载到这个 cpu-ref 环境；不要 clone HiFloat4 / ICME-Demo / VBench 进本仓库（只留链接）。

对照权重（生成侧 HiF4 vs NVFP4，不是 LLM attention 替代实验）：

- https://huggingface.co/ReopenAI/Wan2.2-I2V-14B-HiF4
- https://huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4

## 4. 本仓库本地 Mini-Challenge harness

IEEE ICME 2026 **不要报名**。本仓库能跑的是 linear + optional QKᵀ 的 LLM 协议（embedding / lm_head / softmax / PV 保持高精度），不是 Wan2.2 + VBench。

配置：[`configs/mini_challenge.yaml`](../configs/mini_challenge.yaml)  
脚本：[`scripts/mini_challenge_eval.py`](../scripts/mini_challenge_eval.py)

量化走已有 `cpu-ref` 公式（[`src/hif4_nvfp4/hif4.py`](../src/hif4_nvfp4/hif4.py) Algorithm 1；[`src/hif4_nvfp4/nvfp4.py`](../src/hif4_nvfp4/nvfp4.py) TE + PTS peak-to-2688）。`--device cuda-sim` 只把 GEMM 放到 CUDA，量化器仍是 cpu-ref；没有 GPU 时拒绝并报错，不会改标成 cpu-ref。

```bash
pip install -e ".[dev,eval]"
# CI / 默认：随机初始化 tiny Transformer，不拉 7B
python scripts/mini_challenge_eval.py --model smoke --format hif4 --device cpu-ref
python scripts/mini_challenge_eval.py --model smoke --format nvfp4_pts --device cpu-ref
# +attention-score（量化 QKᵀ；PV 仍高精度）。默认关。
python scripts/mini_challenge_eval.py --model smoke --quantize-qk

# Llama-2-7B 是 opt-in（需 HF 权限；失败则 skip，不编造 PPL）
pip install -e ".[eval-full]"
python scripts/mini_challenge_eval.py --model llama2-7b --format hif4
# 或: HIF4_EVAL_MODEL=meta-llama/Llama-2-7b-hf
```

跑序与 [`docs/02-eval-plan.md`](02-eval-plan.md) 一致：先 WikiText-2 word PPL canary，再 lm-eval subset（hellaswag / arc_easy / arc_challenge / piqa / mmlu）。YAML 里写明 Mini-Challenge W4A4 名：SuperGPQA / IFEval / AIME2025 / LiveCodeBench / BFCL；缺数据集或 lm-eval 任务时 **skip + 原因**，不填假分。

外部仿真/赛题仓（链接，不 clone）：

- HiFloat4：https://github.com/global-computing-consortium/HiFloat4
- ICME-Demo：https://github.com/global-computing-consortium/ICME-Demo
- VBench：https://github.com/Vchitect/VBench

