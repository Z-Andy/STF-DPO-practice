# 期末大作业：Q-Align 复现与基于 Qwen2-VL 的美学评分超越

本仓库包含一次计算机视觉期末项目：以 2024 顶会美学/图像质量评估模型 **Q-Align (OneAlign)** 为基线，先完成复现与问题分析，再通过 **更换基座到 Qwen2-VL-7B + 思维链(CoT) SFT + DPO 偏好对齐** 在未见数据集上取得更好的相关性指标（SRCC）。

- 项目综述与实验分析见：`项目概述.md`

## 1. 项目结构

- `项目概述.md`：完整实验报告（复现、问题、改进、曲线与图表）。
- `Q-Align_benchmark.py`：Q-Align 复现/基准脚本（AVA），计算 SRCC/PLCC。
- `Q-Align_benchmark_final_test.py`：更稳健的 final_test 版本（ID 清洗、jpg/JPG 兼容）。
- `qwen_aesthetic_inference.py`：Qwen2-VL-7B 原生（未微调）推理/可视化概率分布示例。
- `qwen_afterSTF.py`：注入 **SFT LoRA** 后的推理与跑分脚本。
- `qwen_afterDPO.py`：注入 **DPO LoRA** 后的推理与跑分脚本（AADB unseen 评测的主脚本）。
- `backend.py`：FastAPI 后端服务，提供单图美学分析、分数与 1~10 概率分布。
- `requirements_qalign.txt`：Q-Align/OneAlign 相关依赖（含 CUDA/torch/flash-attn 版本约束）。
- `requirements_qwen2vl.txt`：Qwen2-VL 推理/微调相关依赖（如有）。
- `AVA_demo/`：Demo 数据（`demo.csv` + 若干图片）用于快速验证推理流程。
- `final_test/`：AADB final test 子集（`AADB_final_test_40.csv` + `Final_Contest_Images/`）。

## 2. 主要方法说明（做了什么）

### 2.1 Q-Align 基线
- 使用 `q-future/one-align` 的官方 `model.score(..., task_="aesthetics")` 接口对图片打分。
- 在 AADB 未见数据集上得到基线 SRCC（报告中记录为 0.6557）。

### 2.2 Qwen2-VL 10 档概率加权打分（Q-Align 风格）
本项目在 Qwen2-VL 上复刻 Q-Align 的“档位概率加权”做法：
- Prompt 让模型输出 `Rating Level: [1~10]`。
- 从 `generate(..., output_scores=True)` 取出“分数 token 位置”的 logits。
- 对 1~10 的 token 做 softmax 得到概率分布，再与权重 `[1..10]` 加权求期望作为最终分数。

相关脚本：`qwen_aesthetic_inference.py`、`qwen_afterSTF.py`、`qwen_afterDPO.py`。

### 2.3 CoT + SFT
- 利用外部大模型生成审美分析文本（Composition/Lighting 等），形成 `Analysis: ... Rating Level: ...` 的监督格式数据。
- 对 Qwen2-VL-7B 做 SFT（LoRA），显著提升 SRCC。

### 2.4 DPO
- 构造偏好对（相近分数的 hard pairs），用 DPO 进一步强化排序能力。
- 在 AADB unseen 上 SRCC 进一步提升，并最终超越 Q-Align。

## 3. 环境与依赖

> 说明：仓库已将脚本中的硬编码绝对路径改为**相对路径 + 环境变量可覆盖**（见 `paths.py`）。

- 相对路径默认以仓库根目录为基准：
  - 数据：`final_test/`、`AVA_demo/`
  - HF 缓存：`.cache/hf/`
  - 模型/LoRA（默认占位）：`models/`、`saves/`（可用环境变量指向你真实的权重目录）

常用环境变量：
- `HF_HOME`：HuggingFace 缓存目录
- `DATA_ROOT`：数据根目录（默认 `./final_test`）
- `QWEN_MODEL_PATH`：Qwen2-VL 模型路径或 Hub 名称
- `QWEN_LORA_SFT_PATH` / `QWEN_LORA_DPO_PATH`：SFT/DPO LoRA 目录

### 3.1 Q-Align/OneAlign 依赖
- 依赖文件：`requirements_qalign.txt`
- 关键点：`transformers==4.36.2`、对应 CUDA 的 `torch`，以及 `flash-attn`（通常需要 Linux + CUDA 编译/whl）。

### 3.2 Qwen2-VL 推理/服务依赖
- 关键包：`transformers`、`torch`、`fastapi`、`uvicorn`、`pillow`、`scipy`、`tqdm`。

## 4. 使用方式（脚本入口）

### 4.1 跑 Q-Align 基线（AADB final_test）
- 脚本：`Q-Align_benchmark_final_test.py`
- 你需要修改：
  - `IMG_DIR`
  - `CSV_PATH`
  - （可选）`CACHE_DIR`/`HF_HOME`

### 4.2 跑 Qwen2-VL + SFT/DPO 的 AADB 评测
- SFT：`qwen_afterSTF.py`
- DPO：`qwen_afterDPO.py`（一般以这个为最终结果）
- 你需要修改：
  - `MODEL_PATH`：Qwen2-VL-7B-Instruct 权重路径/Model Hub 名称
  - `LORA_PATH`：对应 SFT 或 DPO 的 LoRA 目录
  - `IMG_DIR`、`CSV_PATH`

### 4.3 启动单图评分 API
- 脚本：`backend.py`
- 接口：`POST /analyze`，表单字段：`file`（图片）
- 返回：`score`（加权期望分）、`analysis`（模型生成分析）、`distribution`（1~10 概率分布）、`raw_text`。

## 5. 指标口径
- **SRCC**：Spearman Rank Correlation，衡量排序一致性（本任务核心）。

## 6. 常见问题（FAQ）

1) **Windows 上无法安装/使用 flash-attn？**
- `flash-attn` 通常对 Linux + CUDA 环境更友好；建议在 Linux 服务器/WSL 环境跑 Q-Align。

2) **CSV 的 `ID` 带小数或带后缀导致图片匹配失败？**
- `Q-Align_benchmark_final_test.py` 与 `qwen_afterDPO.py` 内已加入 `clean_id` 逻辑，尽量兼容 `1001.0`、`1001.jpg` 等情况。

3) **分数 token 位置找不到导致全是 5.0？**
- `qwen_afterDPO.py` 中通过“跳过前若干 token”避免抓到 Analysis 里的数字；必要时可调整 `pos > 20` 的阈值和 prompt 模板。

4) **训练/数据 json 里还有 `/data6t/...` 这种绝对路径？**
- `ava_aesthetic_sft_final_v3.json`、`ava_aesthetic_dpo_final_v3.json` 内可能写死了图片绝对路径。
- 这类文件通常是训练数据的导出结果，建议：
  - 重新生成数据时改为相对路径；或
  - 批量替换前缀（例如把 `/data6t/EVA/mmmodels/AVA_demo/demo/` 替换为 `AVA_demo/demo/`）。

## 7. 致谢
- Q-Align / OneAlign：`q-future/one-align`。
- Qwen2-VL：Qwen 系列多模态模型与 Transformers 生态。
