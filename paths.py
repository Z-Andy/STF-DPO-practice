"""统一管理路径配置：将原来的硬编码绝对路径改为相对路径 + 环境变量覆盖。

使用方式：
- 默认相对路径基于仓库根目录（本文件所在目录）
- 也可通过环境变量覆盖：
  - QWEN_MODEL_PATH / QWEN_LORA_PATH
  - DATA_ROOT（数据根目录）
  - HF_HOME（huggingface 缓存目录）

示例：
- Windows PowerShell:
  $env:QWEN_MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
  $env:QWEN_LORA_PATH=".\\saves\\qwen2_vl-7b\\dpo_aesthetic_final"
  $env:DATA_ROOT=".\\final_test"
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def env_or(default: Path | str, env_key: str) -> str:
    val = os.getenv(env_key)
    if val and str(val).strip():
        return str(Path(val))
    return str(default)


# HuggingFace cache
HF_HOME = env_or(REPO_ROOT / ".cache" / "hf", "HF_HOME")

# 模型与 LoRA（默认放在仓库下，自行按需调整或用环境变量覆盖）
MODEL_PATH = env_or(REPO_ROOT / "models" / "Qwen2-VL-7B-Instruct", "QWEN_MODEL_PATH")
LORA_SFT_PATH = env_or(REPO_ROOT / "saves" / "qwen2_vl-7b" / "sft_aesthetic_final", "QWEN_LORA_SFT_PATH")
LORA_DPO_PATH = env_or(REPO_ROOT / "saves" / "qwen2_vl-7b" / "dpo_aesthetic_final", "QWEN_LORA_DPO_PATH")

# 数据根目录（默认指向仓库下的 final_test）
DATA_ROOT = Path(env_or(REPO_ROOT / "final_test", "DATA_ROOT"))

# AADB final test 默认结构
FINAL_TEST_IMG_DIR = str(DATA_ROOT / "Final_Contest_Images")
FINAL_TEST_CSV_PATH = str(DATA_ROOT / "AADB_final_test_40.csv")

# AVA demo 默认结构
AVA_DEMO_DIR = str(REPO_ROOT / "AVA_demo" / "demo")
AVA_DEMO_CSV_PATH = str(REPO_ROOT / "AVA_demo" / "demo.csv")
