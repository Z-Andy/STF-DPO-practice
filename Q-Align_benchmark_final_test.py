import os
from paths import HF_HOME, FINAL_TEST_IMG_DIR, FINAL_TEST_CSV_PATH

# ================= 配置环境变量 (必须在 import torch 之前) =================
CACHE_DIR = HF_HOME
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR
# =========================================================================

import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np

# 路径配置
IMG_DIR = FINAL_TEST_IMG_DIR
CSV_PATH = FINAL_TEST_CSV_PATH

def clean_id(val):
    """稳健的 ID 清洗函数"""
    s = str(val).strip()
    try:
        # 处理 1001.0 -> 1001
        return str(int(float(s)))
    except:
        # 如果自带后缀，去掉它 (后面统一加)
        if s.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            return os.path.splitext(s)[0]
        return s

def main():
    print(f"🚀 正在加载 Q-Align 模型 (OneAlign)...")
    
    try:
        # ⚠️ trust_remote_code=True 会自动下载 q-future/one-align 的代码
        # 它依赖 transformers 4.36+ 和 flash-attn
        model = AutoModelForCausalLM.from_pretrained(
            "q-future/one-align", 
            trust_remote_code=True, 
            torch_dtype=torch.float16, 
            device_map="auto",
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("💡 建议检查: pip list | grep transformers 是否为 4.36.2")
        return

    # 读取并处理 CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到 CSV: {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    df['ID'] = df['ID'].apply(clean_id)
    
    gt_scores = []
    pred_scores = []
    valid_count = 0

    print(f"📊 开始推理 {len(df)} 张图片 (美学评分任务)...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['ID']
        # 尝试匹配 jpg 和 JPG
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMG_DIR, f"{img_id}.JPG")
        
        if not os.path.exists(img_path):
            # print(f"⚠️ 图片未找到: {img_id}") # 太多可以注释掉
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            
            with torch.no_grad():
                # Q-Align 官方 API: task_="aesthetics"
                # 返回值通常是一个 list 或 tensor
                score = model.score([image], task_="aesthetics", input_="image")
                
                # 提取标量值
                if isinstance(score, list):
                    val = float(score[0])
                elif isinstance(score, torch.Tensor):
                    val = score.item() if score.numel() == 1 else score[0].item()
                else:
                    val = float(score)

            pred_scores.append(val)
            gt_scores.append(float(row['score']))
            valid_count += 1
            
        except Exception as e:
            print(f"❌ 处理出错 ID {img_id}: {e}")

    # 计算结果
    if valid_count > 1:
        srcc, _ = spearmanr(gt_scores, pred_scores)
        plcc, _ = pearsonr(gt_scores, pred_scores)
        
        print("\n" + "="*50)
        print(f"🏆 Q-Align (OneAlign) Benchmark Result")
        print(f"有效样本: {valid_count}/{len(df)}")
        print(f"SRCC: {srcc:.4f}")
        print(f"PLCC: {plcc:.4f}")
        print("="*50)
    else:
        print("\n❌ 有效样本不足，无法计算指标。")

if __name__ == "__main__":
    main()