import pandas as pd
import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# ================= 1. 配置 =================
from paths import MODEL_PATH, AVA_DEMO_DIR, AVA_DEMO_CSV_PATH

MODEL_PATH = MODEL_PATH
IMG_DIR = AVA_DEMO_DIR
CSV_PATH = AVA_DEMO_CSV_PATH
BATCH_SIZE = 4  # 开启详细显示建议 Batch 先设小一点，如 4 或 8
MAX_PIXELS = 301056 

def main():
    print("🚀 正在初始化 Q-Align 算法加速引擎...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 预准备 Token IDs: 获取词表中 "1" 到 "10" 的索引
    target_tokens = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    token_ids = [processor.tokenizer.encode(t, add_special_tokens=False)[-1] for t in target_tokens]
    weights = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32).to(model.device)

    df = pd.read_csv(CSV_PATH)
    df['ID'] = df['ID'].astype(str)
    results = []

    print(f"\n" + "="*80)
    print(f"{'IMAGE_ID':<15} | {'GT':<6} | {'PRED':<6} | {'DISTRIBUTION (L1 -> L10)'}")
    print("-"*80)

    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_messages, valid_indices = [], []

        for idx, row in batch_df.iterrows():
            img_path = os.path.join(IMG_DIR, f"{row['ID']}.jpg")
            if os.path.exists(img_path):
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path, "max_pixels": MAX_PIXELS},
                        {"type": "text", "text": "Rate the aesthetics from 1 to 10. Output: Rating Level: [score]."}
                    ],
                }]
                batch_messages.append(msg)
                valid_indices.append(idx)

        if not batch_messages: continue

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, _ = process_vision_info(batch_messages)
        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10, 
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

        # 核心逻辑：在生成的 Token 序列中寻找分数位置
        # 由于 Prompt 是 "Rating Level: [score]"，分数通常在第 3-5 个生成的 Token
        for b_idx in range(len(batch_messages)):
            sample_id = df.loc[valid_indices[b_idx], 'ID']
            gt_score = df.loc[valid_indices[b_idx], 'score']
            
            # 我们遍历生成的前几个 Token，找到对数字 Token 响应最强的那个位置
            best_weighted_score = 5.0
            max_prob_sum = 0
            best_probs = None

            # 搜索生成的前 5 个 Token 位置，寻找“分数位”
            for t_pos in range(min(5, len(outputs.scores))):
                logits = outputs.scores[t_pos][b_idx]
                relevant_logits = logits[token_ids]
                probs = F.softmax(relevant_logits.float(), dim=-1)
                
                current_prob_sum = probs.sum().item()
                if current_prob_sum > max_prob_sum:
                    max_prob_sum = current_prob_sum
                    best_probs = probs
                    best_weighted_score = torch.sum(probs * weights).item()

            results.append({"gt": gt_score, "pred": best_weighted_score})

            # --- 终端可视化显示 ---
            # 构建概率分布的简易条形图 (用字符表示)
            dist_str = ""
            for p in best_probs.tolist():
                bar = "█" if p > 0.2 else "▒" if p > 0.05 else "░"
                dist_str += bar
            
            print(f"{sample_id:<15} | {gt_score:<6.2f} | {best_weighted_score:<6.2f} | {dist_str} (MaxP: {max_prob_sum:.2%})")

    # 计算最终指标
    res_df = pd.DataFrame(results)
    srcc, _ = spearmanr(res_df['gt'], res_df['pred'])
    plcc, _ = pearsonr(res_df['gt'], res_df['pred'])
    print("="*80)
    print(f"✅ 加速跑分完成!")
    print(f"SRCC: {srcc:.4f} | PLCC: {plcc:.4f}")

if __name__ == "__main__":
    main()