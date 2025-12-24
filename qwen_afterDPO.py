import pandas as pd
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# ================= 1. 配置路径 =================
from paths import MODEL_PATH, LORA_DPO_PATH, FINAL_TEST_IMG_DIR, FINAL_TEST_CSV_PATH, HF_HOME

os.environ.setdefault("HF_HOME", HF_HOME)

MODEL_PATH = MODEL_PATH
# ⚠️ 这里必须指向你刚跑完的 DPO 文件夹
LORA_PATH = LORA_DPO_PATH 

IMG_DIR = FINAL_TEST_IMG_DIR
CSV_PATH = FINAL_TEST_CSV_PATH
BATCH_SIZE = 10 # 双 3090 建议 10-12
MAX_PIXELS = 301056

def main():
    print(f"🚀 正在加载模型并注入 DPO 终极权重...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # 加载 DPO 适配器
    model.load_adapter(LORA_PATH) 
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = 'left' 

    # 准备 1-10 的 Token ID 和权重
    target_tokens = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    token_ids = [processor.tokenizer.encode(t, add_special_tokens=False)[-1] for t in target_tokens]
    weights = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32).to(model.device)

    # 读取 CSV
    df = pd.read_csv(CSV_PATH)

    # === 【修复 1】更稳健的 ID 处理函数 ===
    def clean_id(x):
        try:
            # 尝试处理像 "1001.0" 这样的纯数字 ID，转为 "1001"
            return str(int(float(x)))
        except (ValueError, TypeError):
            # 如果报错（说明是文件名字符串），直接去除空格保留原样
            return str(x).strip()

    df['ID'] = df['ID'].apply(clean_id)
    # ===================================
    
    results = []
    print(f"📊 开始最终大考 (样本数: {len(df)})...")

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_msgs, batch_gt = [], []

        for _, row in batch_df.iterrows():
            # === 【修复 2】智能判断是否需要加 .jpg 后缀 ===
            img_name = row['ID']
            # 如果 ID 结尾不是常见的图片后缀，则手动添加 .jpg
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                img_name = f"{img_name}.jpg"
            
            img_path = os.path.join(IMG_DIR, img_name)
            # ==========================================

            if os.path.exists(img_path):
                # ⚠️ 还原训练时的 CoT Prompt 顺序
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": img_path, "max_pixels": MAX_PIXELS},
                    {"type": "text", "text": "Analyze this image's aesthetics. Briefly describe the composition and lighting, then provide a rating level from 1 to 10. Format: Analysis: [text] Rating Level: [score]"}
                ]}]
                batch_msgs.append(msg)
                batch_gt.append(row['score'])
            else:
                print(f"⚠️ Warning: Image not found: {img_path}")

        if not batch_msgs: continue

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
        image_inputs, _ = process_vision_info(batch_msgs)
        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150, # 给足够的长度写 Analysis
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

        for b in range(len(batch_msgs)):
            # 1. 获取生成的 token 序列
            gen_ids = outputs.sequences[b][len(inputs.input_ids[b]):]
            
            # 2. 精准定位：寻找第一个出现在 Analysis 之后的数字 Token
            rating_token_pos = -1
            for pos, tid in enumerate(gen_ids):
                # 这里的 20 是为了跳过开头可能出现的无关数字，确保取到的是 Rating 附近的
                if tid.item() in token_ids and pos > 20: 
                    rating_token_pos = pos
                    break
            
            if rating_token_pos != -1:
                # 3. 提取该位置的 Logits 进行概率加权
                logits = outputs.scores[rating_token_pos][b]
                relevant_logits = logits[token_ids]
                
                # 应用 Temperature 进行缩放
                temperature = 0.5
                probs = F.softmax(relevant_logits.float() / temperature, dim=-1)
                
                final_score = torch.sum(probs * weights).item()
            else:
                final_score = 5.0 # 兜底分数

            results.append({"gt": batch_gt[b], "pred": final_score})

    # 计算最终硬指标
    if len(results) > 0:
        res_df = pd.DataFrame(results)
        srcc, _ = spearmanr(res_df['gt'], res_df['pred'])
        plcc, _ = pearsonr(res_df['gt'], res_df['pred'])
        
        print("\n" + "="*50)
        print(f"🏆 DPO 最终跑分结果:")
        print(f"SRCC: {srcc:.4f}")
        print(f"PLCC: {plcc:.4f}")
        print(f"Q-Align 基准: 0.6557")
        print("="*50)
    else:
        print("\n❌ 没有生成任何有效结果，请检查图片路径或 CSV ID。")

if __name__ == "__main__":
    main()