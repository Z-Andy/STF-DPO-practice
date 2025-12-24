import pandas as pd
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm
# ================= 1. 配置 =================
from paths import MODEL_PATH, LORA_SFT_PATH, AVA_DEMO_DIR, AVA_DEMO_CSV_PATH

MODEL_PATH = MODEL_PATH
LORA_PATH = LORA_SFT_PATH
IMG_DIR = AVA_DEMO_DIR
CSV_PATH = AVA_DEMO_CSV_PATH
BATCH_SIZE = 10 

def main():
    print(f"🚀 加载 SFT 模型中...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.load_adapter(LORA_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = 'left' 

    # 1-10 的 Token ID
    target_tokens = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    token_ids = [processor.tokenizer.encode(t, add_special_tokens=False)[-1] for t in target_tokens]
    weights = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32).to(model.device)

    df = pd.read_csv(CSV_PATH)
    df['ID'] = df['ID'].apply(lambda x: str(int(float(x))))
    results = []

    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_msgs, batch_gt = [], []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(IMG_DIR, f"{row['ID']}.jpg")
            if os.path.exists(img_path):
                batch_msgs.append([{"role": "user", "content": [{"type": "image", "image": img_path, "max_pixels": 301056}, 
                                   {"type": "text", "text": "Analyze this image's aesthetics. Briefly describe the composition and lighting, then provide a rating level from 1 to 10. Format: Analysis: [text] Rating Level: [score]"}]}])
                batch_gt.append(row['score'])
        
        if not batch_msgs: continue
        
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
        image_inputs, _ = process_vision_info(batch_msgs)
        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, return_dict_in_generate=True, output_scores=True)

        for b in range(len(batch_msgs)):
            # --- 修正后的精准提取逻辑 ---
            # 1. 获取模型生成的 Token 序列（不含 prompt）
            gen_ids = outputs.sequences[b][len(inputs.input_ids[b]):]
            
            # 2. 我们通过查找数字 Token 在生成序列中的实际位置来定位
            # 寻找序列中第一个出现的 1-10 数字 Token 的索引
            rating_token_pos = -1
            for pos, tid in enumerate(gen_ids):
                if tid.item() in token_ids:
                    # 额外检查：确保这个数字出现在文本的后半段（避开 Analysis 里的数字）
                    if pos > 15: 
                        rating_token_pos = pos
                        break
            
            if rating_token_pos != -1:
                # 3. 提取这个位置的原始 Logits
                # 注意：outputs.scores[pos] 对应的是 gen_ids[pos]
                logits = outputs.scores[rating_token_pos][b]
                relevant_logits = logits[token_ids]
                probs = F.softmax(relevant_logits.float(), dim=-1)
                weighted_score = torch.sum(probs * weights).item()
            else:
                weighted_score = 5.0 # 兜底

            results.append({"gt": batch_gt[b], "pred": weighted_score})

    res_df = pd.DataFrame(results)
    srcc, _ = spearmanr(res_df['gt'], res_df['pred'])
    print(f"\n✅ 跑分完成! 样本数: {len(res_df)} | SRCC: {srcc:.4f}")

if __name__ == "__main__":
    main()