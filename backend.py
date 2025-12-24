import os
import torch
import uvicorn
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from PIL import Image
from paths import HF_HOME, MODEL_PATH, LORA_DPO_PATH

# ================= 1. 配置路径 =================
CACHE_DIR = HF_HOME
os.environ.setdefault("HF_HOME", CACHE_DIR)

MODEL_PATH = MODEL_PATH
LORA_PATH = LORA_DPO_PATH
MAX_PIXELS = 301056

# ================= 2. 初始化 API =================
app = FastAPI(
    title="Qwen2-VL Aesthetic DPO API",
    description="提供美学评分、分析及详细的Logits概率分布",
    version="1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
processor = None
token_ids = []
weights = None
# 定义评分标签 (字符串)
target_tokens = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

@app.on_event("startup")
async def load_model():
    global model, processor, token_ids, weights
    print(f"🚀 [System] 正在加载 Qwen2-VL 模型并注入 DPO 权重...")
    
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        model.load_adapter(LORA_PATH)
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        processor.tokenizer.padding_side = 'left' 

        # 获取 1-10 的 Token ID
        token_ids = [processor.tokenizer.encode(t, add_special_tokens=False)[-1] for t in target_tokens]
        # 对应的数值权重
        weights = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32).to(model.device)
        
        print("✅ [System] 模型加载完成！")
    except Exception as e:
        print(f"❌ [System] 模型加载失败: {e}")
        raise e

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="模型未就绪")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    try:
        # 1. 处理图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. 构造 Prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": "Analyze this image's aesthetics. Briefly describe the composition and lighting, then provide a rating level from 1 to 10. Format: Analysis: [text] Rating Level: [score]"}
            ]
        }]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        # 3. 推理 (开启 output_scores)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False, 
                return_dict_in_generate=True,
                output_scores=True 
            )

        # 4. 后处理
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        # === 核心逻辑：提取 Logits 分布 ===
        final_score = 5.0
        score_distribution = {} # 用于存储 "1": 0.01, "2": 0.05 ...
        rating_token_pos = -1
        
        # 定位 Rating 数字的位置
        for pos, tid in enumerate(generated_ids):
            if tid.item() in token_ids and pos > 10: 
                rating_token_pos = pos
                break
        
        if rating_token_pos != -1:
            # A. 提取 Logits
            logits = outputs.scores[rating_token_pos][0]
            relevant_logits = logits[token_ids]
            
            # B. 计算概率 (Temperature=0.5)
            # Logits 本身范围很大且有负数，不适合直接展示，前端通常需要概率
            temperature = 0.5
            probs = F.softmax(relevant_logits.float() / temperature, dim=-1)
            
            # C. 计算加权分
            final_score = torch.sum(probs * weights).item()
            
            # D. [新增] 构建详细分布字典
            # 将 Tensor 转为 Python List 以便 JSON 序列化
            probs_list = probs.tolist() 
            
            for i, score_label in enumerate(target_tokens):
                # 将概率保留 4 位小数
                score_distribution[score_label] = round(probs_list[i], 4)
        else:
            # 兜底：如果没有找到数字，返回均匀分布或空
            for label in target_tokens:
                score_distribution[label] = 0.0
        
        return {
            "status": "success",
            "data": {
                "score": round(final_score, 4),
                "analysis": generated_text.replace("Analysis:", "").replace("Rating Level:", "").strip(),
                "distribution": score_distribution, # <--- 新增字段
                "raw_text": generated_text
            }
        }

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)


# 示例返回格式：
#     {
#     "status": "success",
#     "data": {
#         "score": 8.7421,
#         "analysis": "The composition is excellent...",
#         "distribution": {
#             "1": 0.0000,
#             "2": 0.0000,
#             "3": 0.0001,
#             "4": 0.0005,
#             "5": 0.0023,
#             "6": 0.0150,
#             "7": 0.1200,
#             "8": 0.6500,  <-- 模型最倾向于 8 分
#             "9": 0.2000,
#             "10": 0.0121
#         },
#         "raw_text": "..."
#     }
# }