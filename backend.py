"""FastAPI backend for pairwise aesthetic comparison using Qwen2-VL + (optional) DPO LoRA.

API:
  GET  /health
  GET  /status
  POST /compare   (multipart/form-data: image_a, image_b)

Return:
  - winner: "A" | "B" | "tie"
  - score_a/score_b: expected score from 1..10 based on token-prob distribution
  - distribution_a/distribution_b: probabilities for labels "1".."10"
  - reasoning: model-generated text (chain-of-thought may be present, depending on model behavior)

Notes on policy/safety:
  This server does not attempt to force the model to reveal hidden reasoning.
  It asks for a brief explanation; the model may or may not include internal chain-of-thought.
"""

import io
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from starlette.responses import JSONResponse
except Exception:  # pragma: no cover
    from fastapi.responses import JSONResponse

from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# qwen-vl-utils: 处理多模态输入（图片/视频）
try:
    from qwen_vl_utils import process_vision_info
except Exception as e:  # pragma: no cover
    process_vision_info = None
    _qwen_vl_utils_import_error = str(e)

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

# ---- Config (env-overridable) ----
CACHE_DIR = os.getenv("CACHE_DIR", "")
os.environ.setdefault("HF_HOME", CACHE_DIR)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "",
)
LORA_PATH = os.getenv(
    "LORA_PATH",
    "",
)
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "301056"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
PORT = int(os.getenv("PORT", "6006"))

TARGET_TOKENS: List[str] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# 固定使用单卡，避免 device_map="auto" 在部分环境下触发 meta tensor
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda:0")


# ---- App ----
app = FastAPI(
    title="Qwen2-VL Pairwise Aesthetic Compare API",
    description="Upload two images and compare aesthetics with Qwen2-VL + DPO LoRA.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Globals ----
model = None
processor = None
weights = None
score_token_ids: List[int] = []
model_ready = False
model_load_error: Optional[str] = None


def api_success(data: dict):
    return {"status": "success", "data": data}


def api_error(code: str, message: str, http_status: int = 400, details=None):
    payload = {"status": "error", "error": {"code": code, "message": message}}
    if details is not None:
        payload["error"]["details"] = details
    return JSONResponse(status_code=http_status, content=payload)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    code = "HTTP_ERROR"
    if exc.status_code == 400:
        code = "BAD_REQUEST"
    elif exc.status_code == 503:
        code = "MODEL_NOT_READY"
    return api_error(code=code, message=str(exc.detail), http_status=exc.status_code)


@app.get("/health")
async def health():
    return api_success({"ok": True})


@app.get("/status")
async def status():
    device = None
    dtype = None
    has_meta_params = None
    if model is not None:
        try:
            p = next(model.parameters())
            device = str(p.device)
            dtype = str(p.dtype)
            # 检测是否仍有 meta 参数（若为 True，通常会导致 "Cannot copy out of meta tensor"）
            has_meta_params = any(getattr(pp, "is_meta", False) for pp in model.parameters())
        except Exception:
            pass

    return api_success(
        {
            "model_ready": bool(model_ready),
            "model_path": MODEL_PATH,
            "lora_path": LORA_PATH,
            "model_load_error": model_load_error,
            "device": device,
            "dtype": dtype,
            "has_meta_params": has_meta_params,
            "max_pixels": MAX_PIXELS,
            "temperature": TEMPERATURE,
            "target_tokens": TARGET_TOKENS,
            "has_process_vision_info": process_vision_info is not None,
            "cuda_device": CUDA_DEVICE,
        }
    )


@app.on_event("startup")
async def load_model():
    global model, processor, weights, score_token_ids, model_ready, model_load_error

    model_ready = False
    model_load_error = None

    try:
        # 1) 基座模型：固定单卡加载（更稳定，优先跑通）
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        # 显式搬到指定设备
        model = model.to(CUDA_DEVICE)

        # 2) LoRA：优先使用 PEFT 方式加载（比 load_adapter 更通用/稳定）
        if LORA_PATH and os.path.exists(LORA_PATH):
            if PeftModel is None:
                print("⚠️ [System] peft 未安装或不可用，尝试使用模型自带 load_adapter")
                try:
                    model.load_adapter(LORA_PATH)
                except Exception as e:
                    raise RuntimeError(f"LoRA 加载失败（peft 不可用且 load_adapter 失败）: {e}")
            else:
                model = PeftModel.from_pretrained(model, LORA_PATH)
        else:
            print("⚠️ [System] LoRA not loaded (LORA_PATH missing/empty)")

        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        processor.tokenizer.padding_side = "left"

        score_token_ids = [processor.tokenizer.encode(t, add_special_tokens=False)[-1] for t in TARGET_TOKENS]
        weights = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32).to(model.device)

        model.eval()

        # 最后做一次 meta 自检
        if any(getattr(pp, "is_meta", False) for pp in model.parameters()):
            raise RuntimeError("Model parameters contain meta tensors after loading. Check torch/transformers/peft setup.")

        model_ready = True
        print(f"✅ [System] Model ready on {CUDA_DEVICE}")
    except Exception as e:
        model_load_error = str(e)
        print(f"❌ [System] Model load failed: {model_load_error}")
        model_ready = False


def _extract_score_and_distribution(outputs, inputs, b: int) -> Dict:
    """Extract expected score from token-prob distribution for a single sample."""

    gen_ids = outputs.sequences[b][len(inputs.input_ids[b]) :]

    rating_token_pos = -1
    for pos, tid in enumerate(gen_ids):
        if tid.item() in score_token_ids and pos > 10:
            rating_token_pos = pos
            break

    if rating_token_pos == -1:
        # fall back
        return {
            "rating_token_found": False,
            "score": 5.0,
            "distribution": {t: 0.0 for t in TARGET_TOKENS},
        }

    logits = outputs.scores[rating_token_pos][b]
    relevant_logits = logits[score_token_ids]
    probs = F.softmax(relevant_logits.float() / TEMPERATURE, dim=-1)

    score = torch.sum(probs * weights).item()
    probs_list = probs.tolist()
    dist = {t: round(float(probs_list[i]), 4) for i, t in enumerate(TARGET_TOKENS)}

    return {
        "rating_token_found": True,
        "score": float(score),
        "distribution": dist,
        "rating_token_pos": int(rating_token_pos),
    }


@app.post("/compare")
async def compare_images(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    if process_vision_info is None:
        return api_error(
            code="MISSING_DEPENDENCY",
            message="缺少 qwen_vl_utils.process_vision_info。建议在环境中安装 qwen-vl-utils 或加入 PYTHONPATH。",
            http_status=500,
            details={"import_error": _qwen_vl_utils_import_error},
        )

    if not model_ready or model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未就绪")

    # Validate
    for f, name in [(image_a, "image_a"), (image_b, "image_b")]:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{name} 不是图片")

    try:
        img_a = Image.open(io.BytesIO(await image_a.read())).convert("RGB")
        img_b = Image.open(io.BytesIO(await image_b.read())).convert("RGB")

        # 1) Ask model to compare (brief explanation + winner)
        # NOTE: We request a brief explanation; model may or may not reveal hidden chain-of-thought.
        compare_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_a, "max_pixels": MAX_PIXELS},
                    {"type": "image", "image": img_b, "max_pixels": MAX_PIXELS},
                    {
                        "type": "text",
                        "text": (
                            "Compare the aesthetics of Image A and Image B. "
                            "Give a brief rationale (2-4 sentences) focusing on composition, lighting, and subject clarity. "
                            "Then output a single line: Winner: A or Winner: B or Winner: tie."
                        ),
                    },
                ],
            }
        ]

        compare_text = processor.apply_chat_template(compare_messages, tokenize=False, add_generation_prompt=True)
        compare_image_inputs, compare_video_inputs = process_vision_info(compare_messages)
        compare_inputs = processor(
            text=[compare_text],
            images=compare_image_inputs,
            videos=compare_video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            cmp_out = model.generate(
                **compare_inputs,
                max_new_tokens=220,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=False,
            )

        cmp_gen_ids = cmp_out.sequences[0][len(compare_inputs.input_ids[0]) :]
        reasoning = processor.decode(cmp_gen_ids, skip_special_tokens=True)

        winner = "tie"
        low = reasoning.lower()
        if "winner:" in low:
            if "winner: a" in low:
                winner = "A"
            elif "winner: b" in low:
                winner = "B"
            elif "winner: tie" in low:
                winner = "tie"

        # 2) Score each image independently (score distribution)
        sample_a = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_a, "max_pixels": MAX_PIXELS},
                    {
                        "type": "text",
                        "text": (
                            "Analyze the aesthetics briefly, then provide Rating Level from 1 to 10. "
                            "Format: Analysis: ... Rating Level: [score]"
                        ),
                    },
                ],
            }
        ]
        sample_b = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_b, "max_pixels": MAX_PIXELS},
                    {
                        "type": "text",
                        "text": (
                            "Analyze the aesthetics briefly, then provide Rating Level from 1 to 10. "
                            "Format: Analysis: ... Rating Level: [score]"
                        ),
                    },
                ],
            }
        ]

        score_texts = [
            processor.apply_chat_template(sample_a, tokenize=False, add_generation_prompt=True),
            processor.apply_chat_template(sample_b, tokenize=False, add_generation_prompt=True),
        ]
        # process_vision_info expects a list of per-sample messages
        score_image_inputs, score_video_inputs = process_vision_info([sample_a, sample_b])

        score_inputs = processor(
            text=score_texts,
            images=score_image_inputs,
            videos=score_video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **score_inputs,
                max_new_tokens=160,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        a = _extract_score_and_distribution(out, score_inputs, b=0)
        b = _extract_score_and_distribution(out, score_inputs, b=1)

        score_a = a["score"]
        score_b = b["score"]
        score_diff = float(score_a - score_b)

        # 兜底：若模型未明确给出 Winner，则按分数差判定
        if "winner:" not in low:
            if abs(score_diff) < 0.15:
                winner = "tie"
            elif score_diff > 0:
                winner = "A"
            else:
                winner = "B"

        return api_success(
            {
                "winner": winner,
                "score_a": round(score_a, 4),
                "score_b": round(score_b, 4),
                "score_diff": round(score_diff, 4),
                "distribution_a": a["distribution"],
                "distribution_b": b["distribution"],
                "reasoning": reasoning,
                "temperature": TEMPERATURE,
            }
        )

    except Exception as e:
        return api_error(code="INFERENCE_ERROR", message=str(e), http_status=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
