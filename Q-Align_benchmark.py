import pandas as pd
import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from paths import HF_HOME, FINAL_TEST_IMG_DIR, FINAL_TEST_CSV_PATH

# ================= 1. 路径与环境配置 =================
CACHE_DIR = HF_HOME
os.environ["HF_HOME"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

IMG_DIR = FINAL_TEST_IMG_DIR
CSV_PATH = FINAL_TEST_CSV_PATH

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
# ====================================================

def main():
    # 1. 加载模型
    print(f"正在加载 Q-Align (OneAlign) 预训练模型...")
    print(f"模型将被下载/缓存至: {CACHE_DIR}")
    
    try:
        # 使用 trust_remote_code=True 加载 Q-Align 的自定义代码
        model = AutoModelForCausalLM.from_pretrained(
            "q-future/one-align", 
            trust_remote_code=True, 
            torch_dtype=torch.float16, 
            device_map="auto",
            cache_dir=CACHE_DIR  # 显式指定缓存目录
        )
        model.eval() # 设置为评估模式
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("提示: 请确认网络连接正常，且已安装指定版本的 transformers。")
        return

    # 2. 读取 CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到 CSV 文件: {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    # 强制将 ID 转为字符串，避免文件名匹配失败
    df['ID'] = df['ID'].astype(str)
    
    gt_scores = []
    pred_scores = []

    print(f"🚀 开始推理 {len(df)} 张图片...")
    
    # 3. 推理循环
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        # 根据 ID 拼接文件名，注意 Linux 对大小写敏感
        img_name = f"{row['ID']}.jpg"
        img_path = os.path.join(IMG_DIR, img_name)
        
        if not os.path.exists(img_path):
            # 如果 .jpg 找不到，尝试 .JPG
            img_path_alt = os.path.join(IMG_DIR, f"{row['ID']}.JPG")
            if os.path.exists(img_path_alt):
                img_path = img_path_alt
            else:
                continue
            
        try:
            img = Image.open(img_path).convert("RGB")
            
            # 使用官方内置接口进行打分
            with torch.no_grad():
                # task_="aesthetics" 是美学评估任务
                # input_="image" 是图像模式
                score_tensor = model.score([img], task_="aesthetics", input_="image")
                
                # 处理可能的返回类型（可能是 tensor 标量或数组）
                if isinstance(score_tensor, torch.Tensor):
                    score = score_tensor.cpu().item()
                else:
                    score = float(score_tensor[0])
            
            pred_scores.append(score)
            gt_scores.append(float(row['score']))
            
        except Exception as e:
            print(f"\n⚠️ 处理图片 {row['ID']} 出错: {e}")

    # 4. 计算指标 (Spearman Rank Correlation)
    print("\n" + "-"*40)
    if len(pred_scores) > 1:
        srcc, _ = spearmanr(gt_scores, pred_scores)
        plcc, _ = pearsonr(gt_scores, pred_scores)

        print(f"✅ Q-Align Baseline 复现完成!")
        print(f"有效样本数: {len(pred_scores)}")
        print(f"SRCC (排名相关系数): {srcc:.4f}")
        print(f"PLCC (线性相关系数): {plcc:.4f}")
        print("-"*40)
        print("请记录上述 SRCC 数值，作为后续微调模型的对比基准。")
    else:
        print("❌ 错误: 成功处理的图片样本数过少，无法计算统计指标。")

if __name__ == "__main__":
    main()