from unsloth import FastModel # 如果報錯，請嘗試改用 FastLanguageModel
import torch
import json
from tqdm import tqdm
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# 1. DEFINE PATHS
# 指向您的 GRPO 訓練後的 8B LoRA 權重
lora_model_path = str(path("paths.grpo_kg_checkpoint", env="GRPO_KG_CHECKPOINT"))

# 【修改點 1】：將輸入資料集換成抽樣後的 BioMaze json 檔案 (此處以 20 為例，如果是 200 請自行修改檔名)
input_data_path = str(path("paths.ood_biomaze") / "biomaze_sampled_200.json")
output_results_path = str(path("paths.ood_biomaze") / "biomaze_openended_results_8B_rl.json")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_results_path), exist_ok=True)

# =============================================================================
# 2. LOAD MODEL AND TOKENIZER (USING UNSLOTH)
# =============================================================================
print("--- 正在使用 Unsloth FastModel 載入模型 ---")
model, tokenizer = FastModel.from_pretrained(
    model_name = lora_model_path,
    max_seq_length = 4096,
    load_in_4bit = False,
)
print("--- 模型載入完成 ---")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 建議在使用 Unsloth 進行推論時調用此方法（能讓生成速度翻倍），若您的版本不支援也可直接使用 model.eval()
try:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
except:
    model.eval()
# =============================================================================


# 3. GENERATION FUNCTION
def generate_answer(question):
    """
    Generates an open-ended answer from the model given a question.
    Returns the raw generated text directly.
    """
    # 【修改點 2】：移除 MCQ 選項，改為單純的問答 Instruction
    system_prompt = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""
    instruction = "Answer the following biological question in detail and think step by step.\n"

    content = f"{instruction}\nQuestion: {question}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=True,      # 保持您原本的設定
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the full response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 【修改點 3】：直接返回完整的 generated_text，不再做正則提取或 \boxed{} 處理
    return generated_text


# 4. EVALUATION SCRIPT
def evaluate_dataset(file_path):
    """
    Loads the JSON dataset, runs model inference, and saves the full raw answers.
    """
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {e}")
        return

    results_data = []

    # 【修改點 4】：移除正確率計算邏輯，單純跑迴圈生成並保存
    for item in tqdm(data, desc=f"Processing {os.path.basename(file_path)}"):
        question = item.get('question', '')

        if not question:
            continue

        prediction_text = generate_answer(question)

        # 截斷打印，防止在終端機刷屏影響觀察
        print(f"\nQuestion: {question[:100]}...")
        print(f"Model Prediction: {prediction_text[:150]}...\n[...truncated for console...]")

        # 將完整預測結果存入 JSON 的新欄位中
        item['model_prediction'] = prediction_text
        results_data.append(item)

    # Save the detailed results to a new JSON file
    with open(output_results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"\nInference complete! Detailed results saved to {output_results_path}")


# Run the script
if __name__ == "__main__":
    evaluate_dataset(input_data_path)
