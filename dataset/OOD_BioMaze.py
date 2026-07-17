from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
from tqdm import tqdm
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import get, path

# 1. DEFINE PATHS
# Define model paths
base_model_path = get("models.biomistral_7b", env="BIOMISTRAL_7B_BASE_MODEL")

# Define data and output paths
input_data_path = str(path("paths.ood_biomaze") / "biomaze_sampled_200.json")
output_results_path = str(path("paths.ood_biomaze") / "biomaze_openended_results_biomistral-7b.json")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_results_path), exist_ok=True)

# 2. LOAD MODEL AND TOKENIZER
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Base Model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("Loading LoRA Adapter...")
# model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()  # Set model to evaluation mode


# 3. GENERATION FUNCTION
def generate_answer(question):
    """
    Generates an open-ended answer from the model given a question.
    Returns the raw generated text directly.
    """
    # 保留 System Prompt 引导模型先思考再作答，保证推理质量
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
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the full response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 【修改点 1】：直接返回完整的 generated_text，不再做正则提取
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

    for item in tqdm(data, desc=f"Processing {os.path.basename(file_path)}"):
        question = item.get('question', '')

        if not question:
            continue

        # 【修改点 2】：只接收单个返回值（完整文本）
        prediction_text = generate_answer(question)

        # Print snippets to monitor progress in the terminal
        # 截断打印，防止在控制台刷屏影响观察
        print(f"\nQuestion: {question[:100]}...")
        print(f"Model Prediction: {prediction_text[:150]}...\n[...truncated for console...]")

        # 【修改点 3】：只存储完整的预测结果，移除了多余的 extracted 字段
        item['model_prediction'] = prediction_text
        results_data.append(item)

    # Save the detailed results to a new JSON file
    with open(output_results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"\nInference complete! Detailed results saved to {output_results_path}")


# Run the script
if __name__ == "__main__":
    evaluate_dataset(input_data_path)
