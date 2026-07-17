import os
import glob
import pandas as pd
import torch
import re
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import get, path

# ================= 配置区域 =================
# 1. 模型路径配置
# 基座模型路径 (请确认你的基座模型还在这个位置)
BASE_MODEL_PATH = get("models.qwen3_8b", env="QWEN3_8B_BASE_MODEL")
# SFT 微调后的模型路径 (LoRA Adapter 路径)
SFT_ADAPTER_PATH = str(path("paths.sft_kg_adapter", env="SFT_KG_ADAPTER"))

# 2. 数据路径配置
# 测试文件所在的文件夹
TEST_DATA_DIR = str(path("paths.test_data_dir", env="TEST_DATA_DIR"))

# 结果保存的文件夹
OUTPUT_DIR = str(path("paths.dataset_results") / "KG_clean2" / "qwen3-8b")

# 3. 生成参数
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.9


# ===========================================

def setup_model():
    print(f"🔄 Loading Tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🔄 Loading Base Model from: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 加载 LoRA 微调权重
    # 如果 SFT_ADAPTER_PATH 是一个有效的 LoRA 目录，则加载它
    if os.path.exists(SFT_ADAPTER_PATH):
        print(f"🔄 Loading SFT Adapter (LoRA) from: {SFT_ADAPTER_PATH}")
        try:
            model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
            print("✅ SFT Adapter loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load adapter (maybe it's a full model?): {e}")
            print("Running with base model only or checking path...")
    else:
        print(f"⚠️ SFT Path does not exist: {SFT_ADAPTER_PATH}, running Base Model only.")

    model.eval()  # 设置为评估模式
    return tokenizer, model


def generate_response(model, tokenizer, question):
    """
    生成回答，保留思考过程标签 <think> 和 <answer>
    """
    # 构造 Prompt
    system_prompt = (
        "Respond in the following format:\n"
        "<think>\n...\n</think>\n"
        "<answer>\n...\n</answer>\n"
        "Answer the question and think step by step.\n"
    )

    user_content = f"Question: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # 应用 Chat Template
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,  # 开启采样，增加多样性
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码输出 (只取生成的这部分)
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response_text


def main():
    # 0. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: {OUTPUT_DIR}")

    # 1. 加载模型
    tokenizer, model = setup_model()

    # 2. 获取测试文件列表
    # 假设是 csv 文件，如果不是请修改后缀
    test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))

    if not test_files:
        print(f"❌ No CSV files found in {TEST_DATA_DIR}")
        return

    print(f"📂 Found {len(test_files)} files to process.")

    # 3. 循环处理每个文件
    for file_path in test_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        print(f"\nProcessing: {file_name} -> {output_path}")

        # 检查是否已处理 (简单的断点续传: 文件存在则跳过)
        # 如果你想强制重跑，可以注释掉这几行
        if os.path.exists(output_path):
            print(f"⚠️ Output file exists, skipping: {file_name}")
            continue

        try:
            df = pd.read_csv(file_path)

            # 检查是否有 question 列
            if 'question' not in df.columns:
                print(f"⚠️ Column 'question' not found in {file_name}, skipping.")
                continue

            # 存储结果列表
            results = []

            # 遍历每一行进行推理
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {file_name}"):
                question = str(row['question'])

                try:
                    response = generate_response(model, tokenizer, question)
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    response = "Error during generation"

                results.append(response)

            # 将结果保存到新的一列
            df['model_response'] = results

            # 保存文件
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Saved results to {output_path}")

        except Exception as e:
            print(f"❌ Failed to process file {file_name}: {e}")

    print("\n🎉 All tasks completed!")


if __name__ == "__main__":
    main()
