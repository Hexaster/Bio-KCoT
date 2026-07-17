import os
import glob
import pandas as pd
import torch
import re
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import get, path

print("=== Script started successfully ===")

# ================= 配置区域 =================
# 1. 模型路径配置
# 基座模型路径
BASE_MODEL_PATH = get("models.qwen3_14b", env="QWEN3_14B_BASE_MODEL")

# 2. 数据路径配置
# 测试文件所在的文件夹
TEST_DATA_DIR = str(path("paths.test_data_dir", env="TEST_DATA_DIR"))

# 结果保存的文件夹
OUTPUT_DIR = str(path("paths.baseline_results") / "Qwen3-14b")

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

    print("✅ Base Model loaded successfully!")
    model.eval()  
    return tokenizer, model


def generate_response(model, tokenizer, question):
    """
    生成回答，保留思考过程标签 <think> 和 <answer>
    """
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

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response_text


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: {OUTPUT_DIR}")

    tokenizer, model = setup_model()

    test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))

    if not test_files:
        print(f"❌ No CSV files found in {TEST_DATA_DIR}")
        return

    print(f"📂 Found {len(test_files)} files to process.")

    for file_path in test_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        print(f"\nProcessing: {file_name} -> {output_path}")

        try:
            # === 断点续传核心逻辑 ===
            # 如果输出文件已经存在，就读取输出文件继续跑；如果不存在，读取原始测试文件
            if os.path.exists(output_path):
                print(f"⚠️ Found existing output file, attempting to resume...")
                df = pd.read_csv(output_path)
            else:
                df = pd.read_csv(file_path)

            if 'question' not in df.columns:
                print(f"⚠️ Column 'question' not found in {file_name}, skipping.")
                continue

            # 确保 dataframe 里有一列用来存结果
            if 'model_response' not in df.columns:
                df['model_response'] = None

            # 统计已经跑了多少条，没跑的还有多少条
            total_rows = len(df)
            processed_mask = df['model_response'].notna() & (df['model_response'] != "")
            processed_count = processed_mask.sum()

            if processed_count == total_rows:
                print(f"✅ File {file_name} is already fully processed. Skipping.")
                continue
            else:
                print(f"▶️ Resuming task: {processed_count}/{total_rows} already processed.")

            # 遍历每一行进行推理
            for idx, row in tqdm(df.iterrows(), total=total_rows, desc=f"Generating {file_name}"):
                
                # 如果这一行已经有结果了，直接跳过（实现断点续传）
                if pd.notna(row.get('model_response')) and str(row.get('model_response')).strip() != "":
                    continue

                question = str(row['question'])

                try:
                    response = generate_response(model, tokenizer, question)
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    response = f"Error: {e}"

                # === 实时保存核心逻辑 ===
                # 更新当前行的结果
                df.at[idx, 'model_response'] = response
                
                # 立刻保存到硬盘，覆盖原有的 output_path
                df.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"✅ Finished processing {file_name}")

        except Exception as e:
            print(f"❌ Failed to process file {file_name}: {e}")

    print("\n🎉 All base model tasks completed!")


if __name__ == "__main__":
    main()
