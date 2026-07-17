import os
import glob
import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# ==========================================
# 1. 配置区域
# ==========================================

# GRPO 训练后的模型路径 (Unsloth 会自动读取 adapter_config.json 找到基座模型)
# 这里填你训练结束后的保存路径
MODEL_PATH = str(path("paths.grpo_kg_checkpoint", env="GRPO_KG_CHECKPOINT"))

# 测试数据文件夹 (假设里面是 .csv 文件)
TEST_DATA_DIR = str(path("paths.test_data_dir", env="TEST_DATA_DIR"))

# 结果保存文件夹
OUTPUT_DIR = str(path("paths.dataset_results") / "GRPO" / "qwen3-8b-kg")

# 生成参数
MAX_NEW_TOKENS = 4096  # 思考过程通常很长，给大一点
TEMPERATURE = 0.6  # 稍微降低一点温度，让思考更稳定
TOP_P = 0.9


# ==========================================
# 2. 模型加载 (Unsloth 方式)
# ==========================================

def load_model():
    print(f"🔄 正在加载模型: {MODEL_PATH}")

    # max_seq_length 需要足够大以容纳 Prompt + 思考 + 回答
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=5120,
        dtype=None,  # 自动检测 float16/bfloat16
        load_in_4bit=False,  # 推荐 True，推理速度快且省显存；如果显存极其充裕可改 False
    )

    # 启用原生 2 倍速推理加速
    FastLanguageModel.for_inference(model)

    print("✅ 模型加载完成！")
    return model, tokenizer


# ==========================================
# 3. 推理函数
# ==========================================

def generate_response(model, tokenizer, question):
    """
    生成完整的回答，包含 <think> 标签的内容
    """
    # 1. 构造 Prompt
    # 必须加上这个 System Prompt，否则模型可能忘记输出 <think>
    system_prompt = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
Answer the question and think step by step.
"""
    user_content = f"Question: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # 2. 应用 Chat Template
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # 3. 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_cache=True
        )

    # 4. 解码 (只保留新生成的部分)
    # output[0] 包含 input + generated，我们需要切片只取 generated
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response_text


# ==========================================
# 4. 主程序 (文件处理循环)
# ==========================================

def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 加载模型
    model, tokenizer = load_model()

    # 获取所有 csv 文件
    csv_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))

    if not csv_files:
        print(f"⚠️ 在 {TEST_DATA_DIR} 没有找到 CSV 文件")
        return

    print(f"📂 找到 {len(csv_files)} 个文件待处理...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, file_name)

        print(f"\n🚀 正在处理: {file_name}")

        try:
            # 读取数据
            df = pd.read_csv(file_path)

            # 检查是否有 question 列 (如果没有，尝试找第一列)
            if 'question' not in df.columns:
                print(f"⚠️ 列名中没找到 'question'，尝试使用第一列作为问题...")
                question_col = df.columns[0]
            else:
                question_col = 'question'

            results = []

            # 遍历每一行
            # desc=file_name 显示进度条
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=file_name):
                q = str(row[question_col])

                try:
                    # 调用推理
                    output = generate_response(model, tokenizer, q)
                except Exception as e:
                    print(f"❌ Error at index {idx}: {e}")
                    output = "GENERATION_ERROR"

                results.append(output)

            # 将结果保存到新的一列 'model_output'
            df['model_output'] = results

            # 保存回 CSV
            # encoding='utf-8-sig' 防止 Excel 打开乱码
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"💾 已保存结果到: {output_path}")

        except Exception as e:
            print(f"❌ 处理文件 {file_name} 失败: {e}")

    print("\n🎉 所有任务完成！")


if __name__ == "__main__":
    main()
