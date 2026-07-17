import pandas as pd
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# =================配置区域=================
# 1. 定义输入文件列表
split_dir = path("paths.train_data_dir") / "sft_rl_aggressive_split"
input_files = [str(split_dir / filename) for filename in [
    "disease-indication_train_rl.csv",
    "drug-synergy_train_rl.csv",
    "PPI_reasoning_train_rl.csv",
    "reactome_reasoning_train_rl.csv",
]]

# 2. 定义输出文件路径 (建议以 .jsonl 结尾)
output_file = str(path("paths.rl_dataset", env="RL_DATASET"))
output_dir = os.path.dirname(output_file)

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 3. 定义 System Prompt (完全按照你的要求)
SYSTEM_PROMPT = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

# =================处理逻辑=================

all_datasets = []

print(f"开始处理数据，目标格式为 Unsloth GRPO Chat 格式...")

for file_path in input_files:
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 文件不存在，跳过: {file_path}")
        continue

    try:
        # 读取 CSV
        df = pd.read_csv(file_path)

        # 检查必要的列是否存在 (explanation 不是必须的，但如果有最好保留)
        # 注意：这里我们放宽限制，只要有 question 和 answer 即可
        if 'question' not in df.columns or 'answer' not in df.columns:
            print(f"⚠️ 警告: 文件 {os.path.basename(file_path)} 缺少 question 或 answer 列，跳过。")
            continue


        # --- 核心修改：构造 Prompt 格式 ---

        def format_chat_prompt(row):
            """将每一行转换为 Chat 列表格式"""
            question_text = str(row['question'])

            # 构造 User content：问题 + 你的指令后缀
            user_content = f"{question_text}\nAnswer the question and think step by step."

            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]


        # 1. 生成 prompt 列 (List of Dicts)
        df['prompt'] = df.apply(format_chat_prompt, axis=1)

        # 2. 重命名/提取 answer 列 (保持为 answer)
        # 如果 csv 里本来就是 answer，这行其实可以省略，为了保险起见
        if 'answer' in df.columns:
            df['answer'] = df['answer'].astype(str)  # 确保是字符串

        # 3. 提取并重命名 evidence_KG -> evidence_graph
        if 'evidence_KG' in df.columns:
            df['evidence_graph'] = df['evidence_KG']
        else:
            # 如果没有这一列，填入空字符串防止报错
            df['evidence_graph'] = ""

        # 4. 筛选最终需要的列
        # prompt: 用于模型输入
        # answer: 用于 reward function 对比 (ground truth)
        # evidence_graph: 用于辅助 (metadata)
        # question: 保留原始问题方便 debug
        df_final = df[['prompt', 'answer', 'evidence_graph', 'question']]

        # 添加到列表
        all_datasets.append(df_final)
        print(f"✅ 已加载: {os.path.basename(file_path)} (行数: {len(df_final)})")

    except Exception as e:
        print(f"❌ 处理文件 {file_path} 时出错: {e}")

# =================保存逻辑=================

if all_datasets:
    merged_df = pd.concat(all_datasets, ignore_index=True)

    # 转换为 JSONL 格式
    # orient='records', lines=True 是标准的 JSONL 格式
    merged_df.to_json(output_file, orient='records', lines=True, force_ascii=False)

    print("-" * 30)
    print(f"🎉 转换成功！")
    print(f"总数据量: {len(merged_df)} 条")
    print(f"保存路径: {output_file}")

    # 打印一条样例数据供检查结构
    print("\n样例数据结构 (First Item):")
    sample = merged_df.iloc[0].to_dict()
    print(json.dumps(sample, indent=2, ensure_ascii=False))
else:
    print("❌ 没有数据被处理。")
