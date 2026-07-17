import pandas as pd
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# --- 配置 ---
sft_input_dir = str(path("paths.train_data_dir") / "sft_rl_aggressive_split")
raw_reasoning_dir = str(path("paths.data_dir") / "withKG" / "sft2")
output_json_path = os.path.join(sft_input_dir, "sft_KG_data2.json")

file_map = {
    "disease-indication_train_sft.csv": "disease-indication_train_sft_reasoning.csv",
    "drug-synergy_train_sft.csv": "drug-synergy_train_sft_reasoning.csv",
    "PPI_reasoning_train_sft.csv": "PPI_reasoning_train_sft_reasoning.csv",
    "reactome_reasoning_train_sft.csv": "reactome_reasoning_train_sft_reasoning.csv"
}

SYSTEM_PROMPT = "Respond in the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>\n"
INSTRUCTION_TEXT = "Answer the question and think step by step.\n"


def generate_sft_json_debug():
    all_sft_data = []

    print(f"开始生成 SFT JSON (带详细 Debug)...")

    for sft_file, raw_file in file_map.items():
        sft_path = os.path.join(sft_input_dir, sft_file)
        raw_path = os.path.join(raw_reasoning_dir, raw_file)

        if not os.path.exists(sft_path) or not os.path.exists(raw_path):
            continue

        print(f"处理: {sft_file}")

        try:
            # 1. 读取
            df_sft = pd.read_csv(sft_path)
            cols_to_load = ['question', 'cot_raw_think', 'answer']
            df_raw = pd.read_csv(raw_path, usecols=lambda c: c in cols_to_load)

            # 2. 标准化 Question 列 (转字符串 + 去除首尾空格)
            df_sft['question_clean'] = df_sft['question'].astype(str).str.strip()
            df_raw['question_clean'] = df_raw['question'].astype(str).str.strip()

            # 3. 合并
            merged_df = pd.merge(
                df_sft[['question', 'question_clean']],
                df_raw[['question_clean', 'cot_raw_think', 'answer']],
                on='question_clean',
                how='inner'
            )

            # 去重
            merged_df = merged_df.drop_duplicates(subset=['question_clean'])

            print(f"  - SFT原行数: {len(df_sft)}")
            print(f"  - 匹配后行数: {len(merged_df)}")

            # -----------------------------------------------------------
            # [新增] 这里是打印丢失数据的核心逻辑
            # -----------------------------------------------------------
            if len(df_sft) != len(merged_df):
                diff = len(df_sft) - len(merged_df)
                print(f"  ⚠️ 依然有 {diff} 条数据未匹配！")
                print("  🔍 [Debug] 打印未匹配的 Question 内容:")

                # 找出哪些 question_clean 在 merged_df 里不存在
                matched_questions = set(merged_df['question_clean'])
                missing_mask = ~df_sft['question_clean'].isin(matched_questions)
                missing_rows = df_sft[missing_mask]

                for idx, row in missing_rows.iterrows():
                    q_raw = row['question']
                    q_clean = row['question_clean']
                    print(f"    --------------------------------------------------")
                    print(f"    行号: {idx}")
                    # 使用 repr() 打印，这样如果里面有换行符 \n 或引号，都能看见
                    print(f"    原始内容 (repr): {repr(q_raw)}")
                    print(f"    清洗后内容 (clean): {repr(q_clean)}")

                    # 尝试在 Raw 数据里找个最像的，看看是不是还差了一点点
                    # 比如中间有双空格，或者句号不一样
                    # 这里做一个简单的包含测试
                    print(f"    [尝试匹配 Raw 数据...]")
                    potential_match = df_raw[df_raw['question_clean'].str.contains(q_clean[:20], regex=False)]
                    if not potential_match.empty:
                        print(f"      -> 在 Raw 中找到了 {len(potential_match)} 条开头相似的数据:")
                        for _, r_row in potential_match.head(1).iterrows():
                            print(f"      -> Raw 中的样子: {repr(r_row['question_clean'])}")
                    else:
                        print(f"      -> Raw 中似乎完全找不到相似开头的数据。")
            # -----------------------------------------------------------

            # 4. 生成 JSON (只处理匹配上的)
            for _, row in merged_df.iterrows():
                question = row['question'].strip()
                cot = str(row.get('cot_raw_think', '')).strip()
                answer = str(row.get('answer', '')).strip()

                if not cot or not answer or cot == 'nan' or answer == 'nan':
                    continue

                entry = {
                    "instruction": INSTRUCTION_TEXT,
                    "input": question,
                    "output": f"<think>\n{cot}\n</think>\n<answer>\n{answer}\n</answer>",
                    "system": SYSTEM_PROMPT
                }
                all_sft_data.append(entry)

        except Exception as e:
            print(f"  处理错误: {e}")

    # 保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_sft_data, f, ensure_ascii=False, indent=2)
    print(f"\n保存完成: {output_json_path}")
    print(f"总条数: {len(all_sft_data)}")


if __name__ == "__main__":
    generate_sft_json_debug()
