import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# 1. 定义源文件路径列表
score_dir = path("paths.data_dir") / "score"
source_files = [str(score_dir / filename) for filename in [
    "disease-indication-drug_synthesized_scored.csv",
    "drug-synergy-drug_scored.csv",
    "PPI_reasoning_questions_scored.csv",
    "reactome_reasoning_dataset_scored.csv",
]]

# 2. 定义输出目录
output_dir = str(path("paths.data_dir") / "withKG" / "filter")


def filter_high_score_data():
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")

    print("-" * 50)

    for file_path in source_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        print(f"正在处理文件: {filename}")

        try:
            # 读取 CSV
            df = pd.read_csv(file_path)

            # 检查是否有 score 列
            if 'score' not in df.columns:
                print(f"  [警告] 跳过: 文件中未找到 'score' 列")
                continue

            # 转换为数值类型，无法转换的变为 NaN
            df['score'] = pd.to_numeric(df['score'], errors='coerce')

            # 统计原始数量
            original_count = len(df)

            # 筛选分数 >= 8 的数据
            filtered_df = df[df['score'] >= 8]
            filtered_count = len(filtered_df)

            # 保存筛选后的文件
            filtered_df.to_csv(output_path, index=False)

            print(f"  原始数量: {original_count}")
            print(f"  筛选后数量 (>=8): {filtered_count}")
            print(f"  已保存至: {output_path}")

        except FileNotFoundError:
            print(f"  [错误] 找不到文件: {file_path}")
        except Exception as e:
            print(f"  [错误] 处理失败: {e}")

        print("-" * 50)


if __name__ == "__main__":
    filter_high_score_data()
