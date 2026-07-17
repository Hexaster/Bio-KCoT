import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# 目标文件路径 (也就是您刚才生成的，或者原始的 reasoning 文件)
input_file_path = str(path("paths.data_dir") / "withKG" / "sft" / "reactome_reasoning_train_sft_reasoning.csv")

# 输出文件路径 (去重后的文件)
output_file_path = str(path("paths.data_dir") / "withKG" / "sft" / "reactome_reasoning_train_sft_reasoning_deduplicated.csv")


def remove_exact_duplicates(in_path, out_path):
    print(f"正在读取文件: {in_path}")
    try:
        # 读取 CSV
        df = pd.read_csv(in_path)
        original_rows = len(df)
        print(f"原始行数: {original_rows}")

        # 【核心步骤】去重
        # keep='first' 表示保留第一次出现的行，删除后面重复的
        # 不填写 subset 参数，默认就是检查“所有列”
        df_dedup = df.drop_duplicates(keep='first')

        final_rows = len(df_dedup)
        removed_rows = original_rows - final_rows

        print("-" * 30)
        if removed_rows == 0:
            print("结果: 文件中不存在完全重复的行。")
        else:
            print(f"结果: 发现了 {removed_rows} 行完全重复的数据并已删除。")
            print(f"剩余行数: {final_rows}")

        # 保存
        print(f"正在保存到: {out_path}")
        df_dedup.to_csv(out_path, index=False)
        print("完成！")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {in_path}")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    remove_exact_duplicates(input_file_path, output_file_path)
